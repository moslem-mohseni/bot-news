"""
ماژول کلاس پایه مدل برای CryptoNewsBot

این ماژول کلاس پایه مشترک برای تمام مدل‌های هوش مصنوعی را تعریف می‌کند که قابلیت‌های
زیر را برای همه مدل‌ها فراهم می‌سازد:

- مدیریت بهینه حافظه و منابع سیستمی
- بارگیری تنبل (Lazy Loading) مدل‌ها
- کش‌گذاری هوشمند نتایج
- پردازش خطاها و مکانیسم تلاش مجدد
- یادگیری و بهبود مستمر
- مانیتورینگ عملکرد و منابع
- مدیریت سازگار با CPU و GPU
"""

import os
import psutil
import gc
import hashlib
import time
import json
import functools
import threading
import traceback
import logging
from datetime import datetime
from typing import Any, Dict, List, Set, Optional, Tuple, Union, Callable, TypeVar, Generic
from contextlib import contextmanager

import numpy as np
import torch
from transformers import logging as transformers_logging

from ..utils.config import Config
from ..utils.logger import Logger
from ..cache.redis_manager import RedisManager
from .data_repository import DataRepository

# تنظیم سطح لاگ برای transformers
transformers_logging.set_verbosity_error()

# تعریف نوع عمومی برای برگشتی تابع‌ها
T = TypeVar('T')

# ثابت‌ها
DEFAULT_CACHE_EXPIRE = 86400  # 1 روز (ثانیه)
DEFAULT_IDLE_THRESHOLD = 10 * 60  # 10 دقیقه (ثانیه)
DEFAULT_MEMORY_THRESHOLD = 85  # حداکثر استفاده از حافظه (درصد)
MAX_RETRY_ATTEMPTS = 3  # حداکثر تعداد تلاش مجدد
RETRY_BACKOFF_FACTOR = 2  # ضریب افزایش زمان انتظار بین تلاش‌ها
RETRY_INITIAL_WAIT = 1  # زمان انتظار اولیه (ثانیه)
MEMORY_CHECK_INTERVAL = 30  # فاصله زمانی بررسی وضعیت حافظه (ثانیه)
BATCH_SIZE = 16  # اندازه پیش‌فرض دسته پردازش
MODEL_PERFORMANCE_LOG_INTERVAL = 100  # فاصله زمانی ثبت آمار عملکرد (تعداد فراخوانی)


def retry_on_error(max_attempts=MAX_RETRY_ATTEMPTS,
                   backoff_factor=RETRY_BACKOFF_FACTOR,
                   initial_wait=RETRY_INITIAL_WAIT,
                   exceptions=(Exception,)):
    """
    دکوراتور برای تلاش مجدد در صورت بروز خطا با زمان انتظار نمایی

    Args:
        max_attempts: حداکثر تعداد تلاش
        backoff_factor: ضریب افزایش زمان انتظار
        initial_wait: زمان انتظار اولیه (ثانیه)
        exceptions: انواع استثنائات برای تلاش مجدد

    Returns:
        نتیجه تابع تزئین شده
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            wait_time = initial_wait

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        # لاگ خطا
                        if hasattr(args[0], 'logger'):
                            args[0].logger.warning(
                                f"خطا در اجرای {func.__name__} (تلاش {attempt}/{max_attempts}): {str(e)}. "
                                f"تلاش مجدد پس از {wait_time} ثانیه..."
                            )
                        # انتظار
                        time.sleep(wait_time)
                        # افزایش زمان انتظار برای تلاش بعدی
                        wait_time *= backoff_factor

            # اگر تمام تلاش‌ها ناموفق باشد، پرتاب آخرین استثنا
            if hasattr(args[0], 'logger'):
                args[0].logger.error(
                    f"تمام تلاش‌ها برای اجرای {func.__name__} ناموفق بود. "
                    f"آخرین خطا: {str(last_exception)}"
                )
            raise last_exception
        return wrapper
    return decorator


class ModelPerformanceTracker:
    """
    رهگیری و ثبت آمار عملکرد مدل‌ها
    """

    def __init__(self, model_name):
        """
        راه‌اندازی اولیه

        Args:
            model_name: نام مدل
        """
        self.model_name = model_name
        self.call_count = 0
        self.total_execution_time = 0
        self.min_execution_time = float('inf')
        self.max_execution_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors_count = 0

        self.logger = Logger(__name__).get_logger()
        self.last_log_time = time.time()

    def record_call(self, method_name, execution_time, from_cache=False, error=False):
        """
        ثبت اطلاعات یک فراخوانی

        Args:
            method_name: نام متد
            execution_time: زمان اجرا (ثانیه)
            from_cache: آیا از کش بارگیری شده؟
            error: آیا خطا رخ داده؟
        """
        self.call_count += 1

        if not from_cache:
            self.total_execution_time += execution_time
            self.min_execution_time = min(self.min_execution_time, execution_time)
            self.max_execution_time = max(self.max_execution_time, execution_time)

        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.errors_count += 1

        # لاگ دوره‌ای آمار عملکرد
        if self.call_count % MODEL_PERFORMANCE_LOG_INTERVAL == 0:
            self._log_performance_stats()

    def _log_performance_stats(self):
        """
        ثبت آمار عملکرد در لاگ
        """
        now = time.time()
        time_since_last_log = now - self.last_log_time

        if time_since_last_log < 60:  # حداقل 1 دقیقه بین لاگ‌ها
            return

        self.last_log_time = now

        if self.call_count == 0:
            return

        cache_hit_ratio = (self.cache_hits / self.call_count) * 100 if self.call_count > 0 else 0
        avg_execution_time = self.total_execution_time / self.cache_misses if self.cache_misses > 0 else 0

        # در صورتی که هنوز هیچ فراخوانی مستقیم نداشته‌ایم
        if self.min_execution_time == float('inf'):
            self.min_execution_time = 0

        self.logger.info(
            f"آمار عملکرد مدل {self.model_name}: "
            f"تعداد فراخوانی: {self.call_count}, "
            f"نسبت کش: {cache_hit_ratio:.1f}%, "
            f"میانگین زمان اجرا: {avg_execution_time:.3f}s, "
            f"حداقل: {self.min_execution_time:.3f}s, "
            f"حداکثر: {self.max_execution_time:.3f}s, "
            f"خطاها: {self.errors_count}"
        )


class ModelMemoryManager:
    """
    مدیریت حافظه برای مدل‌های هوش مصنوعی
    """

    def __init__(self, memory_threshold=DEFAULT_MEMORY_THRESHOLD):
        """
        راه‌اندازی اولیه

        Args:
            memory_threshold: آستانه استفاده از حافظه (درصد)
        """
        self.memory_threshold = memory_threshold
        self.logger = Logger(__name__).get_logger()
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self):
        """
        دریافت میزان استفاده فعلی از حافظه

        Returns:
            درصد استفاده از حافظه
        """
        memory_info = self.process.memory_info()
        memory_usage_percent = (memory_info.rss / psutil.virtual_memory().total) * 100
        return memory_usage_percent

    def is_memory_pressure(self):
        """
        بررسی وجود فشار حافظه

        Returns:
            True اگر فشار حافظه وجود داشته باشد، در غیر این صورت False
        """
        memory_usage = self.get_memory_usage()
        return memory_usage > self.memory_threshold

    def log_memory_status(self):
        """
        ثبت وضعیت حافظه در لاگ
        """
        memory_usage = self.get_memory_usage()
        self.logger.debug(f"میزان استفاده از حافظه: {memory_usage:.1f}%")

    def clear_memory(self):
        """
        تلاش برای آزادسازی حافظه
        """
        # فراخوانی جمع‌آوری زباله پایتون
        collected = gc.collect()

        # پاکسازی کش PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.debug(f"آزادسازی حافظه: {collected} شیء جمع‌آوری شد")


class BaseModel:
    """
    کلاس پایه برای تمام مدل‌های هوش مصنوعی

    این کلاس قابلیت‌های مشترک از جمله کش‌گذاری، بارگیری تنبل، مدیریت حافظه،
    پردازش خطاها و رهگیری عملکرد را برای همه مدل‌ها فراهم می‌کند.
    """

    _instances = {}  # ذخیره نمونه‌های Singleton
    _model_locks = {}  # قفل‌های همگام‌سازی برای بارگیری مدل

    # اولویت مدل‌ها برای تخلیه (مقدار کمتر = اولویت بالاتر برای نگهداری)
    _model_priorities = {}
    _global_model_lock = threading.RLock()  # قفل سراسری برای مدیریت مدل‌ها

    def __new__(cls, *args, **kwargs):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس

        Returns:
            نمونه منحصر به فرد کلاس
        """
        with cls._global_model_lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(BaseModel, cls).__new__(cls)
                cls._model_locks[cls] = threading.RLock()

                # تنظیم اولویت پیش‌فرض
                if cls not in cls._model_priorities:
                    cls._model_priorities[cls] = 10  # اولویت پیش‌فرض

            return cls._instances[cls]

    def __init__(self, model_name=None, priority=None) -> None:
        """
        راه‌اندازی اولیه کلاس پایه

        Args:
            model_name: نام مدل (اختیاری)
            priority: اولویت مدل برای مدیریت حافظه (اختیاری)
        """
        # اگر قبلاً راه‌اندازی شده، خروج
        if hasattr(self, 'initialized') and self.initialized:
            return

        # تنظیمات اولیه
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.data_repo = DataRepository()

        # تنظیم اولویت مدل
        if priority is not None:
            with self._global_model_lock:
                self._model_priorities[self.__class__] = priority

        # نام مدل
        self.model_name = model_name or self.__class__.__name__

        # مسیر کش مدل‌ها
        self.model_cache_dir = getattr(self.config, 'model_cache_dir', 'models/cache')
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir, exist_ok=True)

        # تنظیمات کش
        self.cache_prefix = f"model:{self.model_name}:"
        self.default_cache_expire = DEFAULT_CACHE_EXPIRE

        # رهگیری عملکرد
        self.performance_tracker = ModelPerformanceTracker(self.model_name)

        # مدیریت حافظه
        self.memory_manager = ModelMemoryManager()

        # وضعیت مدل
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_loaded = False
        self.last_used = time.time()
        self.unload_idle_time = self.config.model_idle_time if hasattr(self.config, 'model_idle_time') else DEFAULT_IDLE_THRESHOLD

        # تنظیمات مدل
        self.batch_size = getattr(self.config, 'model_batch_size', BATCH_SIZE)
        self.model_precision = getattr(self.config, 'model_precision', 'fp32')  # 'fp32', 'fp16', 'int8'
        self.model_config = {}

        # شناسایی دستگاه مناسب (CPU/GPU)
        self.device = self._get_device()

        # پیکربندی مدل
        self._setup_model()

        # راه‌اندازی تایمر تخلیه خودکار
        self._start_unload_timer()

        # راه‌اندازی مانیتورینگ حافظه
        self._start_memory_monitor()

        self.initialized = True
        self.logger.info(f"مدل {self.model_name} راه‌اندازی شد (دستگاه: {self.device.type})")

    def _get_device(self) -> torch.device:
        """
        تشخیص دستگاه مناسب برای اجرای مدل (CPU/GPU)

        تنظیمات محیطی می‌تواند استفاده از GPU را غیرفعال کند حتی اگر در دسترس باشد.
        همچنین در صورت نیاز، شناسه GPU خاصی را می‌توان انتخاب کرد.

        Returns:
            دستگاه مناسب
        """
        # بررسی تنظیمات برای غیرفعال کردن GPU
        force_cpu = getattr(self.config, 'force_cpu', False)

        if force_cpu:
            self.logger.info("استفاده اجباری از CPU براساس تنظیمات")
            return torch.device("cpu")

        if torch.cuda.is_available():
            # انتخاب GPU خاص
            gpu_id = getattr(self.config, 'gpu_id', 0)

            # بررسی معتبر بودن شناسه GPU
            if gpu_id >= torch.cuda.device_count():
                self.logger.warning(
                    f"شناسه GPU {gpu_id} نامعتبر است. "
                    f"از GPU 0 استفاده می‌شود."
                )
                gpu_id = 0

            self.logger.info(f"GPU یافت شد، استفاده از GPU {gpu_id}")
            return torch.device(f"cuda:{gpu_id}")
        else:
            self.logger.info("GPU یافت نشد، استفاده از CPU")
            return torch.device("cpu")

    def _setup_model(self) -> None:
        """
        پیکربندی اولیه مدل

        این متد در کلاس‌های فرزند باید پیاده‌سازی شود.
        """
        pass

    def _start_unload_timer(self) -> None:
        """
        راه‌اندازی تایمر تخلیه خودکار مدل برای صرفه‌جویی در حافظه
        """
        def check_and_unload():
            """تابع نظارت بر فعالیت مدل و تخلیه در صورت عدم استفاده"""
            while True:
                try:
                    # بررسی زمان بیکاری
                    if self.model_loaded and time.time() - self.last_used > self.unload_idle_time:
                        self.logger.debug(
                            f"مدل {self.model_name} بیش از {self.unload_idle_time} ثانیه "
                            f"بدون استفاده بوده است، تخلیه از حافظه..."
                        )
                        self.unload_model()
                except Exception as e:
                    self.logger.error(f"خطا در نظارت بر تخلیه مدل: {str(e)}")
                finally:
                    # خواب برای بررسی بعدی
                    time.sleep(30)  # بررسی هر 30 ثانیه

        # راه‌اندازی در یک نخ مجزا
        timer_thread = threading.Thread(target=check_and_unload, daemon=True)
        timer_thread.start()
        self.logger.debug(f"نظارت بر زمان بیکاری مدل {self.model_name} آغاز شد")

    def _start_memory_monitor(self) -> None:
        """
        راه‌اندازی مانیتورینگ حافظه برای مدیریت خودکار مدل‌ها
        """
        def monitor_memory():
            """تابع نظارت بر مصرف حافظه و تخلیه مدل‌ها در صورت نیاز"""
            while True:
                try:
                    # بررسی فشار حافظه
                    if self.memory_manager.is_memory_pressure():
                        self.logger.warning(
                            f"فشار حافظه تشخیص داده شد "
                            f"({self.memory_manager.get_memory_usage():.1f}% > "
                            f"{self.memory_manager.memory_threshold}%), "
                            f"تلاش برای آزادسازی..."
                        )

                        # آزادسازی حافظه
                        self._free_memory_pressure()
                except Exception as e:
                    self.logger.error(f"خطا در مانیتورینگ حافظه: {str(e)}")
                finally:
                    # خواب برای بررسی بعدی
                    time.sleep(MEMORY_CHECK_INTERVAL)

        # راه‌اندازی در یک نخ مجزا
        memory_thread = threading.Thread(target=monitor_memory, daemon=True)
        memory_thread.start()
        self.logger.debug(f"مانیتورینگ حافظه برای مدل {self.model_name} آغاز شد")

    @classmethod
    def _free_memory_pressure(cls):
        """
        آزادسازی حافظه در صورت فشار بالا با تخلیه مدل‌های کم‌اهمیت‌تر
        """
        logger = Logger(__name__).get_logger()
        memory_manager = ModelMemoryManager()

        with cls._global_model_lock:
            # بررسی مدل‌های بارگیری شده
            loaded_models = [
                (model_cls, instance)
                for model_cls, instance in cls._instances.items()
                if hasattr(instance, 'model_loaded') and instance.model_loaded
            ]

            if not loaded_models:
                logger.debug("هیچ مدلی بارگیری نشده است")
                return

            # مرتب‌سازی بر اساس اولویت (اولویت بالاتر = مقدار بیشتر) و زمان آخرین استفاده
            loaded_models.sort(
                key=lambda x: (
                    cls._model_priorities.get(x[0], 10),  # اولویت مدل
                    -x[1].last_used  # زمان آخرین استفاده (منفی برای مرتب‌سازی نزولی)
                )
            )

            # تخلیه مدل‌ها یکی پس از دیگری تا زمانی که فشار حافظه کاهش یابد
            for model_cls, instance in loaded_models:
                logger.info(
                    f"تخلیه مدل {instance.model_name} با اولویت "
                    f"{cls._model_priorities.get(model_cls, 10)} و "
                    f"آخرین استفاده {time.time() - instance.last_used:.1f} ثانیه پیش"
                )

                instance.unload_model()

                # بررسی وضعیت حافظه پس از تخلیه
                if not memory_manager.is_memory_pressure():
                    logger.info("فشار حافظه کاهش یافت")
                    break

            # پاکسازی حافظه
            memory_manager.clear_memory()

    @classmethod
    def set_model_priority(cls, model_class, priority):
        """
        تنظیم اولویت برای یک کلاس مدل در مدیریت حافظه

        مقادیر کوچکتر نشان‌دهنده اولویت بالاتر برای تخلیه است.

        Args:
            model_class: کلاس مدل
            priority: اولویت (0-100)
        """
        with cls._global_model_lock:
            cls._model_priorities[model_class] = priority
            Logger(__name__).get_logger().info(
                f"اولویت مدل {model_class.__name__} به {priority} تنظیم شد"
            )

    def load_model(self) -> None:
        """
        بارگیری مدل

        این متد در کلاس‌های فرزند پیاده‌سازی می‌شود.
        """
        pass

    def ensure_model_loaded(self) -> None:
        """
        اطمینان از بارگیری مدل با در نظر گرفتن همگام‌سازی و مدیریت منابع
        """
        # اگر مدل قبلاً بارگیری شده، فقط زمان را به‌روز کن
        if self.model_loaded:
            self.last_used = time.time()
            return

        # تلاش برای اکتساب قفل
        with self._model_locks[self.__class__]:
            # بررسی مجدد پس از اکتساب قفل
            if not self.model_loaded:
                self.logger.info(f"بارگیری مدل {self.model_name}")

                # بررسی فشار حافظه قبل از بارگیری
                if self.memory_manager.is_memory_pressure():
                    self.logger.warning(
                        f"فشار حافظه قبل از بارگیری مدل: "
                        f"{self.memory_manager.get_memory_usage():.1f}%"
                    )
                    self._free_memory_pressure()

                try:
                    # زمان‌سنجی بارگیری
                    start_time = time.time()

                    # بارگیری واقعی مدل
                    self.load_model()

                    # ثبت زمان بارگیری
                    load_time = time.time() - start_time
                    self.logger.info(f"مدل {self.model_name} در {load_time:.2f} ثانیه بارگیری شد")
                except Exception as e:
                    self.logger.error(
                        f"خطا در بارگیری مدل {self.model_name}: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    raise RuntimeError(f"خطا در بارگیری مدل {self.model_name}") from e

        # به‌روزرسانی زمان آخرین استفاده
        self.last_used = time.time()

    def unload_model(self) -> None:
        """
        تخلیه مدل از حافظه برای آزادسازی منابع

        این متد می‌تواند در کلاس‌های فرزند برای پاکسازی‌های اختصاصی بازنویسی شود.
        """
        if self.model_loaded:
            with self._model_locks[self.__class__]:
                if self.model_loaded:  # بررسی مجدد پس از اکتساب قفل
                    self.logger.info(f"تخلیه مدل {self.model_name} از حافظه")

                    # پاکسازی مدل
                    self.model = None
                    self.tokenizer = None
                    self.processor = None
                    self.model_loaded = False

                    # پاکسازی حافظه
                    self.memory_manager.clear_memory()

                    # ثبت وضعیت حافظه
                    self.memory_manager.log_memory_status()

    def generate_cache_key(self, method_name: str, input_data: Any) -> str:
        """
        تولید کلید کش برای یک متد و داده ورودی

        کلیدها شامل نام کلاس، نام متد و هش ورودی هستند تا منحصر به فرد باشند.

        Args:
            method_name: نام متد
            input_data: داده ورودی

        Returns:
            کلید کش
        """
        # تبدیل ورودی به رشته
        if isinstance(input_data, str):
            input_str = input_data
        elif isinstance(input_data, (list, tuple)) and all(isinstance(item, str) for item in input_data):
            # اگر لیستی از رشته‌ها باشد، آنها را با جداکننده ترکیب کن
            input_str = "||".join(input_data)
        else:
            # سعی کن به رشته تبدیل کنی
            try:
                if isinstance(input_data, (dict, list, tuple)):
                    input_str = json.dumps(input_data, sort_keys=True)
                else:
                    input_str = str(input_data)
            except Exception:
                # اگر نتوانی به رشته تبدیل کنی، از نوع و شناسه استفاده کن
                input_str = f"{type(input_data)}:{id(input_data)}"

        # تولید هش
        input_hash = hashlib.md5(input_str.encode('utf-8')).hexdigest()

        # ترکیب هش با نام کلاس و متد
        return f"{self.cache_prefix}{method_name}:{input_hash}"

    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        دریافت داده از کش با ثبت آمار

        Args:
            cache_key: کلید کش

        Returns:
            داده کش‌شده یا None
        """
        result = self.redis.get_json(cache_key)
        return result

    def save_to_cache(self, cache_key: str, data: Any, expire: Optional[int] = None) -> None:
        """
        ذخیره داده در کش

        Args:
            cache_key: کلید کش
            data: داده
            expire: زمان انقضا (ثانیه)
        """
        if expire is None:
            expire = self.default_cache_expire

        self.redis.set_json(cache_key, data, expire)

    @contextmanager
    def timer(self, method_name: str) -> None:
        """
        مدیریت زمینه (context manager) برای اندازه‌گیری زمان اجرای یک بلوک کد

        Args:
            method_name: نام متد برای ثبت آمار

        Yields:
            هیچ (تنها برای استفاده در with)
        """
        start_time = time.time()
        error_occurred = False
        try:
            yield
        except Exception:
            error_occurred = True
            raise
        finally:
            execution_time = time.time() - start_time
            self.performance_tracker.record_call(
                method_name,
                execution_time,
                from_cache=False,
                error=error_occurred
            )

    def cached_call(self, method_name: str, input_data: Any,
                    callable_func: Callable[[Any], T], expire: Optional[int] = None) -> T:
        """
        فراخوانی یک تابع با استفاده از کش و ثبت آمار

        مکانیسم:
        1. ابتدا از کش بخوان
        2. اگر در کش نبود، تابع را فراخوانی کن
        3. نتیجه را در کش ذخیره کن
        4. آمار عملکرد را ثبت کن

        Args:
            method_name: نام متد
            input_data: داده ورودی
            callable_func: تابع قابل فراخوانی
            expire: زمان انقضا (ثانیه)

        Returns:
            نتیجه فراخوانی
        """
        # تولید کلید کش
        cache_key = self.generate_cache_key(method_name, input_data)

        # زمان شروع
        start_time = time.time()

        # گام 1: تلاش برای خواندن از کش
        try:
            cached_result = self.get_from_cache(cache_key)
            if cached_result is not None:
                # ثبت آمار کش
                execution_time = time.time() - start_time
                self.performance_tracker.record_call(
                    method_name,
                    execution_time,
                    from_cache=True
                )
                return cached_result
        except Exception as e:
            self.logger.warning(f"خطا در دریافت از کش: {str(e)}")
            # ادامه اجرا بدون کش

        # گام 2: فراخوانی تابع
        error_occurred = False
        try:
            with self.timer(method_name):
                # اطمینان از بارگیری مدل
                self.ensure_model_loaded()
                # فراخوانی تابع
                result = callable_func(input_data)
        except Exception as e:
            error_occurred = True
            self.logger.error(
                f"خطا در اجرای {method_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise

        # گام 3: ذخیره در کش (فقط در صورت موفقیت)
        if not error_occurred:
            try:
                self.save_to_cache(cache_key, result, expire)
            except Exception as e:
                self.logger.warning(f"خطا در ذخیره‌سازی در کش: {str(e)}")
                # ادامه بدون ذخیره در کش

        return result

    def batch_cached_call(self, method_name: str, batch_inputs: List[Any],
                         callable_func: Callable[[List[Any]], List[T]],
                         expire: Optional[int] = None) -> List[T]:
        """
        فراخوانی یک تابع روی دسته‌ای از ورودی‌ها با استفاده از کش

        اگر بخشی از نتایج در کش موجود باشد، فقط بخش‌های ناموجود محاسبه می‌شوند.

        Args:
            method_name: نام متد
            batch_inputs: لیست ورودی‌ها
            callable_func: تابع قابل فراخوانی که لیستی از ورودی‌ها را می‌پذیرد
            expire: زمان انقضا (ثانیه)

        Returns:
            لیست نتایج به ترتیب ورودی‌ها
        """
        if not batch_inputs:
            return []

        # نتایج و ایندکس‌ها
        results = [None] * len(batch_inputs)
        cache_misses = []
        cache_miss_indices = []

        # بررسی کش برای هر ورودی
        for i, input_data in enumerate(batch_inputs):
            cache_key = self.generate_cache_key(method_name, input_data)
            cached_result = self.get_from_cache(cache_key)

            if cached_result is not None:
                # نتیجه در کش یافت شد
                results[i] = cached_result
                self.performance_tracker.record_call(
                    method_name,
                    0.0,  # زمان صفر برای نتایج کش
                    from_cache=True
                )
            else:
                # نیاز به محاسبه
                cache_misses.append(input_data)
                cache_miss_indices.append(i)

        # اگر همه نتایج در کش بودند
        if not cache_misses:
            return results

        # محاسبه نتایج ناموجود در کش
        try:
            with self.timer(method_name):
                # اطمینان از بارگیری مدل
                self.ensure_model_loaded()
                # فراخوانی تابع برای محاسبه نتایج ناموجود
                computed_results = callable_func(cache_misses)
        except Exception as e:
            self.logger.error(
                f"خطا در اجرای دسته‌ای {method_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise

        # ذخیره نتایج در کش و قرار دادن در لیست نتایج
        for i, result in zip(cache_miss_indices, computed_results):
            input_data = batch_inputs[i]
            cache_key = self.generate_cache_key(method_name, input_data)

            try:
                self.save_to_cache(cache_key, result, expire)
            except Exception as e:
                self.logger.warning(f"خطا در ذخیره‌سازی در کش: {str(e)}")

            results[i] = result

        return results

    @retry_on_error()
    def generic_model_call(self, method_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        فراخوانی عمومی تابع مدل با مدیریت خطا، زمان‌سنجی و تلاش مجدد

        این متد برای استفاده در متدهایی است که نیاز به کش‌گذاری ندارند
        یا خودشان مدیریت کش را انجام می‌دهند.

        Args:
            method_name: نام متد برای ثبت آمار
            func: تابع مورد نظر
            *args: آرگومان‌های تابع
            **kwargs: آرگومان‌های کلیدی تابع

        Returns:
            نتیجه فراخوانی تابع
        """
        # زمان‌سنجی و ثبت آمار
        with self.timer(method_name):
            # اطمینان از بارگیری مدل
            self.ensure_model_loaded()
            # فراخوانی تابع
            return func(*args, **kwargs)

    def invalidate_cache(self, method_name: Optional[str] = None) -> int:
        """
        ابطال کش برای یک متد خاص یا تمام متدهای مدل

        Args:
            method_name: نام متد (اختیاری - اگر None باشد، تمام کش مدل پاک می‌شود)

        Returns:
            تعداد کلیدهای حذف شده
        """
        pattern = f"{self.cache_prefix}{method_name}:*" if method_name else f"{self.cache_prefix}*"
        count = self.redis.delete_pattern(pattern)
        self.logger.info(f"{count} کلید کش برای {self.model_name}{f' (متد {method_name})' if method_name else ''} حذف شد")
        return count

    def get_cache_stats(self, method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        دریافت آمار کش برای یک متد خاص یا تمام متدهای مدل

        Args:
            method_name: نام متد (اختیاری)

        Returns:
            دیکشنری آمار کش
        """
        pattern = f"{self.cache_prefix}{method_name}:*" if method_name else f"{self.cache_prefix}*"
        keys = self.redis.get_keys(pattern)

        # تفکیک کلیدها بر اساس نام متد
        method_counts = {}
        for key in keys:
            parts = key.split(':')
            if len(parts) >= 2:
                method = parts[1]
                method_counts[method] = method_counts.get(method, 0) + 1

        return {
            "total_keys": len(keys),
            "per_method": method_counts
        }

    def learn_new_term(self, term: str, term_type: Optional[str] = None) -> bool:
        """
        یادگیری اصطلاح جدید و افزودن به مخزن داده

        این متد اصطلاحات مشاهده شده در متون را به سیستم یادگیری می‌افزاید
        تا در طول زمان، مدل‌ها بهبود یابند.

        Args:
            term: اصطلاح
            term_type: نوع اصطلاح (اختیاری)

        Returns:
            نتیجه عملیات
        """
        if not term or len(term) < 3:  # حذف اصطلاحات خیلی کوتاه
            return False

        try:
            # افزودن به اصطلاحات یادگیری‌شده
            result = self.data_repo.add_learned_term(term, 1, term_type)
            return result
        except Exception as e:
            self.logger.error(f"خطا در یادگیری اصطلاح جدید: {str(e)}")
            return False

    def detect_new_terms(self, text: str, known_terms: Set[str]) -> List[str]:
        """
        تشخیص اصطلاحات جدید در متن

        این متد اصطلاحات ناشناخته را در متن تشخیص می‌دهد.

        Args:
            text: متن ورودی
            known_terms: مجموعه اصطلاحات شناخته شده

        Returns:
            لیست اصطلاحات جدید
        """
        if not text:
            return []

        # پیش‌پردازش متن
        import re

        # حذف کاراکترهای خاص و تقسیم به کلمات
        words = re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', text)

        # اصطلاحات جدید
        new_terms = []

        for word in words:
            word = word.lower().strip()
            if word and word not in known_terms and len(word) >= 3:
                new_terms.append(word)

        return list(set(new_terms))  # حذف تکرارها

    def get_model_info(self) -> Dict[str, Any]:
        """
        دریافت اطلاعات کامل مدل

        Returns:
            دیکشنری اطلاعات مدل
        """
        info = {
            "name": self.model_name,
            "class": self.__class__.__name__,
            "loaded": self.model_loaded,
            "device": str(self.device),
            "last_used": self.last_used,
            "idle_time": time.time() - self.last_used if self.model_loaded else 0,
            "priority": self._model_priorities.get(self.__class__, 10),
            "performance": {
                "call_count": self.performance_tracker.call_count,
                "cache_hits": self.performance_tracker.cache_hits,
                "cache_misses": self.performance_tracker.cache_misses,
                "cache_hit_ratio": (self.performance_tracker.cache_hits / self.performance_tracker.call_count * 100)
                    if self.performance_tracker.call_count > 0 else 0,
                "avg_execution_time": (self.performance_tracker.total_execution_time / self.performance_tracker.cache_misses)
                    if self.performance_tracker.cache_misses > 0 else 0,
                "min_execution_time": self.performance_tracker.min_execution_time
                    if self.performance_tracker.min_execution_time != float('inf') else 0,
                "max_execution_time": self.performance_tracker.max_execution_time,
                "errors_count": self.performance_tracker.errors_count
            },
            "memory": {
                "usage_percent": self.memory_manager.get_memory_usage(),
                "threshold": self.memory_manager.memory_threshold
            },
            "config": self.model_config
        }

        # اطلاعات کش
        try:
            info["cache"] = self.get_cache_stats()
        except Exception as e:
            info["cache"] = {"error": str(e)}

        return info

    @classmethod
    def list_all_models(cls) -> List[Dict[str, Any]]:
        """
        لیست تمام نمونه‌های مدل همراه با اطلاعات آنها

        Returns:
            لیست اطلاعات مدل‌ها
        """
        models_info = []

        for model_cls, instance in cls._instances.items():
            if hasattr(instance, 'get_model_info'):
                try:
                    info = instance.get_model_info()
                    models_info.append(info)
                except Exception as e:
                    logger = Logger(__name__).get_logger()
                    logger.error(f"خطا در دریافت اطلاعات مدل {instance.model_name}: {str(e)}")
                    models_info.append({
                        "name": instance.model_name,
                        "class": instance.__class__.__name__,
                        "error": str(e)
                    })

        return models_info

    def __del__(self):
        """
        نابودکننده کلاس

        اطمینان از آزادسازی منابع در زمان حذف نمونه
        """
        try:
            if hasattr(self, 'model_loaded') and self.model_loaded:
                self.unload_model()
        except Exception:
            pass  # نادیده گرفتن خطاها در زمان نابودی
