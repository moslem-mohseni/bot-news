"""
ماژول مدیریت داده‌های مدل‌ها برای CryptoNewsBot

این ماژول مسئول مدیریت داده‌های مورد نیاز مدل‌های هوش مصنوعی از جمله اصطلاحات مالی،
اطلاعات ارزهای دیجیتال، رویدادهای مهم و اصطلاحات یادگیری‌شده است. این کلاس قابلیت‌های
ذخیره‌سازی و بازیابی داده‌ها از کش و دیتابیس، ایجاد خودکار جداول، تشخیص و یادگیری اصطلاحات
جدید، و مدیریت تکامل مستمر پایگاه دانش را فراهم می‌کند.
"""

import json
import hashlib
import time
import re
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Optional, Tuple, Union, TypeVar, Generic, Callable
from contextlib import contextmanager
import functools
import threading
from sqlalchemy import Table, Column, Integer, String, Float, JSON, DateTime, Text, Boolean, MetaData, create_engine
from sqlalchemy.sql import select, insert, update, delete, text, func, and_, or_, not_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
import pandas as pd
import numpy as np

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import now, clean_text, compute_similarity_hash
from ..cache.redis_manager import RedisManager
from ..database.db_manager import DatabaseManager

# تعریف نوع عمومی
T = TypeVar('T')

# ثابت‌ها
CACHE_EXPIRATION = {
    'crypto': 7 * 86400,  # 7 روز (ثانیه)
    'terms': 7 * 86400,  # 7 روز
    'events': 7 * 86400,  # 7 روز
    'learned': 3 * 86400,  # 3 روز
    'default': 86400  # 1 روز
}

MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2
RETRY_INITIAL_WAIT = 1
DEFAULT_BATCH_SIZE = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
IN_MEMORY_CACHE_SIZE = 500  # تعداد آیتم‌ها در کش حافظه
AUTO_TERM_CATEGORIZATION_THRESHOLD = 0.75  # آستانه اطمینان برای دسته‌بندی خودکار

# تعریف Enum برای انواع داده
class DataType:
    """انواع داده قابل ذخیره‌سازی"""
    CRYPTO = 'crypto'
    TERMS = 'terms'
    EVENTS = 'events'
    LEARNED = 'learned'


def retry_on_db_error(max_attempts=MAX_RETRY_ATTEMPTS,
                       backoff_factor=RETRY_BACKOFF_FACTOR,
                       initial_wait=RETRY_INITIAL_WAIT,
                       exceptions=(SQLAlchemyError,)):
    """
    دکوراتور برای تلاش مجدد در صورت بروز خطا در عملیات دیتابیس

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
                    # در صورتی که خطای Integrity باشد، نیازی به تلاش مجدد نیست
                    if isinstance(e, IntegrityError):
                        raise
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


class InMemoryCache:
    """
    کش حافظه داخلی برای کاهش دسترسی به Redis
    """

    def __init__(self, max_size=IN_MEMORY_CACHE_SIZE):
        """
        راه‌اندازی اولیه کش حافظه

        Args:
            max_size: حداکثر تعداد آیتم‌ها
        """
        self.max_size = max_size
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار از کش

        Args:
            key: کلید

        Returns:
            مقدار یا None
        """
        with self.lock:
            if key in self.cache:
                self.timestamps[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """
        تنظیم مقدار در کش

        Args:
            key: کلید
            value: مقدار
        """
        with self.lock:
            # بررسی نیاز به آزادسازی فضا
            if len(self.cache) >= self.max_size and key not in self.cache:
                # حذف قدیمی‌ترین آیتم
                oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]

            # افزودن آیتم جدید
            self.cache[key] = value
            self.timestamps[key] = time.time()

    def delete(self, key: str) -> None:
        """
        حذف یک کلید از کش

        Args:
            key: کلید
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]

    def delete_pattern(self, pattern: str) -> int:
        """
        حذف کلیدهای منطبق با الگو

        Args:
            pattern: الگوی regex

        Returns:
            تعداد کلیدهای حذف شده
        """
        with self.lock:
            pattern_regex = re.compile(pattern.replace('*', '.*'))
            keys_to_delete = [k for k in self.cache.keys() if pattern_regex.match(k)]

            for key in keys_to_delete:
                del self.cache[key]
                del self.timestamps[key]

            return len(keys_to_delete)

    def clear(self) -> None:
        """پاکسازی کامل کش"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class DataRepository:
    """
    کلاس مدیریت داده‌های مدل‌ها

    این کلاس مسئول مدیریت داده‌های مورد نیاز مدل‌های هوش مصنوعی شامل اصطلاحات مالی،
    اطلاعات ارزهای دیجیتال و سایر داده‌های مرتبط است. این کلاس از الگوی Singleton
    استفاده می‌کند تا فقط یک نمونه از آن در برنامه وجود داشته باشد.

    ویژگی‌های کلیدی:
    - مدیریت چندسطحی کش (کش حافظه و Redis)
    - مدیریت خودکار جداول دیتابیس
    - بارگیری خودکار داده‌های پیش‌فرض
    - یادگیری و دسته‌بندی خودکار اصطلاحات
    - پشتیبانی از عملیات‌های دسته‌ای
    - رهگیری تاریخچه و آمار استفاده
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس

        Returns:
            نمونه منحصر به فرد کلاس
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataRepository, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """
        راه‌اندازی اولیه کلاس
        """
        # جلوگیری از راه‌اندازی مجدد
        if getattr(self, '_initialized', False):
            return

        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.db_manager = DatabaseManager()

        # تنظیمات Redis
        self.cache_prefix = "model_data:"
        self.cache_expire = CACHE_EXPIRATION

        # کش حافظه
        self.memory_cache = InMemoryCache()

        # وضعیت‌های داخلی
        self._tables_initialized = False
        self._default_data_loaded = False
        self._schema_version = 1  # نسخه شما برای مدیریت تغییرات ساختار

        # اطلاعات بارگیری‌شده در حافظه برای استفاده متداول
        self._cached_data = {
            DataType.CRYPTO: None,
            DataType.TERMS: None,
            DataType.EVENTS: None
        }

        # تنظیمات دیتابیس
        self.tables = {
            'crypto_currencies': None,
            'financial_terms': None,
            'important_events': None,
            'learned_terms': None,
            'data_stats': None,  # جدید: آمار داده‌ها
            'data_history': None  # جدید: تاریخچه تغییرات
        }

        # تلاش برای ایجاد جداول دیتابیس
        self._create_tables()

        # ثبت شناسه نصب برای رهگیری آمار
        self._register_installation()

        self._initialized = True
        self.logger.info("مخزن داده با موفقیت راه‌اندازی شد")

    @retry_on_db_error()
    def _create_tables(self) -> None:
        """
        ایجاد جداول دیتابیس اگر وجود نداشته باشند
        """
        if self._tables_initialized:
            return

        meta = MetaData()

        # جدول ارزهای دیجیتال
        self.tables['crypto_currencies'] = Table(
            'crypto_currencies', meta,
            Column('id', Integer, primary_key=True),
            Column('symbol', String(20), unique=True, nullable=False),
            Column('name_fa', String(100)),
            Column('name_en', String(100)),
            Column('aliases', JSON),  # سایر نام‌ها و مترادف‌ها
            Column('importance', Float),  # اهمیت نسبی (0 تا 1)
            Column('market_cap_rank', Integer),  # جدید: رتبه سرمایه بازار
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('last_verified', DateTime)  # جدید: آخرین تأیید
        )

        # جدول اصطلاحات مالی
        self.tables['financial_terms'] = Table(
            'financial_terms', meta,
            Column('id', Integer, primary_key=True),
            Column('term_type', String(50), nullable=False),  # نوع اصطلاح (bullish, bearish, etc.)
            Column('term', String(100), nullable=False),  # اصطلاح
            Column('language', String(10), nullable=False),  # زبان (fa, en)
            Column('confidence', Float),  # میزان اطمینان (0 تا 1)
            Column('source', String(100)),  # منبع کشف اصطلاح
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('usage_count', Integer, default=0),  # تعداد استفاده
            Column('related_terms', JSON),  # جدید: اصطلاحات مرتبط
            Column('context_examples', JSON),  # جدید: مثال‌های کاربرد در متن
            Column('verified', Boolean, default=False)  # جدید: تأیید شده توسط انسان
        )

        # جدول رویدادهای مهم
        self.tables['important_events'] = Table(
            'important_events', meta,
            Column('id', Integer, primary_key=True),
            Column('event_name', String(100), nullable=False),  # نام رویداد
            Column('language', String(10), nullable=False),  # زبان (fa, en)
            Column('importance', Float),  # اهمیت (0 تا 1)
            Column('description', Text),  # توضیحات
            Column('created_at', DateTime),
            Column('updated_at', DateTime),
            Column('related_cryptos', JSON),  # جدید: ارزهای مرتبط
            Column('detected_count', Integer, default=0)  # جدید: تعداد تشخیص در اخبار
        )

        # جدول اصطلاحات یادگیری‌شده
        self.tables['learned_terms'] = Table(
            'learned_terms', meta,
            Column('id', Integer, primary_key=True),
            Column('term', String(100), nullable=False, unique=True),  # اصطلاح
            Column('term_type', String(50)),  # نوع احتمالی
            Column('language', String(10)),  # زبان تشخیص‌داده‌شده
            Column('frequency', Integer),  # فراوانی مشاهده
            Column('confidence', Float),  # اعتماد به دسته‌بندی
            Column('verified', Boolean, default=False),  # آیا توسط انسان تأیید شده؟
            Column('first_seen_at', DateTime),  # جدید: اولین مشاهده
            Column('last_seen_at', DateTime),  # جدید: آخرین مشاهده
            Column('context_examples', JSON),  # جدید: مثال‌های کاربرد
            Column('created_at', DateTime),
            Column('updated_at', DateTime)
        )

        # جدول آمار داده‌ها (جدید)
        self.tables['data_stats'] = Table(
            'data_stats', meta,
            Column('id', Integer, primary_key=True),
            Column('data_type', String(50), nullable=False),  # نوع داده
            Column('total_count', Integer, default=0),  # تعداد کل
            Column('verified_count', Integer, default=0),  # تعداد تأیید شده
            Column('last_added_at', DateTime),  # آخرین افزودن
            Column('last_update_stats', DateTime),  # آخرین به‌روزرسانی آمار
            Column('usage_metrics', JSON),  # آمار استفاده
            Column('installation_id', String(50))  # شناسه نصب
        )

        # جدول تاریخچه تغییرات (جدید)
        self.tables['data_history'] = Table(
            'data_history', meta,
            Column('id', Integer, primary_key=True),
            Column('data_type', String(50), nullable=False),  # نوع داده
            Column('item_id', Integer),  # شناسه آیتم
            Column('item_key', String(100)),  # کلید آیتم (برای جستجو)
            Column('action', String(20)),  # عملیات (add, update, delete)
            Column('change_data', JSON),  # داده‌های تغییر
            Column('created_at', DateTime),  # زمان ایجاد
            Column('user_id', String(50)),  # شناسه کاربر ایجادکننده (اختیاری)
            Column('source', String(50))  # منبع تغییر
        )

        # ایجاد جداول در دیتابیس
        try:
            engine = create_engine(self.config.get_db_url())
            meta.create_all(engine)
            self._tables_initialized = True
            self.logger.info("جداول مدیریت داده مدل‌ها با موفقیت ایجاد شدند")
        except Exception as e:
            self.logger.error(f"خطا در ایجاد جداول مدیریت داده مدل‌ها: {str(e)}")
            raise

        # بررسی نیاز به مهاجرت داده‌ها
        self._check_for_migrations()

    def _check_for_migrations(self) -> None:
        """
        بررسی نیاز به مهاجرت و به‌روزرسانی ساختار جداول

        این متد تغییرات ساختاری جداول را در صورت تغییر نسخه شما مدیریت می‌کند.
        """
        try:
            # بررسی وجود جدول نسخه
            query = "SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_version'"
            exists = self.db_manager.execute_raw_sql(query)

            if not exists:
                # ایجاد جدول نسخه
                query = """
                CREATE TABLE IF NOT EXISTS schema_version (
                    id SERIAL PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TIMESTAMP NOT NULL
                )
                """
                self.db_manager.execute_raw_sql(query)

                # درج نسخه اولیه
                query = "INSERT INTO schema_version (version, applied_at) VALUES (1, NOW())"
                self.db_manager.execute_raw_sql(query)
                return

            # دریافت آخرین نسخه
            query = "SELECT MAX(version) AS version FROM schema_version"
            result = self.db_manager.execute_raw_sql(query)
            current_version = result[0]['version'] if result else 0

            # اگر نیاز به مهاجرت باشد
            if current_version < self._schema_version:
                self.logger.info(f"در حال مهاجرت از نسخه {current_version} به نسخه {self._schema_version}")

                # اجرای مهاجرت‌های لازم
                self._run_migrations(current_version)

                # به‌روزرسانی نسخه
                query = "INSERT INTO schema_version (version, applied_at) VALUES (%s, NOW())"
                self.db_manager.execute_raw_sql(query, (self._schema_version,))

                self.logger.info(f"مهاجرت به نسخه {self._schema_version} با موفقیت انجام شد")
        except Exception as e:
            self.logger.error(f"خطا در بررسی مهاجرت: {str(e)}")

    def _run_migrations(self, current_version: int) -> None:
        """
        اجرای مهاجرت‌های لازم برای ارتقاء ساختار دیتابیس

        Args:
            current_version: نسخه فعلی
        """
        # مهاجرت به نسخه 2
        if current_version < 2 and self._schema_version >= 2:
            try:
                # مثال: افزودن ستون جدید
                query = "ALTER TABLE crypto_currencies ADD COLUMN IF NOT EXISTS market_cap_rank INTEGER"
                self.db_manager.execute_raw_sql(query)

                # مثال: تغییر نوع ستون
                # query = "ALTER TABLE financial_terms ALTER COLUMN confidence TYPE FLOAT"
                # self.db_manager.execute_raw_sql(query)
            except Exception as e:
                self.logger.error(f"خطا در مهاجرت به نسخه 2: {str(e)}")
                raise

        # مهاجرت به نسخه 3
        if current_version < 3 and self._schema_version >= 3:
            # افزودن مهاجرت‌های دیگر اینجا
            pass

    def _register_installation(self) -> None:
        """
        ثبت شناسه نصب برای رهگیری آمار
        """
        try:
            # بررسی وجود شناسه نصب
            query = "SELECT COUNT(*) AS count FROM data_stats"
            result = self.db_manager.execute_raw_sql(query)

            if not result or result[0]['count'] == 0:
                # ایجاد شناسه نصب
                installation_id = hashlib.md5(f"{time.time()}_{id(self)}".encode()).hexdigest()

                # ثبت آمار اولیه برای انواع داده
                for data_type in [DataType.CRYPTO, DataType.TERMS, DataType.EVENTS, DataType.LEARNED]:
                    query = """
                    INSERT INTO data_stats 
                    (data_type, total_count, verified_count, last_update_stats, installation_id)
                    VALUES (%s, 0, 0, %s, %s)
                    """
                    self.db_manager.execute_raw_sql(query, (data_type, now(), installation_id))

                self.logger.info(f"شناسه نصب ثبت شد: {installation_id[:8]}...")
        except Exception as e:
            self.logger.warning(f"خطا در ثبت شناسه نصب: {str(e)}")

    def _update_stats(self, data_type: str) -> None:
        """
        به‌روزرسانی آمار برای یک نوع داده

        Args:
            data_type: نوع داده
        """
        try:
            # جدول مربوط به نوع داده
            table_name = self._get_table_name(data_type)

            # شمارش تعداد رکوردها
            query = f"SELECT COUNT(*) as total FROM {table_name}"
            result = self.db_manager.execute_raw_sql(query)
            total_count = result[0]['total'] if result else 0

            # شمارش تعداد رکوردهای تأیید شده
            query = f"SELECT COUNT(*) as verified FROM {table_name} WHERE verified = TRUE"
            result = self.db_manager.execute_raw_sql(query)
            verified_count = result[0]['verified'] if result else 0

            # به‌روزرسانی جدول آمار
            query = """
            UPDATE data_stats 
            SET total_count = %s, verified_count = %s, last_update_stats = %s
            WHERE data_type = %s
            """
            self.db_manager.execute_raw_sql(query, (total_count, verified_count, now(), data_type))

        except Exception as e:
            self.logger.warning(f"خطا در به‌روزرسانی آمار {data_type}: {str(e)}")

    def _get_table_name(self, data_type: str) -> str:
        """
        دریافت نام جدول مربوط به نوع داده

        Args:
            data_type: نوع داده

        Returns:
            نام جدول
        """
        table_map = {
            DataType.CRYPTO: 'crypto_currencies',
            DataType.TERMS: 'financial_terms',
            DataType.EVENTS: 'important_events',
            DataType.LEARNED: 'learned_terms'
        }

        return table_map.get(data_type, '')

    def _record_history(self, data_type: str, action: str, item_id: Optional[int] = None,
                      item_key: Optional[str] = None, change_data: Optional[Dict] = None,
                      source: str = 'system') -> None:
        """
        ثبت تاریخچه تغییرات

        Args:
            data_type: نوع داده
            action: نوع عملیات (add, update, delete)
            item_id: شناسه آیتم (اختیاری)
            item_key: کلید آیتم (اختیاری)
            change_data: داده‌های تغییر (اختیاری)
            source: منبع تغییر
        """
        try:
            # درج رکورد تاریخچه
            query = """
            INSERT INTO data_history 
            (data_type, item_id, item_key, action, change_data, created_at, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            self.db_manager.execute_raw_sql(
                query,
                (data_type, item_id, item_key, action, json.dumps(change_data or {}), now(), source)
            )
        except Exception as e:
            self.logger.warning(f"خطا در ثبت تاریخچه: {str(e)}")

    def _get_cache_key(self, data_type: str, specific_key: Optional[str] = None) -> str:
        """
        ساخت کلید کش

        Args:
            data_type: نوع داده
            specific_key: کلید خاص (اختیاری)

        Returns:
            کلید کش
        """
        if specific_key:
            return f"{self.cache_prefix}{data_type}:{specific_key}"
        return f"{self.cache_prefix}{data_type}"

    def _get_cache_expire(self, data_type: str) -> int:
        """
        دریافت زمان انقضای کش برای نوع داده

        Args:
            data_type: نوع داده

        Returns:
            زمان انقضا (ثانیه)
        """
        return self.cache_expire.get(data_type, self.cache_expire['default'])

    def _invalidate_cache(self, data_type: str, specific_key: Optional[str] = None) -> None:
        """
        ابطال کش برای یک نوع داده

        Args:
            data_type: نوع داده
            specific_key: کلید خاص (اختیاری)
        """
        try:
            # ابطال کش Redis
            cache_key = self._get_cache_key(data_type, specific_key)
            pattern = f"{cache_key}*"
            self.redis.delete_pattern(pattern)

            # ابطال کش حافظه
            self.memory_cache.delete_pattern(pattern)

            # پاکسازی داده‌های بارگیری‌شده در حافظه
            if data_type in self._cached_data:
                self._cached_data[data_type] = None

            self.logger.debug(f"کش {data_type}{' برای ' + specific_key if specific_key else ''} باطل شد")
        except Exception as e:
            self.logger.warning(f"خطا در ابطال کش: {str(e)}")

    @contextmanager
    def _db_session(self):
        """
        مدیریت زمینه برای جلسه دیتابیس

        Yields:
            جلسه دیتابیس
        """
        with self.db_manager.session_scope() as session:
            try:
                yield session
            except Exception as e:
                self.logger.error(f"خطا در جلسه دیتابیس: {str(e)}")
                raise

    def get_data(self, data_type: str, specific_key: Optional[str] = None) -> Any:
        """
        دریافت داده از کش یا دیتابیس

        مکانیسم:
        1. ابتدا از کش حافظه بخوان
        2. اگر در کش حافظه نبود، از کش Redis بخوان
        3. اگر در کش نبود، از دیتابیس بخوان
        4. اگر در دیتابیس نبود، داده پایه را بارگیری کن
        5. داده را در کش ذخیره کن

        Args:
            data_type: نوع داده (crypto, terms, events)
            specific_key: کلید خاص (اختیاری)

        Returns:
            داده درخواست‌شده
        """
        # ساخت کلید کش
        cache_key = self._get_cache_key(data_type, specific_key)

        # گام 1: تلاش برای خواندن از کش حافظه
        memory_data = self.memory_cache.get(cache_key)
        if memory_data is not None:
            return memory_data

        # گام 2: تلاش برای خواندن از کش Redis
        redis_data = self.redis.get_json(cache_key)
        if redis_data is not None:
            # ذخیره در کش حافظه برای استفاده بعدی
            self.memory_cache.set(cache_key, redis_data)
            return redis_data

        # گام 3: خواندن از دیتابیس
        db_data = self._get_from_database(data_type, specific_key)

        # گام 4: اگر داده‌ای یافت نشد، بارگیری داده پایه
        if db_data is None or (isinstance(db_data, (list, dict)) and len(db_data) == 0):
            self.logger.info(f"داده {data_type} در دیتابیس یافت نشد. بارگیری داده پایه...")
            db_data = self._load_default_data(data_type)

        # گام 5: ذخیره در کش
        if db_data is not None:
            # ذخیره در Redis
            expire = self._get_cache_expire(data_type)
            self.redis.set_json(cache_key, db_data, expire)

            # ذخیره در کش حافظه
            self.memory_cache.set(cache_key, db_data)

            # ذخیره در حافظه برای استفاده متداول
            if data_type in self._cached_data and specific_key is None:
                self._cached_data[data_type] = db_data

        return db_data

    def get_data_batch(self, data_type: str, specific_keys: List[str]) -> Dict[str, Any]:
        """
        دریافت داده‌های چندگانه به صورت دسته‌ای

        Args:
            data_type: نوع داده
            specific_keys: لیست کلیدهای خاص

        Returns:
            دیکشنری {کلید: داده}
        """
        if not specific_keys:
            return {}

        results = {}
        missing_keys = []

        # گام 1: تلاش برای خواندن از کش
        for key in specific_keys:
            cache_key = self._get_cache_key(data_type, key)

            # ابتدا از کش حافظه
            memory_data = self.memory_cache.get(cache_key)
            if memory_data is not None:
                results[key] = memory_data
                continue

            # سپس از کش Redis
            redis_data = self.redis.get_json(cache_key)
            if redis_data is not None:
                results[key] = redis_data
                self.memory_cache.set(cache_key, redis_data)
                continue

            # نیاز به خواندن از دیتابیس
            missing_keys.append(key)

        # اگر همه داده‌ها در کش بودند
        if not missing_keys:
            return results

        # گام 2: خواندن داده‌های ناموجود از دیتابیس
        db_data = self._get_from_database_batch(data_type, missing_keys)

        # گام 3: افزودن به نتایج و ذخیره در کش
        expire = self._get_cache_expire(data_type)

        for key, data in db_data.items():
            results[key] = data

            if data is not None:
                cache_key = self._get_cache_key(data_type, key)
                self.redis.set_json(cache_key, data, expire)
                self.memory_cache.set(cache_key, data)

        return results

    @retry_on_db_error()
    def _get_from_database(self, data_type: str, specific_key: Optional[str] = None) -> Any:
        """
        دریافت داده از دیتابیس

        Args:
            data_type: نوع داده
            specific_key: کلید خاص (اختیاری)

        Returns:
            داده درخواست‌شده یا None
        """
        try:
            # انتخاب جدول مناسب
            table_name = self._get_table_name(data_type)
            if not table_name:
                self.logger.warning(f"نوع داده نامعتبر: {data_type}")
                return None

            # ساخت کوئری
            query = f"SELECT * FROM {table_name}"

            if data_type == DataType.CRYPTO and specific_key:
                query += f" WHERE symbol = '{specific_key}'"
            elif data_type == DataType.TERMS and specific_key:
                query += f" WHERE term_type = '{specific_key}'"
            elif data_type == DataType.EVENTS and specific_key:
                query += f" WHERE language = '{specific_key}'"
            elif data_type == DataType.LEARNED and specific_key:
                query += f" WHERE term_type = '{specific_key}'"

            # اجرای کوئری
            results = self.db_manager.execute_raw_sql(query)

            # تبدیل نتایج به فرمت مناسب
            if data_type == DataType.CRYPTO:
                # تبدیل به دیکشنری با کلید symbol
                return {row['symbol']: {
                    'names': [row['name_fa'], row['name_en']] + (row['aliases'] if row['aliases'] else []),
                    'importance': row['importance']
                } for row in results} if results else {}

            elif data_type == DataType.TERMS:
                # گروه‌بندی بر اساس term_type
                grouped = {}
                for row in results:
                    term_type = row['term_type']
                    if term_type not in grouped:
                        grouped[term_type] = []
                    grouped[term_type].append(row['term'])
                return grouped

            elif data_type == DataType.EVENTS:
                # تبدیل به لیست ساده اگر زبان مشخص است
                if specific_key:
                    return [row['event_name'] for row in results]

                # وگرنه گروه‌بندی بر اساس زبان
                grouped = {}
                for row in results:
                    lang = row['language']
                    if lang not in grouped:
                        grouped[lang] = []
                    grouped[lang].append({
                        'name': row['event_name'],
                        'importance': row['importance']
                    })
                return grouped

            elif data_type == DataType.LEARNED:
                # بازگرداندن کل داده
                return results

            return results

        except Exception as e:
            self.logger.error(f"خطا در دریافت داده {data_type} از دیتابیس: {str(e)}")
            return None

    @retry_on_db_error()
    def _get_from_database_batch(self, data_type: str, specific_keys: List[str]) -> Dict[str, Any]:
        """
        دریافت چندین داده از دیتابیس به صورت دسته‌ای

        Args:
            data_type: نوع داده
            specific_keys: لیست کلیدهای خاص

        Returns:
            دیکشنری {کلید: داده}
        """
        if not specific_keys:
            return {}

        try:
            # انتخاب جدول مناسب
            table_name = self._get_table_name(data_type)
            if not table_name:
                self.logger.warning(f"نوع داده نامعتبر: {data_type}")
                return {}

            # تعیین ستون جستجو بر اساس نوع داده
            key_column = ""
            if data_type == DataType.CRYPTO:
                key_column = "symbol"
            elif data_type == DataType.TERMS:
                key_column = "term_type"
            elif data_type == DataType.EVENTS:
                key_column = "language"
            elif data_type == DataType.LEARNED:
                key_column = "term_type"

            if not key_column:
                return {}

            # لیست کلیدها برای جستجو
            keys_str = "', '".join(specific_keys)
            query = f"SELECT * FROM {table_name} WHERE {key_column} IN ('{keys_str}')"

            # اجرای کوئری
            results = self.db_manager.execute_raw_sql(query)

            # تبدیل نتایج به فرمت مناسب
            result_dict = {}

            if data_type == DataType.CRYPTO:
                # گروه‌بندی نتایج بر اساس symbol
                for key in specific_keys:
                    key_results = [r for r in results if r[key_column] == key]
                    if key_results:
                        result_dict[key] = {row['symbol']: {
                            'names': [row['name_fa'], row['name_en']] + (row['aliases'] if row['aliases'] else []),
                            'importance': row['importance']
                        } for row in key_results}
                    else:
                        result_dict[key] = None

            elif data_type == DataType.TERMS:
                # گروه‌بندی نتایج بر اساس term_type
                for key in specific_keys:
                    key_results = [r for r in results if r[key_column] == key]
                    if key_results:
                        result_dict[key] = [row['term'] for row in key_results]
                    else:
                        result_dict[key] = []

            elif data_type == DataType.EVENTS:
                # گروه‌بندی نتایج بر اساس language
                for key in specific_keys:
                    key_results = [r for r in results if r[key_column] == key]
                    if key_results:
                        result_dict[key] = [row['event_name'] for row in key_results]
                    else:
                        result_dict[key] = []

            elif data_type == DataType.LEARNED:
                # گروه‌بندی نتایج بر اساس term_type
                for key in specific_keys:
                    key_results = [r for r in results if r[key_column] == key]
                    result_dict[key] = key_results

            return result_dict

        except Exception as e:
            self.logger.error(f"خطا در دریافت دسته‌ای داده {data_type} از دیتابیس: {str(e)}")
            return {}

    def _load_default_data(self, data_type: str) -> Any:
        """
        بارگیری داده‌های پیش‌فرض و ذخیره در دیتابیس

        Args:
            data_type: نوع داده

        Returns:
            داده‌های پیش‌فرض
        """
        try:
            if data_type == DataType.CRYPTO:
                # داده‌های پیش‌فرض ارزهای دیجیتال
                default_cryptos = {
                    "BTC": {
                        "names": ["بیتکوین", "بیت کوین", "بیت‌کوین", "bitcoin", "btc"],
                        "importance": 1.0
                    },
                    "ETH": {
                        "names": ["اتریوم", "اتر", "اتریم", "ethereum", "eth", "ether"],
                        "importance": 0.9
                    },
                    "USDT": {
                        "names": ["تتر", "تثر", "تتر", "tether", "usdt"],
                        "importance": 0.85
                    },
                    "BNB": {
                        "names": ["بایننس کوین", "بایننس", "binance coin", "bnb"],
                        "importance": 0.8
                    },
                    "XRP": {
                        "names": ["ریپل", "xrp", "ripple"],
                        "importance": 0.75
                    },
                    "ADA": {
                        "names": ["کاردانو", "cardano", "ada"],
                        "importance": 0.7
                    },
                    "SOL": {
                        "names": ["سولانا", "solana", "sol"],
                        "importance": 0.7
                    },
                    "DOT": {
                        "names": ["پولکادات", "polkadot", "dot"],
                        "importance": 0.65
                    },
                    "DOGE": {
                        "names": ["دوج کوین", "دوج", "dogecoin", "doge"],
                        "importance": 0.65
                    },
                    "SHIB": {
                        "names": ["شیبا اینو", "شیبا", "shiba inu", "shib"],
                        "importance": 0.6
                    }
                }

                # ذخیره در دیتابیس
                self._save_crypto_currencies(default_cryptos)
                return default_cryptos

            elif data_type == DataType.TERMS:
                # داده‌های پیش‌فرض اصطلاحات مالی
                default_terms = {
                    "bullish": [
                        "صعودی", "افزایشی", "رشد", "بالا", "رالی", "سبز", "مثبت", "پامپ", "داج", "روند صعودی",
                        "روند مثبت", "روند رو به بالا", "رشد قیمت", "افزایش ارزش", "صعود", "صعودی شدن",
                        "قوی شدن", "تقویت"
                    ],
                    "bearish": [
                        "نزولی", "کاهشی", "افت", "پایین", "سقوط", "قرمز", "منفی", "دامپ", "روند نزولی",
                        "روند منفی", "روند رو به پایین", "کاهش قیمت", "افت ارزش", "نزول", "ضعیف شدن",
                        "تضعیف", "فروش", "ریزش"
                    ],
                    "stable": [
                        "ثابت", "پایدار", "با ثبات", "خنثی", "رنج", "ساید", "تثبیت", "متعادل", "بدون تغییر",
                        "در تعادل", "ثبات قیمت", "حفظ ارزش"
                    ],
                    "volatility": [
                        "نوسان", "نوسانی", "پرنوسان", "متلاطم", "ناپایدار", "پرتلاطم", "پرریسک", "پرنوسانی",
                        "پر تلاطم", "بی‌ثباتی", "تغییرات شدید", "بی ثباتی قیمت", "پر تغییر", "با ریسک بالا"
                    ],
                    "accumulation": [
                        "انباشت", "جمع‌آوری", "اکیومولیشن", "تجمیع", "جمع کردن", "انباشتن", "خرید و نگهداری",
                        "ارزش‌گذاری", "اکومولیشن", "هدف بلندمدت", "کلکسیون", "انبار کردن",
                        "ذخیره‌سازی", "نگهداری"
                    ],
                    "distribution": [
                        "توزیع", "فروش تدریجی", "دیستریبیوشن", "عرضه", "پخش", "توزیع تدریجی", "تقسیم",
                        "فروش", "واگذاری", "تخلیه", "عرضه به بازار", "تقسیم سهام", "پخش کردن", "تخلیه موقعیت"
                    ],
                    "fomo": [
                        "فومو", "ترس از دست دادن", "هیجان خرید", "خرید هیجانی", "fomo", "ترس از جا ماندن",
                        "شتاب‌زدگی", "عجله برای خرید", "هجوم", "ترس از دست دادن فرصت", "خرید بدون تحلیل",
                        "خرید احساسی"
                    ],
                    "fud": [
                        "فاد", "ترس", "تردید", "ابهام", "خبر منفی", "شایعه", "fud", "شک", "عدم اطمینان",
                        "شایعه‌پراکنی", "هراس", "ترس و تردید", "جو منفی", "ایجاد جو منفی", "خبر منفی کاذب",
                        "شایعه بد", "اخبار نگران‌کننده"
                    ],
                    "support": [
                        "حمایت", "کف قیمت", "خط حمایتی", "سطح حمایت", "ساپورت", "نقطه حمایت", "حمایت قیمتی",
                        "خط کف", "نقطه خرید", "شناسایی کف", "مرز پایین"
                    ],
                    "resistance": [
                        "مقاومت", "سقف قیمت", "خط مقاومتی", "سطح مقاومت", "رزیستنس", "نقطه مقاومت", "مقاومت قیمتی",
                        "خط سقف", "نقطه فروش", "شناسایی سقف", "مرز بالا"
                    ]
                }

                # ذخیره در دیتابیس
                self._save_financial_terms(default_terms)
                return default_terms

            elif data_type == DataType.EVENTS:
                # داده‌های پیش‌فرض رویدادهای مهم
                default_events = {
                    "fa": [
                        {"name": "هاوینگ", "importance": 0.9},
                        {"name": "فورک", "importance": 0.8},
                        {"name": "هارد فورک", "importance": 0.85},
                        {"name": "سافت فورک", "importance": 0.7},
                        {"name": "رگولاتوری", "importance": 0.85},
                        {"name": "قانون‌گذاری", "importance": 0.85},
                        {"name": "پذیرش", "importance": 0.8},
                        {"name": "لیست شدن", "importance": 0.75},
                        {"name": "ممنوعیت", "importance": 0.9},
                        {"name": "هک", "importance": 0.85},
                        {"name": "سرقت", "importance": 0.85},
                        {"name": "توکن سوزی", "importance": 0.7},
                        {"name": "ایردراپ", "importance": 0.6},
                        {"name": "عرضه اولیه", "importance": 0.8},
                        {"name": "قرعه‌کشی", "importance": 0.5},
                        {"name": "به روزرسانی", "importance": 0.7},
                        {"name": "ادغام", "importance": 0.75},
                        {"name": "خرید", "importance": 0.7},
                        {"name": "سرمایه‌گذاری", "importance": 0.75},
                        {"name": "شراکت", "importance": 0.75}
                    ],
                    "en": [
                        {"name": "halving", "importance": 0.9},
                        {"name": "fork", "importance": 0.8},
                        {"name": "hard fork", "importance": 0.85},
                        {"name": "soft fork", "importance": 0.7},
                        {"name": "regulation", "importance": 0.85},
                        {"name": "regulatory", "importance": 0.85},
                        {"name": "adoption", "importance": 0.8},
                        {"name": "listing", "importance": 0.75},
                        {"name": "ban", "importance": 0.9},
                        {"name": "hack", "importance": 0.85},
                        {"name": "theft", "importance": 0.85},
                        {"name": "burn", "importance": 0.7},
                        {"name": "airdrop", "importance": 0.6},
                        {"name": "ico", "importance": 0.8},
                        {"name": "ido", "importance": 0.75},
                        {"name": "lottery", "importance": 0.5},
                        {"name": "update", "importance": 0.7},
                        {"name": "upgrade", "importance": 0.75},
                        {"name": "merge", "importance": 0.75},
                        {"name": "acquisition", "importance": 0.8},
                        {"name": "investment", "importance": 0.75},
                        {"name": "partnership", "importance": 0.75}
                    ]
                }

                # ذخیره در دیتابیس
                self._save_important_events(default_events)
                return default_events

            elif data_type == DataType.LEARNED:
                # داده‌های یادگیری‌شده پیش‌فرضی وجود ندارد
                return []

            return None

        except Exception as e:
            self.logger.error(f"خطا در بارگیری داده‌های پیش‌فرض {data_type}: {str(e)}")
            return None

    @retry_on_db_error()
    def _save_crypto_currencies(self, cryptos: Dict[str, Dict[str, Any]]) -> bool:
        """
        ذخیره ارزهای دیجیتال در دیتابیس

        Args:
            cryptos: دیکشنری ارزهای دیجیتال

        Returns:
            وضعیت عملیات
        """
        try:
            current_time = now()

            with self._db_session() as session:
                for symbol, data in cryptos.items():
                    names = data["names"]

                    # جداسازی نام فارسی و انگلیسی
                    name_fa = next((name for name in names if any('\u0600' <= c <= '\u06FF' for c in name)), names[0])
                    name_en = next((name for name in names if all(c.isascii() for c in name)), names[-1])

                    # استخراج مترادف‌ها
                    aliases = [name for name in names if name != name_fa and name != name_en]

                    # بررسی وجود رکورد
                    query = f"SELECT id FROM crypto_currencies WHERE symbol = '{symbol}'"
                    result = session.execute(query).fetchall()

                    # تنظیم رتبه سرمایه بازار پیش‌فرض
                    market_cap_rank = data.get('market_cap_rank', 999)

                    if result:
                        # به‌روزرسانی
                        query = f"""
                        UPDATE crypto_currencies
                        SET name_fa = '{name_fa}',
                            name_en = '{name_en}',
                            aliases = '{json.dumps(aliases)}',
                            importance = {data['importance']},
                            market_cap_rank = {market_cap_rank},
                            updated_at = '{current_time.isoformat()}'
                        WHERE symbol = '{symbol}'
                        """

                        # ثبت تاریخچه تغییرات
                        self._record_history(
                            DataType.CRYPTO,
                            'update',
                            result[0][0],
                            symbol,
                            {'name_fa': name_fa, 'name_en': name_en}
                        )
                    else:
                        # درج جدید
                        query = f"""
                        INSERT INTO crypto_currencies 
                        (symbol, name_fa, name_en, aliases, importance, market_cap_rank, 
                         created_at, updated_at, last_verified)
                        VALUES 
                        ('{symbol}', '{name_fa}', '{name_en}', '{json.dumps(aliases)}', 
                         {data['importance']}, {market_cap_rank},
                         '{current_time.isoformat()}', '{current_time.isoformat()}', '{current_time.isoformat()}')
                        """

                        # ثبت تاریخچه تغییرات
                        self._record_history(
                            DataType.CRYPTO,
                            'add',
                            None,
                            symbol,
                            {'name_fa': name_fa, 'name_en': name_en}
                        )

                    session.execute(query)

            # به‌روزرسانی آمار
            self._update_stats(DataType.CRYPTO)

            # ابطال کش
            self._invalidate_cache(DataType.CRYPTO)

            self.logger.info(f"ذخیره {len(cryptos)} ارز دیجیتال در دیتابیس با موفقیت انجام شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در ذخیره ارزهای دیجیتال: {str(e)}")
            raise

    @retry_on_db_error()
    def _save_crypto_currencies_batch(self, cryptos: List[Dict[str, Any]]) -> bool:
        """
        ذخیره دسته‌ای ارزهای دیجیتال در دیتابیس

        Args:
            cryptos: لیست دیکشنری ارزهای دیجیتال (هر دیکشنری شامل کلیدهای symbol, names, importance)

        Returns:
            وضعیت عملیات
        """
        if not cryptos:
            return True

        try:
            current_time = now()
            batch_size = DEFAULT_BATCH_SIZE

            with self._db_session() as session:
                # تقسیم به دسته‌های کوچکتر
                for i in range(0, len(cryptos), batch_size):
                    batch = cryptos[i:i+batch_size]
                    symbols = [crypto['symbol'] for crypto in batch]

                    # بررسی وجود رکوردها
                    symbols_str = "', '".join(symbols)
                    query = f"SELECT id, symbol FROM crypto_currencies WHERE symbol IN ('{symbols_str}')"
                    existing_records = {row[1]: row[0] for row in session.execute(query).fetchall()}

                    # جدا کردن رکوردهای موجود و جدید
                    updates = []
                    inserts = []

                    for crypto in batch:
                        symbol = crypto['symbol']
                        names = crypto['names']

                        # جداسازی نام فارسی و انگلیسی
                        name_fa = next((name for name in names if any('\u0600' <= c <= '\u06FF' for c in name)), names[0])
                        name_en = next((name for name in names if all(c.isascii() for c in name)), names[-1])

                        # استخراج مترادف‌ها
                        aliases = [name for name in names if name != name_fa and name != name_en]

                        # تنظیم رتبه سرمایه بازار پیش‌فرض
                        market_cap_rank = crypto.get('market_cap_rank', 999)

                        if symbol in existing_records:
                            # به‌روزرسانی
                            updates.append({
                                'id': existing_records[symbol],
                                'symbol': symbol,
                                'name_fa': name_fa,
                                'name_en': name_en,
                                'aliases': json.dumps(aliases),
                                'importance': crypto['importance'],
                                'market_cap_rank': market_cap_rank
                            })

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.CRYPTO,
                                'update',
                                existing_records[symbol],
                                symbol,
                                {'name_fa': name_fa, 'name_en': name_en}
                            )
                        else:
                            # درج جدید
                            inserts.append({
                                'symbol': symbol,
                                'name_fa': name_fa,
                                'name_en': name_en,
                                'aliases': json.dumps(aliases),
                                'importance': crypto['importance'],
                                'market_cap_rank': market_cap_rank,
                                'created_at': current_time.isoformat(),
                                'updated_at': current_time.isoformat(),
                                'last_verified': current_time.isoformat()
                            })

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.CRYPTO,
                                'add',
                                None,
                                symbol,
                                {'name_fa': name_fa, 'name_en': name_en}
                            )

                    # اجرای به‌روزرسانی‌ها
                    if updates:
                        for update in updates:
                            query = f"""
                            UPDATE crypto_currencies
                            SET name_fa = '{update['name_fa']}',
                                name_en = '{update['name_en']}',
                                aliases = '{update['aliases']}',
                                importance = {update['importance']},
                                market_cap_rank = {update['market_cap_rank']},
                                updated_at = '{current_time.isoformat()}'
                            WHERE id = {update['id']}
                            """
                            session.execute(query)

                    # اجرای درج‌ها
                    if inserts:
                        for insert in inserts:
                            query = f"""
                            INSERT INTO crypto_currencies 
                            (symbol, name_fa, name_en, aliases, importance, market_cap_rank, 
                             created_at, updated_at, last_verified)
                            VALUES 
                            ('{insert['symbol']}', '{insert['name_fa']}', '{insert['name_en']}', 
                             '{insert['aliases']}', {insert['importance']}, {insert['market_cap_rank']},
                             '{insert['created_at']}', '{insert['updated_at']}', '{insert['last_verified']}')
                            """
                            session.execute(query)

            # به‌روزرسانی آمار
            self._update_stats(DataType.CRYPTO)

            # ابطال کش
            self._invalidate_cache(DataType.CRYPTO)

            self.logger.info(f"ذخیره دسته‌ای {len(cryptos)} ارز دیجیتال در دیتابیس با موفقیت انجام شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در ذخیره دسته‌ای ارزهای دیجیتال: {str(e)}")
            raise

    @retry_on_db_error()
    def _save_financial_terms(self, terms: Dict[str, List[str]]) -> bool:
        """
        ذخیره اصطلاحات مالی در دیتابیس

        Args:
            terms: دیکشنری اصطلاحات مالی

        Returns:
            وضعیت عملیات
        """
        try:
            current_time = now()

            with self._db_session() as session:
                for term_type, term_list in terms.items():
                    for term in term_list:
                        # تشخیص زبان
                        lang = 'en' if all(c.isascii() for c in term) else 'fa'

                        # بررسی وجود رکورد
                        query = f"SELECT id FROM financial_terms WHERE term = '{term}' AND term_type = '{term_type}'"
                        result = session.execute(query).fetchall()

                        if result:
                            # به‌روزرسانی
                            query = f"""
                            UPDATE financial_terms
                            SET confidence = 1.0,
                                updated_at = '{current_time.isoformat()}'
                            WHERE id = {result[0][0]}
                            """

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.TERMS,
                                'update',
                                result[0][0],
                                term,
                                {'term_type': term_type}
                            )
                        else:
                            # درج جدید
                            query = f"""
                            INSERT INTO financial_terms 
                            (term_type, term, language, confidence, source, created_at, updated_at, usage_count, verified)
                            VALUES 
                            ('{term_type}', '{term}', '{lang}', 1.0, 'default', 
                             '{current_time.isoformat()}', '{current_time.isoformat()}', 0, true)
                            """

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.TERMS,
                                'add',
                                None,
                                term,
                                {'term_type': term_type}
                            )

                        session.execute(query)

            # به‌روزرسانی آمار
            self._update_stats(DataType.TERMS)

            # ابطال کش
            self._invalidate_cache(DataType.TERMS)

            self.logger.info(f"ذخیره اصطلاحات مالی در دیتابیس با موفقیت انجام شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در ذخیره اصطلاحات مالی: {str(e)}")
            raise

    @retry_on_db_error()
    def _save_important_events(self, events: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        ذخیره رویدادهای مهم در دیتابیس

        Args:
            events: دیکشنری رویدادهای مهم

        Returns:
            وضعیت عملیات
        """
        try:
            current_time = now()

            with self._db_session() as session:
                for lang, event_list in events.items():
                    for event in event_list:
                        event_name = event["name"]
                        importance = event["importance"]

                        # بررسی وجود رکورد
                        query = f"SELECT id FROM important_events WHERE event_name = '{event_name}' AND language = '{lang}'"
                        result = session.execute(query).fetchall()

                        if result:
                            # به‌روزرسانی
                            query = f"""
                            UPDATE important_events
                            SET importance = {importance},
                                updated_at = '{current_time.isoformat()}'
                            WHERE id = {result[0][0]}
                            """

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.EVENTS,
                                'update',
                                result[0][0],
                                event_name,
                                {'importance': importance}
                            )
                        else:
                            # درج جدید
                            query = f"""
                            INSERT INTO important_events 
                            (event_name, language, importance, description, created_at, updated_at, detected_count)
                            VALUES 
                            ('{event_name}', '{lang}', {importance}, NULL, 
                             '{current_time.isoformat()}', '{current_time.isoformat()}', 0)
                            """

                            # ثبت تاریخچه تغییرات
                            self._record_history(
                                DataType.EVENTS,
                                'add',
                                None,
                                event_name,
                                {'language': lang, 'importance': importance}
                            )

                        session.execute(query)

            # به‌روزرسانی آمار
            self._update_stats(DataType.EVENTS)

            # ابطال کش
            self._invalidate_cache(DataType.EVENTS)

            self.logger.info(f"ذخیره رویدادهای مهم در دیتابیس با موفقیت انجام شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در ذخیره رویدادهای مهم: {str(e)}")
            raise

    def add_new_term(self, term: str, term_type: str, confidence: float = 0.7,
                     source: str = 'auto', context: Optional[str] = None) -> bool:
        """
        افزودن اصطلاح جدید به دیتابیس و کش

        Args:
            term: اصطلاح جدید
            term_type: نوع اصطلاح
            confidence: میزان اطمینان (0 تا 1)
            source: منبع کشف اصطلاح
            context: متن نمونه برای زمینه استفاده (اختیاری)

        Returns:
            وضعیت عملیات
        """
        try:
            # تمیزسازی اصطلاح
            term = clean_text(term)

            if not term or len(term) < 2:
                self.logger.warning(f"اصطلاح نامعتبر یا خیلی کوتاه: '{term}'")
                return False

            # تشخیص زبان
            lang = 'en' if all(c.isascii() for c in term) else 'fa'
            current_time = now()

            # آماده‌سازی داده‌های زمینه
            context_data = None
            if context:
                context_data = json.dumps([{
                    'text': context[:500],  # محدودیت طول
                    'added_at': current_time.isoformat()
                }])

            with self._db_session() as session:
                # بررسی وجود رکورد
                query = f"SELECT id, usage_count, context_examples FROM financial_terms WHERE term = '{term}' AND term_type = '{term_type}'"
                result = session.execute(query).fetchall()

                if result:
                    # استخراج اطلاعات موجود
                    term_id = result[0][0]
                    usage_count = result[0][1] or 0
                    existing_contexts = result[0][2] or "[]"

                    # به‌روزرسانی زمینه
                    updated_contexts = existing_contexts
                    if context and context_data:
                        try:
                            contexts = json.loads(existing_contexts)
                            # افزودن زمینه جدید (حداکثر 5 زمینه)
                            contexts.append(json.loads(context_data)[0])
                            if len(contexts) > 5:
                                contexts = contexts[-5:]  # حفظ 5 مورد آخر
                            updated_contexts = json.dumps(contexts)
                        except json.JSONDecodeError:
                            updated_contexts = context_data

                    # به‌روزرسانی رکورد
                    query = f"""
                    UPDATE financial_terms
                    SET confidence = GREATEST(confidence, {confidence}),
                        updated_at = '{current_time.isoformat()}',
                        usage_count = {usage_count + 1}
                    """

                    if updated_contexts:
                        query += f", context_examples = '{updated_contexts}'"

                    query += f" WHERE id = {term_id}"

                    session.execute(query)

                    # ثبت تاریخچه تغییرات
                    self._record_history(
                        DataType.TERMS,
                        'update',
                        term_id,
                        term,
                        {'confidence': confidence, 'usage_count': usage_count + 1}
                    )

                    self.logger.debug(
                        f"اصطلاح '{term}' از نوع '{term_type}' به‌روزرسانی شد (تعداد استفاده: {usage_count + 1})")
                else:
                    # درج اصطلاح جدید
                    verified = source == 'manual'  # اگر منبع دستی باشد، تأیید شده است

                    query = f"""
                    INSERT INTO financial_terms 
                    (term_type, term, language, confidence, source, created_at, updated_at, 
                     usage_count, context_examples, verified)
                    VALUES 
                    ('{term_type}', '{term}', '{lang}', {confidence}, '{source}', 
                     '{current_time.isoformat()}', '{current_time.isoformat()}', 1, 
                     {'NULL' if not context_data else f"'{context_data}'"}, {verified})
                    """

                    result = session.execute(query)
                    term_id = session.execute("SELECT lastval()").fetchone()[0]

                    # ثبت تاریخچه تغییرات
                    self._record_history(
                        DataType.TERMS,
                        'add',
                        term_id,
                        term,
                        {'term_type': term_type, 'language': lang, 'source': source}
                    )

                    self.logger.info(f"اصطلاح جدید '{term}' از نوع '{term_type}' با موفقیت ذخیره شد")

            # به‌روزرسانی کش
            self._update_terms_cache(term, term_type)

            # به‌روزرسانی آمار در پس‌زمینه
            threading.Thread(target=self._update_stats, args=(DataType.TERMS,), daemon=True).start()

            return True

        except Exception as e:
            self.logger.error(f"خطا در افزودن اصطلاح جدید: {str(e)}")
            return False

    def add_multiple_terms(self, terms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        افزودن چندین اصطلاح به صورت دسته‌ای

        Args:
            terms_data: لیست دیکشنری‌های حاوی اطلاعات اصطلاحات
                هر دیکشنری باید شامل کلیدهای term و term_type باشد

        Returns:
            دیکشنری نتایج {'success': count, 'failed': count, 'details': list}
        """
        if not terms_data:
            return {'success': 0, 'failed': 0, 'details': []}

        results = {'success': 0, 'failed': 0, 'details': []}

        for term_data in terms_data:
            term = term_data.get('term', '').strip()
            term_type = term_data.get('term_type', '').strip()
            confidence = float(term_data.get('confidence', 0.7))
            source = term_data.get('source', 'auto')
            context = term_data.get('context')

            if not term or not term_type:
                results['failed'] += 1
                results['details'].append({
                    'term': term,
                    'term_type': term_type,
                    'success': False,
                    'error': 'اصطلاح یا نوع آن خالی است'
                })
                continue

            try:
                success = self.add_new_term(term, term_type, confidence, source, context)
                if success:
                    results['success'] += 1
                    results['details'].append({
                        'term': term,
                        'term_type': term_type,
                        'success': True
                    })
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'term': term,
                        'term_type': term_type,
                        'success': False,
                        'error': 'خطا در افزودن اصطلاح'
                    })
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'term': term,
                    'term_type': term_type,
                    'success': False,
                    'error': str(e)
                })

        return results

    def _update_terms_cache(self, term: str, term_type: str) -> None:
        """
        به‌روزرسانی کش اصطلاحات مالی

        Args:
            term: اصطلاح جدید
            term_type: نوع اصطلاح
        """
        try:
            # ابطال کش کامل
            self._invalidate_cache(DataType.TERMS)
        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی کش اصطلاحات: {str(e)}")

    def add_learned_term(self, term: str, frequency: int = 1, term_type: Optional[str] = None,
                         context: Optional[str] = None) -> bool:
        """
        افزودن یا به‌روزرسانی اصطلاح یادگیری‌شده

        Args:
            term: اصطلاح
            frequency: فراوانی مشاهده
            term_type: نوع احتمالی (اختیاری)
            context: متن نمونه برای زمینه استفاده (اختیاری)

        Returns:
            وضعیت عملیات
        """
        try:
            # تمیزسازی اصطلاح
            term = clean_text(term)

            if not term or len(term) < 3:
                return False

            # تشخیص زبان
            lang = 'en' if all(c.isascii() for c in term) else 'fa'
            current_time = now()

            # آماده‌سازی داده‌های زمینه
            context_data = None
            if context:
                context_data = json.dumps([{
                    'text': context[:500],  # محدودیت طول
                    'added_at': current_time.isoformat()
                }])

            with self._db_session() as session:
                # بررسی وجود رکورد
                query = f"SELECT id, frequency, context_examples, first_seen_at FROM learned_terms WHERE term = '{term}'"
                result = session.execute(query).fetchall()

                if result:
                    # به‌روزرسانی فراوانی
                    term_id = result[0][0]
                    new_frequency = result[0][1] + frequency
                    existing_contexts = result[0][2] or "[]"
                    first_seen_at = result[0][3]

                    # به‌روزرسانی زمینه
                    updated_contexts = existing_contexts
                    if context and context_data:
                        try:
                            contexts = json.loads(existing_contexts)
                            # افزودن زمینه جدید (حداکثر 5 زمینه)
                            contexts.append(json.loads(context_data)[0])
                            if len(contexts) > 5:
                                contexts = contexts[-5:]  # حفظ 5 مورد آخر
                            updated_contexts = json.dumps(contexts)
                        except json.JSONDecodeError:
                            updated_contexts = context_data

                    # تشخیص خودکار نوع اصطلاح در صورت فراوانی بالا
                    auto_term_type = None
                    confidence_update = ""

                    if new_frequency >= 5 and not term_type:
                        auto_term_type = self._categorize_term(term, existing_contexts)
                        if auto_term_type:
                            confidence_update = f", confidence = {AUTO_TERM_CATEGORIZATION_THRESHOLD}, term_type = '{auto_term_type}'"

                    # به‌روزرسانی
                    query = f"""
                    UPDATE learned_terms
                    SET frequency = {new_frequency},
                        updated_at = '{current_time.isoformat()}',
                        last_seen_at = '{current_time.isoformat()}'
                    """

                    # افزودن نوع اصطلاح اگر ارائه شده باشد
                    if term_type:
                        query += f", term_type = '{term_type}', confidence = 0.9"
                    elif auto_term_type:
                        query += confidence_update

                    # افزودن زمینه جدید
                    if updated_contexts != existing_contexts:
                        query += f", context_examples = '{updated_contexts}'"

                    query += f" WHERE id = {term_id}"

                    session.execute(query)

                    # ثبت تاریخچه تغییرات
                    history_data = {'frequency': new_frequency}
                    if term_type or auto_term_type:
                        history_data['term_type'] = term_type or auto_term_type

                    self._record_history(
                        DataType.LEARNED,
                        'update',
                        term_id,
                        term,
                        history_data
                    )

                    # اگر تشخیص خودکار نوع اصطلاح انجام شد، ممکن است به اصطلاحات مالی اضافه کنیم
                    if auto_term_type and new_frequency >= 10:
                        self.add_new_term(
                            term,
                            auto_term_type,
                            AUTO_TERM_CATEGORIZATION_THRESHOLD,
                            'learned',
                            json.loads(updated_contexts)[0]['text'] if updated_contexts else None
                        )
                else:
                    # درج جدید
                    query = f"""
                    INSERT INTO learned_terms 
                    (term, term_type, language, frequency, confidence, verified, 
                     first_seen_at, last_seen_at, context_examples, created_at, updated_at)
                    VALUES 
                    ('{term}', {f"'{term_type}'" if term_type else 'NULL'}, '{lang}', {frequency}, 
                     {0.9 if term_type else 0.5}, false, '{current_time.isoformat()}', 
                     '{current_time.isoformat()}', 
                     {'NULL' if not context_data else f"'{context_data}'"}, 
                     '{current_time.isoformat()}', '{current_time.isoformat()}')
                    """

                    session.execute(query)
                    term_id = session.execute("SELECT lastval()").fetchone()[0]

                    # ثبت تاریخچه تغییرات
                    self._record_history(
                        DataType.LEARNED,
                        'add',
                        term_id,
                        term,
                        {'term_type': term_type, 'language': lang, 'frequency': frequency}
                    )

            # ابطال کش
            self._invalidate_cache(DataType.LEARNED)

            # به‌روزرسانی آمار در پس‌زمینه
            threading.Thread(target=self._update_stats, args=(DataType.LEARNED,), daemon=True).start()

            self.logger.debug(f"اصطلاح یادگیری‌شده '{term}' با موفقیت به‌روزرسانی شد (تعداد: {frequency})")
            return True

        except Exception as e:
            self.logger.error(f"خطا در افزودن اصطلاح یادگیری‌شده: {str(e)}")
            return False

    def add_multiple_learned_terms(self, terms: List[str], text_context: Optional[str] = None) -> Dict[str, Any]:
        """
        افزودن چندین اصطلاح یادگیری‌شده به صورت دسته‌ای

        Args:
            terms: لیست اصطلاحات
            text_context: متن زمینه مشترک (اختیاری)

        Returns:
            دیکشنری نتایج {'added': count, 'updated': count, 'failed': count}
        """
        if not terms:
            return {'added': 0, 'updated': 0, 'failed': 0}

        # تمیزسازی اصطلاحات
        clean_terms = [clean_text(term) for term in terms]
        clean_terms = [term for term in clean_terms if term and len(term) >= 3]

        if not clean_terms:
            return {'added': 0, 'updated': 0, 'failed': 0}

        results = {'added': 0, 'updated': 0, 'failed': 0}

        # گروه‌بندی اصطلاحات
        term_frequency = {}
        for term in clean_terms:
            term_frequency[term] = term_frequency.get(term, 0) + 1

        # افزودن هر اصطلاح با فراوانی آن
        for term, frequency in term_frequency.items():
            try:
                # بررسی وجود رکورد
                with self._db_session() as session:
                    query = f"SELECT id FROM learned_terms WHERE term = '{term}'"
                    result = session.execute(query).fetchall()

                    # افزودن اصطلاح
                    success = self.add_learned_term(term, frequency, None, text_context)

                    if success:
                        if result:
                            results['updated'] += 1
                        else:
                            results['added'] += 1
                    else:
                        results['failed'] += 1
            except Exception as e:
                self.logger.error(f"خطا در افزودن اصطلاح یادگیری‌شده '{term}': {str(e)}")
                results['failed'] += 1

        return results

    def _categorize_term(self, term: str, context_examples: Optional[str] = None) -> Optional[str]:
        """
        دسته‌بندی خودکار اصطلاح بر اساس زمینه و الگوهای موجود

        Args:
            term: اصطلاح
            context_examples: مثال‌های زمینه (JSON string)

        Returns:
            نوع اصطلاح تشخیص داده شده یا None
        """
        # بررسی زمینه
        contexts = []
        if context_examples:
            try:
                contexts = json.loads(context_examples)
                contexts = [ctx.get('text', '') for ctx in contexts if 'text' in ctx]
            except json.JSONDecodeError:
                pass

        if not contexts:
            return None

        # دریافت اصطلاحات مالی موجود
        financial_terms = self.get_data(DataType.TERMS)
        if not financial_terms:
            return None

        # بررسی فراوانی کلمات کلیدی در زمینه
        term_type_scores = {}

        for term_type, terms_list in financial_terms.items():
            score = 0
            for ctx in contexts:
                ctx_lower = ctx.lower()
                for fin_term in terms_list:
                    if fin_term.lower() in ctx_lower:
                        score += 1

            if score > 0:
                term_type_scores[term_type] = score

        # انتخاب نوع اصطلاح با بیشترین امتیاز
        if term_type_scores:
            max_score_type = max(term_type_scores.items(), key=lambda x: x[1])[0]
            return max_score_type

        return None

    def get_potential_terms(self, min_frequency: int = 5, limit: int = 100,
                            term_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        دریافت اصطلاحات یادگیری‌شده با فراوانی بالا

        Args:
            min_frequency: حداقل فراوانی
            limit: حداکثر تعداد نتایج
            term_type: نوع اصطلاح (اختیاری)

        Returns:
            لیست اصطلاحات
        """
        try:
            query = f"""
            SELECT * FROM learned_terms 
            WHERE frequency >= {min_frequency} AND verified = FALSE
            """

            if term_type:
                query += f" AND term_type = '{term_type}'"

            query += f" ORDER BY frequency DESC, last_seen_at DESC LIMIT {limit}"

            results = self.db_manager.execute_raw_sql(query)
            return results

        except Exception as e:
            self.logger.error(f"خطا در دریافت اصطلاحات یادگیری‌شده: {str(e)}")
            return []

    def verify_learned_term(self, term_id: int, verified: bool,
                            term_type: Optional[str] = None) -> bool:
        """
        تأیید یا رد یک اصطلاح یادگیری‌شده

        Args:
            term_id: شناسه اصطلاح
            verified: وضعیت تأیید
            term_type: نوع اصطلاح (اختیاری)

        Returns:
            وضعیت عملیات
        """
        try:
            current_time = now()

            with self._db_session() as session:
                # دریافت اطلاعات اصطلاح
                query = f"SELECT term, frequency, context_examples FROM learned_terms WHERE id = {term_id}"
                result = session.execute(query).fetchone()

                if not result:
                    self.logger.warning(f"اصطلاح با شناسه {term_id} یافت نشد")
                    return False

                term, frequency, context_examples = result

                # به‌روزرسانی
                query = f"""
                UPDATE learned_terms
                SET verified = {verified},
                    updated_at = '{current_time.isoformat()}'
                """

                if term_type:
                    query += f", term_type = '{term_type}', confidence = 1.0"

                query += f" WHERE id = {term_id}"

                session.execute(query)

                # ثبت تاریخچه
                history_data = {'verified': verified}
                if term_type:
                    history_data['term_type'] = term_type

                self._record_history(
                    DataType.LEARNED,
                    'update',
                    term_id,
                    term,
                    history_data
                )

                # اگر تأیید شده و نوع اصطلاح مشخص است، به اصطلاحات مالی اضافه کنیم
                if verified and term_type:
                    context = None
                    if context_examples:
                        try:
                            contexts = json.loads(context_examples)
                            if contexts and isinstance(contexts, list) and len(contexts) > 0:
                                context = contexts[0].get('text')
                        except json.JSONDecodeError:
                            pass

                    self.add_new_term(
                        term,
                        term_type,
                        1.0,  # حداکثر اطمینان
                        'verified',
                        context
                    )

            # ابطال کش
            self._invalidate_cache(DataType.LEARNED)

            # به‌روزرسانی آمار
            self._update_stats(DataType.LEARNED)

            self.logger.info(f"اصطلاح '{term}' با موفقیت {'تأیید' if verified else 'رد'} شد")
            return True

        except Exception as e:
            self.logger.error(f"خطا در تأیید اصطلاح: {str(e)}")
            return False

    def search_terms(self, query: str, term_type: Optional[str] = None,
                     include_learned: bool = True, limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        جستجو در اصطلاحات

        Args:
            query: متن جستجو
            term_type: نوع اصطلاح (اختیاری)
            include_learned: شامل اصطلاحات یادگیری‌شده
            limit: حداکثر تعداد نتایج

        Returns:
            دیکشنری نتایج {'financial': [], 'learned': []}
        """
        try:
            results = {'financial': [], 'learned': []}

            if not query or len(query) < 2:
                return results

            # تمیزسازی کوئری
            clean_query = clean_text(query)

            with self._db_session() as session:
                # جستجو در اصطلاحات مالی
                fin_query = f"""
                SELECT * FROM financial_terms 
                WHERE term ILIKE '%{clean_query}%'
                """

                if term_type:
                    fin_query += f" AND term_type = '{term_type}'"

                fin_query += f" ORDER BY usage_count DESC, confidence DESC LIMIT {limit}"

                financial_results = session.execute(fin_query).fetchall()
                results['financial'] = [dict(row._mapping) for row in financial_results]

                # جستجو در اصطلاحات یادگیری‌شده
                if include_learned:
                    learn_query = f"""
                    SELECT * FROM learned_terms 
                    WHERE term ILIKE '%{clean_query}%'
                    """

                    if term_type:
                        learn_query += f" AND term_type = '{term_type}'"

                    learn_query += f" ORDER BY frequency DESC, verified DESC LIMIT {limit}"

                    learned_results = session.execute(learn_query).fetchall()
                    results['learned'] = [dict(row._mapping) for row in learned_results]

            return results

        except Exception as e:
            self.logger.error(f"خطا در جستجوی اصطلاحات: {str(e)}")
            return {'financial': [], 'learned': []}

    def increment_event_detection(self, event_name: str, language: str = 'fa') -> bool:
        """
        افزایش شمارنده تشخیص یک رویداد

        Args:
            event_name: نام رویداد
            language: زبان رویداد

        Returns:
            وضعیت عملیات
        """
        try:
            # تمیزسازی نام رویداد
            event_name = clean_text(event_name)
            if not event_name:
                return False

            with self._db_session() as session:
                # بررسی وجود رویداد
                query = f"SELECT id, detected_count FROM important_events WHERE event_name = '{event_name}' AND language = '{language}'"
                result = session.execute(query).fetchone()

                if result:
                    # افزایش شمارنده
                    event_id, count = result
                    new_count = (count or 0) + 1

                    query = f"""
                    UPDATE important_events
                    SET detected_count = {new_count},
                        updated_at = '{now().isoformat()}'
                    WHERE id = {event_id}
                    """

                    session.execute(query)

                    # ثبت تاریخچه
                    self._record_history(
                        DataType.EVENTS,
                        'update',
                        event_id,
                        event_name,
                        {'detected_count': new_count}
                    )

                    # ابطال کش
                    self._invalidate_cache(DataType.EVENTS)

                    return True

                return False

        except Exception as e:
            self.logger.error(f"خطا در افزایش شمارنده تشخیص رویداد: {str(e)}")
            return False

    def update_crypto_importance(self, symbol: str, importance: float) -> bool:
        """
        به‌روزرسانی اهمیت یک ارز دیجیتال

        Args:
            symbol: نماد ارز
            importance: میزان اهمیت (0 تا 1)

        Returns:
            وضعیت عملیات
        """
        try:
            if not symbol or importance < 0 or importance > 1:
                return False

            with self._db_session() as session:
                # بررسی وجود ارز
                query = f"SELECT id FROM crypto_currencies WHERE symbol = '{symbol}'"
                result = session.execute(query).fetchone()

                if result:
                    # به‌روزرسانی اهمیت
                    crypto_id = result[0]

                    query = f"""
                    UPDATE crypto_currencies
                    SET importance = {importance},
                        updated_at = '{now().isoformat()}'
                    WHERE id = {crypto_id}
                    """

                    session.execute(query)

                    # ثبت تاریخچه
                    self._record_history(
                        DataType.CRYPTO,
                        'update',
                        crypto_id,
                        symbol,
                        {'importance': importance}
                    )

                    # ابطال کش
                    self._invalidate_cache(DataType.CRYPTO)

                    return True

                return False

        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی اهمیت ارز دیجیتال: {str(e)}")
            return False

    def get_data_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار کلی داده‌ها

        Returns:
            دیکشنری آمار
        """
        try:
            # به‌روزرسانی آمار تمام انواع داده
            for data_type in [DataType.CRYPTO, DataType.TERMS, DataType.EVENTS, DataType.LEARNED]:
                self._update_stats(data_type)

            # دریافت آمار
            query = "SELECT * FROM data_stats"
            results = self.db_manager.execute_raw_sql(query)

            if not results:
                return {
                    'crypto': {'total': 0, 'verified': 0},
                    'terms': {'total': 0, 'verified': 0},
                    'events': {'total': 0, 'verified': 0},
                    'learned': {'total': 0, 'verified': 0}
                }

            # تبدیل به دیکشنری با کلید data_type
            stats = {}
            for row in results:
                data_type = row['data_type']
                stats[data_type] = {
                    'total': row['total_count'],
                    'verified': row['verified_count'],
                    'last_updated': row['last_update_stats'].isoformat() if row['last_update_stats'] else None
                }

            return stats

        except Exception as e:
            self.logger.error(f"خطا در دریافت آمار داده‌ها: {str(e)}")
            return {
                'crypto': {'total': 0, 'verified': 0},
                'terms': {'total': 0, 'verified': 0},
                'events': {'total': 0, 'verified': 0},
                'learned': {'total': 0, 'verified': 0},
                'error': str(e)
            }

    def get_change_history(self, data_type: str, days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """
        دریافت تاریخچه تغییرات

        Args:
            data_type: نوع داده
            days: تعداد روزهای گذشته
            limit: حداکثر تعداد نتایج

        Returns:
            لیست تغییرات
        """
        try:
            past_date = (datetime.now() - timedelta(days=days)).isoformat()

            query = f"""
            SELECT * FROM data_history 
            WHERE data_type = '{data_type}' AND created_at >= '{past_date}'
            ORDER BY created_at DESC LIMIT {limit}
            """

            results = self.db_manager.execute_raw_sql(query)
            return results

        except Exception as e:
            self.logger.error(f"خطا در دریافت تاریخچه تغییرات: {str(e)}")
            return []

    def export_data(self, data_type: str) -> Dict[str, Any]:
        """
        خروجی گرفتن از داده‌ها

        Args:
            data_type: نوع داده

        Returns:
            دیکشنری داده‌ها
        """
        try:
            # انتخاب جدول مناسب
            table_name = self._get_table_name(data_type)
            if not table_name:
                return {'error': f'نوع داده نامعتبر: {data_type}', 'data': []}

            # دریافت تمام داده‌ها
            query = f"SELECT * FROM {table_name}"
            results = self.db_manager.execute_raw_sql(query)

            # بررسی وجود داده
            if not results:
                return {'count': 0, 'data': []}

            # تبدیل به لیست دیکشنری‌ها
            data_list = []
            for row in results:
                row_dict = {key: value for key, value in row.items()}

                # تبدیل تاریخ‌ها به رشته
                for key, value in row_dict.items():
                    if isinstance(value, datetime):
                        row_dict[key] = value.isoformat()

                data_list.append(row_dict)

            return {
                'count': len(data_list),
                'data': data_list
            }

        except Exception as e:
            self.logger.error(f"خطا در خروجی گرفتن از داده‌ها: {str(e)}")
            return {'error': str(e), 'data': []}

    def import_data(self, data_type: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        وارد کردن داده‌ها

        Args:
            data_type: نوع داده
            data: لیست داده‌ها

        Returns:
            نتیجه عملیات
        """
        if not data:
            return {'success': True, 'count': 0, 'message': 'داده‌ای برای وارد کردن وجود ندارد'}

        try:
            if data_type == DataType.CRYPTO:
                # تبدیل به فرمت مناسب
                crypto_list = []
                for item in data:
                    if 'symbol' not in item:
                        continue

                    crypto = {
                        'symbol': item['symbol'],
                        'names': [],
                        'importance': item.get('importance', 0.5)
                    }

                    # افزودن نام‌ها
                    if 'name_fa' in item and item['name_fa']:
                        crypto['names'].append(item['name_fa'])

                    if 'name_en' in item and item['name_en']:
                        crypto['names'].append(item['name_en'])

                    if 'aliases' in item and item['aliases']:
                        try:
                            if isinstance(item['aliases'], str):
                                aliases = json.loads(item['aliases'])
                                if isinstance(aliases, list):
                                    crypto['names'].extend(aliases)
                        except json.JSONDecodeError:
                            pass

                    if not crypto['names']:
                        crypto['names'] = [crypto['symbol']]

                    crypto_list.append(crypto)

                # ذخیره داده‌ها
                result = self._save_crypto_currencies_batch(crypto_list)
                return {
                    'success': result,
                    'count': len(crypto_list),
                    'message': f"{len(crypto_list)} ارز دیجیتال با موفقیت وارد شد" if result else "خطا در وارد کردن ارزهای دیجیتال"
                }

            elif data_type == DataType.TERMS:
                # تبدیل به فرمت مناسب
                terms_dict = {}

                for item in data:
                    if 'term' not in item or 'term_type' not in item:
                        continue

                    term_type = item['term_type']
                    term = item['term']

                    if term_type not in terms_dict:
                        terms_dict[term_type] = []

                    terms_dict[term_type].append(term)

                # ذخیره داده‌ها
                result = self._save_financial_terms(terms_dict)

                total_terms = sum(len(terms) for terms in terms_dict.values())

                return {
                    'success': result,
                    'count': total_terms,
                    'message': f"{total_terms} اصطلاح مالی با موفقیت وارد شد" if result else "خطا در وارد کردن اصطلاحات مالی"
                }

            elif data_type == DataType.EVENTS:
                # تبدیل به فرمت مناسب
                events_dict = {}

                for item in data:
                    if 'event_name' not in item or 'language' not in item:
                        continue

                    language = item['language']
                    event = {
                        'name': item['event_name'],
                        'importance': item.get('importance', 0.5)
                    }

                    if language not in events_dict:
                        events_dict[language] = []

                    events_dict[language].append(event)

                # ذخیره داده‌ها
                result = self._save_important_events(events_dict)

                total_events = sum(len(events) for events in events_dict.values())

                return {
                    'success': result,
                    'count': total_events,
                    'message': f"{total_events} رویداد مهم با موفقیت وارد شد" if result else "خطا در وارد کردن رویدادهای مهم"
                }

            else:
                return {'success': False, 'count': 0, 'message': f'وارد کردن داده‌های نوع {data_type} پشتیبانی نمی‌شود'}

        except Exception as e:
            self.logger.error(f"خطا در وارد کردن داده‌ها: {str(e)}")
            return {'success': False, 'count': 0, 'message': f'خطا: {str(e)}'}

    def cleanup_data(self, older_than_days: int = 90) -> Dict[str, int]:
        """
        پاکسازی داده‌های قدیمی

        Args:
            older_than_days: قدیمی‌تر از چند روز

        Returns:
            دیکشنری تعداد رکوردهای حذف شده
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
            cleanup_counts = {}

            with self._db_session() as session:
                # پاکسازی تاریخچه
                query = f"DELETE FROM data_history WHERE created_at < '{cutoff_date}'"
                result = session.execute(query)
                cleanup_counts['history'] = result.rowcount

                # پاکسازی اصطلاحات یادگیری‌شده کم استفاده
                query = f"""
                DELETE FROM learned_terms 
                WHERE frequency < 3 AND updated_at < '{cutoff_date}' AND verified = FALSE
                """
                result = session.execute(query)
                cleanup_counts['learned_terms'] = result.rowcount

            # به‌روزرسانی آمار
            for data_type in [DataType.LEARNED]:
                self._update_stats(data_type)

            # ابطال کش
            self._invalidate_cache(DataType.LEARNED)

            self.logger.info(f"پاکسازی داده‌های قدیمی‌تر از {older_than_days} روز انجام شد")
            return cleanup_counts

        except Exception as e:
            self.logger.error(f"خطا در پاکسازی داده‌های قدیمی: {str(e)}")
            return {'error': str(e)}

    def rebuild_cache(self) -> Dict[str, bool]:
        """
        بازسازی کامل کش

        Returns:
            وضعیت عملیات
        """
        try:
            result = {}

            # پاکسازی کامل کش
            for data_type in [DataType.CRYPTO, DataType.TERMS, DataType.EVENTS, DataType.LEARNED]:
                self._invalidate_cache(data_type)

                # بارگیری مجدد داده‌ها در کش
                try:
                    self.get_data(data_type)
                    result[data_type] = True
                except Exception as e:
                    self.logger.error(f"خطا در بارگیری مجدد داده‌های {data_type}: {str(e)}")
                    result[data_type] = False

            self.logger.info("بازسازی کامل کش انجام شد")
            return result

        except Exception as e:
            self.logger.error(f"خطا در بازسازی کش: {str(e)}")
            return {'error': str(e)}
