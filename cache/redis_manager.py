"""
ماژول مدیریت کش Redis برای CryptoNewsBot

این ماژول مسئول مدیریت ارتباط با Redis و عملیات کش‌گذاری است.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Set, Union
import redis

from ..utils.config import Config
from ..utils.logger import Logger


class RedisManager:
    """
    کلاس مدیریت کش Redis

    این کلاس مسئول ارتباط با Redis و انجام عملیات کش‌گذاری است.
    از الگوی Singleton استفاده می‌کند تا در کل برنامه فقط یک نمونه از آن وجود داشته باشد.
    """

    _instance = None

    def __new__(cls):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس

        Returns:
            نمونه منحصر به فرد از کلاس
        """
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        راه‌اندازی اولیه کلاس و اتصال به Redis
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()

        # تلاش برای اتصال به Redis
        try:
            self.client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,  # رمزگشایی پاسخ‌ها به رشته
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # تست اتصال
            self.client.ping()
            self.logger.info(f"اتصال به Redis موفقیت‌آمیز بود - {self.config.redis_host}:{self.config.redis_port}")
            self.connected = True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.error(f"خطا در اتصال به Redis: {str(e)}")
            self.connected = False

    def reconnect(self) -> bool:
        """
        تلاش مجدد برای اتصال به Redis

        Returns:
            وضعیت اتصال
        """
        try:
            self.client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.client.ping()
            self.logger.info("اتصال مجدد به Redis موفقیت‌آمیز بود")
            self.connected = True
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self.logger.error(f"خطا در اتصال مجدد به Redis: {str(e)}")
            self.connected = False
            return False

    def _ensure_connected(self) -> bool:
        """
        اطمینان از اتصال به Redis

        Returns:
            وضعیت اتصال
        """
        if not self.connected:
            return self.reconnect()
        return True

    # === روش‌های اصلی کش ===

    def get(self, key: str) -> Optional[str]:
        """
        دریافت مقدار یک کلید

        Args:
            key: کلید مورد نظر

        Returns:
            مقدار کلید یا None در صورت عدم وجود
        """
        if not self._ensure_connected():
            return None

        try:
            return self.client.get(key)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت مقدار کلید {key}: {str(e)}")
            return None

    def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """
        تنظیم مقدار یک کلید

        Args:
            key: کلید مورد نظر
            value: مقدار کلید
            expire: زمان انقضا (ثانیه)

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            self.client.set(key, value)
            if expire:
                self.client.expire(key, expire)
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در تنظیم مقدار کلید {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """
        حذف یک کلید

        Args:
            key: کلید مورد نظر

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.delete(key))
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف کلید {key}: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """
        بررسی وجود یک کلید

        Args:
            key: کلید مورد نظر

        Returns:
            وضعیت وجود کلید
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.exists(key))
        except redis.RedisError as e:
            self.logger.error(f"خطا در بررسی وجود کلید {key}: {str(e)}")
            return False

    def expire(self, key: str, time: int) -> bool:
        """
        تنظیم زمان انقضا برای یک کلید

        Args:
            key: کلید مورد نظر
            time: زمان انقضا (ثانیه)

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.expire(key, time))
        except redis.RedisError as e:
            self.logger.error(f"خطا در تنظیم انقضا برای کلید {key}: {str(e)}")
            return False

    # === روش‌های کار با اشیاء پایتون ===

    def get_object(self, key: str) -> Any:
        """
        دریافت یک شیء پایتون از کش (با استفاده از pickle)

        Args:
            key: کلید مورد نظر

        Returns:
            شیء پایتون یا None در صورت عدم وجود
        """
        if not self._ensure_connected():
            return None

        try:
            # برای اشیاء باید decode_responses غیرفعال باشد
            data = self.client.get(key)
            if data is None:
                return None

            # تبدیل داده باینری به شیء پایتون
            if isinstance(data, str):
                data = data.encode('utf-8')
            return pickle.loads(data)
        except (redis.RedisError, pickle.PickleError) as e:
            self.logger.error(f"خطا در دریافت شیء برای کلید {key}: {str(e)}")
            return None

    def set_object(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        ذخیره یک شیء پایتون در کش (با استفاده از pickle)

        Args:
            key: کلید مورد نظر
            value: شیء پایتون
            expire: زمان انقضا (ثانیه)

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            # تبدیل شیء پایتون به داده باینری
            data = pickle.dumps(value)

            # ذخیره در Redis
            self.client.set(key, data)
            if expire:
                self.client.expire(key, expire)
            return True
        except (redis.RedisError, pickle.PickleError) as e:
            self.logger.error(f"خطا در ذخیره شیء برای کلید {key}: {str(e)}")
            return False

    # === روش‌های کار با JSON ===

    def get_json(self, key: str) -> Optional[Dict]:
        """
        دریافت یک شیء JSON از کش

        Args:
            key: کلید مورد نظر

        Returns:
            شیء JSON یا None در صورت عدم وجود
        """
        if not self._ensure_connected():
            return None

        try:
            data = self.get(key)
            if data is None:
                return None
            return json.loads(data)
        except (redis.RedisError, json.JSONDecodeError) as e:
            self.logger.error(f"خطا در دریافت JSON برای کلید {key}: {str(e)}")
            return None

    def set_json(self, key: str, value: Dict, expire: Optional[int] = None) -> bool:
        """
        ذخیره یک شیء JSON در کش

        Args:
            key: کلید مورد نظر
            value: شیء JSON
            expire: زمان انقضا (ثانیه)

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            data = json.dumps(value, ensure_ascii=False)
            return self.set(key, data, expire)
        except (redis.RedisError, TypeError) as e:
            self.logger.error(f"خطا در ذخیره JSON برای کلید {key}: {str(e)}")
            return False

    # === روش‌های کار با لیست ===

    def list_push(self, key: str, value: str, left: bool = False) -> bool:
        """
        افزودن یک مقدار به لیست

        Args:
            key: کلید لیست
            value: مقدار جدید
            left: افزودن به ابتدای لیست

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            if left:
                self.client.lpush(key, value)
            else:
                self.client.rpush(key, value)
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در افزودن به لیست {key}: {str(e)}")
            return False

    def list_pop(self, key: str, left: bool = False) -> Optional[str]:
        """
        حذف و بازگرداندن یک مقدار از لیست

        Args:
            key: کلید لیست
            left: حذف از ابتدای لیست

        Returns:
            مقدار حذف شده یا None
        """
        if not self._ensure_connected():
            return None

        try:
            if left:
                return self.client.lpop(key)
            else:
                return self.client.rpop(key)
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف از لیست {key}: {str(e)}")
            return None

    def list_length(self, key: str) -> int:
        """
        طول یک لیست

        Args:
            key: کلید لیست

        Returns:
            طول لیست
        """
        if not self._ensure_connected():
            return 0

        try:
            return self.client.llen(key)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت طول لیست {key}: {str(e)}")
            return 0

    def list_range(self, key: str, start: int = 0, end: int = -1) -> List[str]:
        """
        دریافت محدوده‌ای از لیست

        Args:
            key: کلید لیست
            start: شاخص شروع
            end: شاخص پایان

        Returns:
            لیست مقادیر
        """
        if not self._ensure_connected():
            return []

        try:
            return self.client.lrange(key, start, end)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت محدوده لیست {key}: {str(e)}")
            return []

    # === روش‌های کار با مجموعه ===

    def set_add(self, key: str, value: str) -> bool:
        """
        افزودن یک مقدار به مجموعه

        Args:
            key: کلید مجموعه
            value: مقدار جدید

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            self.client.sadd(key, value)
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در افزودن به مجموعه {key}: {str(e)}")
            return False

    def set_remove(self, key: str, value: str) -> bool:
        """
        حذف یک مقدار از مجموعه

        Args:
            key: کلید مجموعه
            value: مقدار

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.srem(key, value))
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف از مجموعه {key}: {str(e)}")
            return False

    def set_members(self, key: str) -> Set[str]:
        """
        دریافت تمام اعضای مجموعه

        Args:
            key: کلید مجموعه

        Returns:
            مجموعه اعضا
        """
        if not self._ensure_connected():
            return set()

        try:
            return set(self.client.smembers(key))
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت اعضای مجموعه {key}: {str(e)}")
            return set()

    def set_is_member(self, key: str, value: str) -> bool:
        """
        بررسی عضویت یک مقدار در مجموعه

        Args:
            key: کلید مجموعه
            value: مقدار

        Returns:
            وضعیت عضویت
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.sismember(key, value))
        except redis.RedisError as e:
            self.logger.error(f"خطا در بررسی عضویت در مجموعه {key}: {str(e)}")
            return False

    # === روش‌های کار با Hash ===

    def hash_set(self, key: str, field: str, value: str) -> bool:
        """
        تنظیم مقدار یک فیلد در هش

        Args:
            key: کلید هش
            field: نام فیلد
            value: مقدار

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            self.client.hset(key, field, value)
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در تنظیم فیلد {field} در هش {key}: {str(e)}")
            return False

    def hash_get(self, key: str, field: str) -> Optional[str]:
        """
        دریافت مقدار یک فیلد از هش

        Args:
            key: کلید هش
            field: نام فیلد

        Returns:
            مقدار فیلد یا None
        """
        if not self._ensure_connected():
            return None

        try:
            return self.client.hget(key, field)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت فیلد {field} از هش {key}: {str(e)}")
            return None

    def hash_delete(self, key: str, field: str) -> bool:
        """
        حذف یک فیلد از هش

        Args:
            key: کلید هش
            field: نام فیلد

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.hdel(key, field))
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف فیلد {field} از هش {key}: {str(e)}")
            return False

    def hash_get_all(self, key: str) -> Dict[str, str]:
        """
        دریافت تمام فیلدها و مقادیر یک هش

        Args:
            key: کلید هش

        Returns:
            دیکشنری فیلدها و مقادیر
        """
        if not self._ensure_connected():
            return {}

        try:
            return self.client.hgetall(key)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت همه فیلدهای هش {key}: {str(e)}")
            return {}

    # === روش‌های کار با Sorted Set ===

    def zset_add(self, key: str, value: str, score: float) -> bool:
        """
        افزودن یک مقدار به مجموعه مرتب‌شده

        Args:
            key: کلید مجموعه
            value: مقدار
            score: امتیاز برای مرتب‌سازی

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            self.client.zadd(key, {value: score})
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در افزودن به مجموعه مرتب {key}: {str(e)}")
            return False

    def zset_range(self, key: str, start: int = 0, end: int = -1,
                   with_scores: bool = False, desc: bool = False) -> List[Union[str, Tuple[str, float]]]:
        """
        دریافت محدوده‌ای از مجموعه مرتب‌شده

        Args:
            key: کلید مجموعه
            start: شاخص شروع
            end: شاخص پایان
            with_scores: شامل امتیازها
            desc: مرتب‌سازی نزولی

        Returns:
            لیست مقادیر (و امتیازها)
        """
        if not self._ensure_connected():
            return []

        try:
            if desc:
                return self.client.zrevrange(key, start, end, withscores=with_scores)
            else:
                return self.client.zrange(key, start, end, withscores=with_scores)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت محدوده مجموعه مرتب {key}: {str(e)}")
            return []

    def zset_remove(self, key: str, value: str) -> bool:
        """
        حذف یک مقدار از مجموعه مرتب‌شده

        Args:
            key: کلید مجموعه
            value: مقدار

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            return bool(self.client.zrem(key, value))
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف از مجموعه مرتب {key}: {str(e)}")
            return False

    # === روش‌های مدیریت کلیدها ===

    def delete_pattern(self, pattern: str) -> int:
        """
        حذف کلیدهای منطبق با یک الگو

        Args:
            pattern: الگوی کلید

        Returns:
            تعداد کلیدهای حذف شده
        """
        if not self._ensure_connected():
            return 0

        try:
            pipeline = self.client.pipeline()
            keys = self.client.keys(pattern)

            if not keys:
                return 0

            pipeline.delete(*keys)
            results = pipeline.execute()
            return results[0] if results else 0
        except redis.RedisError as e:
            self.logger.error(f"خطا در حذف کلیدهای منطبق با {pattern}: {str(e)}")
            return 0

    def flush_db(self) -> bool:
        """
        پاک کردن کل دیتابیس (خطرناک!)

        Returns:
            وضعیت موفقیت عملیات
        """
        if not self._ensure_connected():
            return False

        try:
            self.client.flushdb()
            self.logger.warning("دیتابیس Redis پاک شد!")
            return True
        except redis.RedisError as e:
            self.logger.error(f"خطا در پاک کردن دیتابیس: {str(e)}")
            return False

    def get_keys(self, pattern: str = "*") -> List[str]:
        """
        دریافت کلیدهای منطبق با یک الگو

        Args:
            pattern: الگوی کلید

        Returns:
            لیست کلیدها
        """
        if not self._ensure_connected():
            return []

        try:
            return self.client.keys(pattern)
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت کلیدهای منطبق با {pattern}: {str(e)}")
            return []

    def get_info(self) -> Dict[str, Any]:
        """
        دریافت اطلاعات Redis

        Returns:
            دیکشنری اطلاعات
        """
        if not self._ensure_connected():
            return {}

        try:
            return self.client.info()
        except redis.RedisError as e:
            self.logger.error(f"خطا در دریافت اطلاعات Redis: {str(e)}")
            return {}
        