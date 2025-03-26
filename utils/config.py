"""
ماژول مدیریت تنظیمات برای CryptoNewsBot

این ماژول مسئول خواندن تنظیمات از فایل .env و فراهم کردن دسترسی آسان به این تنظیمات
برای سایر بخش‌های پروژه است.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()


class Config:
    """
    کلاس مدیریت تنظیمات پروژه

    این کلاس مسئول خواندن، بارگذاری و ارائه تنظیمات به سایر بخش‌های پروژه است.
    متغیرهای محیطی را از فایل .env می‌خواند و مقادیر پیش‌فرض را در صورت نیاز استفاده می‌کند.
    """

    _instance = None

    def __new__(cls):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس Config
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self) -> None:
        """
        بارگذاری همه تنظیمات از متغیرهای محیطی
        """
        # === تنظیمات عمومی ===
        self.debug = self._get_env_bool("DEBUG", False)
        self.collection_interval = self._get_env_int("COLLECTION_INTERVAL", 600)  # 10 دقیقه
        self.min_importance_score = self._get_env_float("MIN_IMPORTANCE_SCORE", 0.5)

        # === تنظیمات تلگرام ===
        self.telegram_api_id = self._get_env("TELEGRAM_API_ID")
        self.telegram_api_hash = self._get_env("TELEGRAM_API_HASH")
        self.telegram_phone = self._get_env("TELEGRAM_PHONE")
        self.telegram_bot_token = self._get_env("TELEGRAM_BOT_TOKEN")
        self.telegram_session_name = self._get_env("TELEGRAM_SESSION_NAME", "crypto_news_bot")
        self.telegram_channels = self._get_env_list("TELEGRAM_CHANNELS", [])
        self.telegram_output_channel = self._get_env("TELEGRAM_OUTPUT_CHANNEL")

        # === تنظیمات توییتر ===
        self.twitter_api_key = self._get_env("TWITTER_API_KEY")
        self.twitter_api_secret = self._get_env("TWITTER_API_SECRET")
        self.twitter_access_token = self._get_env("TWITTER_ACCESS_TOKEN")
        self.twitter_access_token_secret = self._get_env("TWITTER_ACCESS_TOKEN_SECRET")
        self.twitter_accounts = self._get_env_list("TWITTER_ACCOUNTS", [])
        self.twitter_hashtags = self._get_env_list("TWITTER_HASHTAGS", [])

        # === تنظیمات وب‌سایت‌ها ===
        self.websites = self._get_env_json("WEBSITES", [])

        # === تنظیمات دیتابیس ===
        self.db_user = self._get_env("POSTGRES_USER", "postgres")
        self.db_password = self._get_env("POSTGRES_PASSWORD", "postgres")
        self.db_name = self._get_env("POSTGRES_DB", "crypto_news_bot")
        self.db_host = self._get_env("POSTGRES_HOST", "localhost")
        self.db_port = self._get_env_int("POSTGRES_PORT", 5432)

        # === تنظیمات Redis ===
        self.redis_host = self._get_env("REDIS_HOST", "localhost")
        self.redis_port = self._get_env_int("REDIS_PORT", 6379)
        self.redis_db = self._get_env_int("REDIS_DB", 0)
        self.redis_password = self._get_env("REDIS_PASSWORD", None)

        # === تنظیمات مدل‌ها ===
        self.parsbert_model_path = self._get_env("PARSBERT_MODEL_PATH", "HooshvareLab/bert-base-parsbert-uncased")
        self.multilingual_model_path = self._get_env("MULTILINGUAL_MODEL_PATH", "xlm-roberta-base")
        self.model_cache_dir = self._get_env("MODEL_CACHE_DIR", "./models/cache")

        # === تنظیمات لاگینگ ===
        self.log_level = self._get_env("LOG_LEVEL", "INFO")
        self.log_file = self._get_env("LOG_FILE", "./logs/crypto_news_bot.log")
        self.log_format = self._get_env("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        # === تنظیمات انتشار ===
        self.publish_interval = self._get_env_int("PUBLISH_INTERVAL", 1800)  # 30 دقیقه
        self.max_daily_news = self._get_env_int("MAX_DAILY_NEWS", 20)

    def _get_env(self, key: str, default: Any = None) -> Any:
        """
        خواندن یک متغیر محیطی با مقدار پیش‌فرض در صورت عدم وجود

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی یا مقدار پیش‌فرض
        """
        return os.environ.get(key, default)

    def _get_env_int(self, key: str, default: int = 0) -> int:
        """
        خواندن یک متغیر محیطی عددی صحیح

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی به صورت عدد صحیح
        """
        value = self._get_env(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _get_env_float(self, key: str, default: float = 0.0) -> float:
        """
        خواندن یک متغیر محیطی عددی اعشاری

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی به صورت عدد اعشاری
        """
        value = self._get_env(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _get_env_bool(self, key: str, default: bool = False) -> bool:
        """
        خواندن یک متغیر محیطی بولین

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی به صورت بولین
        """
        value = self._get_env(key)
        if value is None:
            return default
        return value.lower() in ("true", "yes", "1", "t", "y")

    def _get_env_list(self, key: str, default: List[str] = None) -> List[str]:
        """
        خواندن یک متغیر محیطی به صورت لیست با جداکننده کاما

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی به صورت لیست
        """
        if default is None:
            default = []
        value = self._get_env(key)
        if value is None or value == "":
            return default
        return [item.strip() for item in value.split(",")]

    def _get_env_json(self, key: str, default: Any = None) -> Any:
        """
        خواندن یک متغیر محیطی به صورت JSON

        Args:
            key: کلید متغیر محیطی
            default: مقدار پیش‌فرض در صورت عدم وجود متغیر

        Returns:
            مقدار متغیر محیطی به صورت شیء پایتون
        """
        value = self._get_env(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def get_db_url(self) -> str:
        """
        ساخت URL اتصال به دیتابیس

        Returns:
            URL اتصال به دیتابیس
        """
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def set(self, key: str, value: Any) -> None:
        """
        تنظیم یک مقدار پیکربندی در زمان اجرا

        Args:
            key: کلید تنظیم
            value: مقدار جدید
        """
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        تبدیل تمام تنظیمات به دیکشنری

        Returns:
            دیکشنری شامل تمام تنظیمات
        """
        result = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                result[key] = getattr(self, key)
        return result

    def __str__(self) -> str:
        """
        نمایش رشته‌ای تنظیمات

        Returns:
            رشته نمایشی تنظیمات
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
