"""
ماژول لاگینگ برای CryptoNewsBot

این ماژول سیستم لاگینگ یکپارچه را برای کل پروژه فراهم می‌کند
و امکان ثبت رخدادها را در فایل و کنسول ایجاد می‌کند.
"""

import os
import logging
import logging.handlers
from typing import Optional

from .config import Config


class Logger:
    """
    کلاس مدیریت لاگینگ

    این کلاس سیستم لاگینگ یکپارچه برای کل پروژه فراهم می‌کند.
    امکان ثبت لاگ در سطوح مختلف و در فایل و کنسول را ایجاد می‌کند.
    """

    _instance = None
    _loggers = {}

    def __new__(cls, name: str = 'crypto_news_bot'):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس Logger

        Args:
            name: نام لاگر (پیش‌فرض: 'crypto_news_bot')

        Returns:
            نمونه لاگر
        """
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._configure(name)
        return cls._instance

    def _configure(self, name: str) -> None:
        """
        پیکربندی اولیه لاگر

        Args:
            name: نام لاگر
        """
        self.config = Config()
        self.name = name
        self.logger = self._get_logger(name)

    def _get_logger(self, name: str) -> logging.Logger:
        """
        ایجاد یا بازیابی یک لاگر با نام مشخص

        Args:
            name: نام لاگر

        Returns:
            شیء لاگر
        """
        if name in self._loggers:
            return self._loggers[name]

        # تبدیل سطح لاگ از رشته به ثابت
        log_level_str = self.config.log_level.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        # تنظیم فرمت لاگ
        log_format = self.config.log_format
        formatter = logging.Formatter(log_format)

        # ایجاد لاگر
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # حذف هندلرهای قبلی اگر وجود داشته باشند
        if logger.handlers:
            logger.handlers.clear()

        # افزودن هندلر کنسول
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # افزودن هندلر فایل
        try:
            # اطمینان از وجود پوشه لاگ
            log_dir = os.path.dirname(self.config.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # ایجاد هندلر فایل چرخشی (تا 5 فایل 10 مگابایتی)
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=10_000_000,  # 10 مگابایت
                backupCount=5,
                encoding='utf-8',
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # در صورت بروز خطا در ایجاد فایل لاگ
            console_handler.setLevel(logging.WARNING)
            logger.warning(f"خطا در ایجاد فایل لاگ: {str(e)}. فقط لاگ کنسول فعال است.")

        # ذخیره لاگر در کش
        self._loggers[name] = logger
        return logger

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        دریافت یک لاگر با نام مشخص

        Args:
            name: نام لاگر (اگر None باشد، از نام پیش‌فرض استفاده می‌شود)

        Returns:
            شیء لاگر
        """
        if name is None:
            return self.logger

        return self._get_logger(name)

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        ثبت پیام در سطح DEBUG

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """
        ثبت پیام در سطح INFO

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        ثبت پیام در سطح WARNING

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """
        ثبت پیام در سطح ERROR

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """
        ثبت پیام در سطح CRITICAL

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args, exc_info=True, **kwargs) -> None:
        """
        ثبت پیام استثنا با جزئیات خطا

        Args:
            message: پیام لاگ
            *args: آرگومان‌های اضافی
            exc_info: آیا اطلاعات استثنا ثبت شود
            **kwargs: آرگومان‌های کلیدی اضافی
        """
        self.logger.exception(message, *args, exc_info=exc_info, **kwargs)
