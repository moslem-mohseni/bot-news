"""
پکیج کش (cache) برای CryptoNewsBot

این پکیج مسئول مدیریت کش با استفاده از Redis است.
"""

from .redis_manager import RedisManager

# تنظیم نام‌های قابل دسترس از خارج پکیج
__all__ = ['RedisManager']
