"""
پکیج جمع‌آوری داده (collector) برای CryptoNewsBot

این پکیج مسئول جمع‌آوری اخبار از منابع مختلف شامل تلگرام، وب‌سایت‌ها و توییتر است.
"""

from .telegram_collector import TelegramCollector
from .website_collector import WebsiteCollector
from .twitter_collector import TwitterCollector
from .collection_manager import CollectionManager

# تنظیم نام‌های قابل دسترس از خارج پکیج
__all__ = [
    'TelegramCollector',
    'WebsiteCollector',
    'TwitterCollector',
    'CollectionManager'
]
