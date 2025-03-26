"""
پکیج دیتابیس (database) برای CryptoNewsBot

این پکیج مسئول مدیریت دسترسی به دیتابیس و مدل‌های داده است.
"""

from .models import (
    Base, RawNews, NewsAnalysis, GeneratedContent, UserFeedback, 
    TrainingData, SourceCredibility, NewsCategory, NewsSource,
    PublishStatus, SentimentType
)
from .db_manager import DatabaseManager
from .query_manager import QueryManager

# تنظیم نام‌های قابل دسترس از خارج پکیج
__all__ = [
    'Base',
    'RawNews',
    'NewsAnalysis',
    'GeneratedContent',
    'UserFeedback',
    'TrainingData',
    'SourceCredibility',
    'NewsCategory',
    'NewsSource',
    'PublishStatus',
    'SentimentType',
    'DatabaseManager',
    'QueryManager'
]
