"""
پکیج ابزارها (utils) برای CryptoNewsBot

این پکیج شامل ابزارها و توابع کمکی است که در سراسر پروژه استفاده می‌شوند.
"""

from .config import Config
from .logger import Logger
from .helpers import (
    # توابع زمان و تاریخ
    now, to_jalali, format_jalali_date, format_jalali_datetime,
    human_readable_time, time_ago,

    # توابع پردازش متن
    clean_text, extract_urls, extract_hashtags, extract_mentions,
    extract_numbers, extract_percentages, extract_cryptocurrencies,
    compute_hash, compute_similarity_hash, limit_text_length,
    remove_html_tags,

    # توابع متفرقه
    generate_random_string, retry, chunks, is_valid_url,
)

# تنظیم نام‌های قابل دسترس از خارج پکیج
__all__ = [
    'Config',
    'Logger',
    'now',
    'to_jalali',
    'format_jalali_date',
    'format_jalali_datetime',
    'human_readable_time',
    'time_ago',
    'clean_text',
    'extract_urls',
    'extract_hashtags',
    'extract_mentions',
    'extract_numbers',
    'extract_percentages',
    'extract_cryptocurrencies',
    'compute_hash',
    'compute_similarity_hash',
    'limit_text_length',
    'remove_html_tags',
    'generate_random_string',
    'retry',
    'chunks',
    'is_valid_url',
]
