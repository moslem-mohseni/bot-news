"""
ماژول توابع کمکی برای CryptoNewsBot

این ماژول شامل توابع و ابزارهای کمکی مختلفی است که در سراسر پروژه استفاده می‌شوند.
"""

import re
import time
import random
import string
import hashlib
import unicodedata
import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
import jdatetime
import pytz

# تنظیم منطقه زمانی تهران
TEHRAN_TZ = pytz.timezone('Asia/Tehran')


def now() -> datetime.datetime:
    """
    زمان فعلی با منطقه زمانی تهران

    Returns:
        زمان فعلی با منطقه زمانی تهران
    """
    return datetime.datetime.now(TEHRAN_TZ)


def to_jalali(date: datetime.datetime) -> jdatetime.datetime:
    """
    تبدیل تاریخ میلادی به شمسی

    Args:
        date: تاریخ میلادی

    Returns:
        تاریخ شمسی
    """
    if date.tzinfo is None:
        date = TEHRAN_TZ.localize(date)
    return jdatetime.datetime.fromgregorian(datetime=date)


def format_jalali_date(date: Union[datetime.datetime, jdatetime.datetime],
                       format_str: str = "%Y/%m/%d") -> str:
    """
    فرمت‌بندی تاریخ شمسی

    Args:
        date: تاریخ میلادی یا شمسی
        format_str: قالب فرمت‌بندی

    Returns:
        رشته فرمت‌بندی شده
    """
    if isinstance(date, datetime.datetime):
        date = to_jalali(date)
    return date.strftime(format_str)


def format_jalali_datetime(date: Union[datetime.datetime, jdatetime.datetime],
                           format_str: str = "%Y/%m/%d %H:%M:%S") -> str:
    """
    فرمت‌بندی تاریخ و زمان شمسی

    Args:
        date: تاریخ میلادی یا شمسی
        format_str: قالب فرمت‌بندی

    Returns:
        رشته فرمت‌بندی شده
    """
    return format_jalali_date(date, format_str)


def human_readable_time(seconds: float) -> str:
    """
    تبدیل ثانیه به فرمت قابل خواندن انسان

    Args:
        seconds: زمان به ثانیه

    Returns:
        زمان به فرمت قابل خواندن
    """
    if seconds < 60:
        return f"{seconds:.1f} ثانیه"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} دقیقه"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} ساعت"
    else:
        days = seconds / 86400
        return f"{days:.1f} روز"


def time_ago(date: datetime.datetime) -> str:
    """
    زمان سپری شده از یک تاریخ تا کنون به فرمت قابل خواندن

    Args:
        date: تاریخ مبدا

    Returns:
        زمان سپری شده به فرمت قابل خواندن
    """
    if date.tzinfo is None:
        date = TEHRAN_TZ.localize(date)

    current_time = now()
    diff = current_time - date

    seconds = diff.total_seconds()
    return human_readable_time(seconds) + " پیش"


def clean_text(text: str) -> str:
    """
    پاکسازی متن از کاراکترهای اضافی و یکسان‌سازی آن

    Args:
        text: متن ورودی

    Returns:
        متن پاکسازی شده
    """
    if not text:
        return ""

    # حذف فاصله‌های اضافی
    text = re.sub(r'\s+', ' ', text)

    # تبدیل ارقام عربی به فارسی
    text = text.replace('٠', '۰').replace('١', '۱').replace('٢', '۲').replace('٣', '۳')
    text = text.replace('٤', '۴').replace('٥', '۵').replace('٦', '۶').replace('٧', '۷')
    text = text.replace('٨', '۸').replace('٩', '۹')

    # تبدیل کاراکترهای عربی به فارسی
    text = text.replace('ي', 'ی').replace('ك', 'ک')

    # نرمال‌سازی یونیکد
    text = unicodedata.normalize('NFKC', text)

    # حذف فاصله‌های ابتدا و انتها
    return text.strip()


def extract_urls(text: str) -> List[str]:
    """
    استخراج تمام URLs از متن

    Args:
        text: متن ورودی

    Returns:
        لیست URLs
    """
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)


def extract_hashtags(text: str) -> List[str]:
    """
    استخراج تمام هشتگ‌ها از متن

    Args:
        text: متن ورودی

    Returns:
        لیست هشتگ‌ها
    """
    hashtag_pattern = r'#(\w+)'
    return re.findall(hashtag_pattern, text)


def extract_mentions(text: str) -> List[str]:
    """
    استخراج تمام منشن‌ها از متن

    Args:
        text: متن ورودی

    Returns:
        لیست منشن‌ها
    """
    mention_pattern = r'@(\w+)'
    return re.findall(mention_pattern, text)


def extract_numbers(text: str) -> List[float]:
    """
    استخراج تمام اعداد از متن

    Args:
        text: متن ورودی

    Returns:
        لیست اعداد
    """
    # الگو برای اعداد فارسی و انگلیسی
    number_pattern = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
    numbers = re.findall(number_pattern, text)

    result = []
    for num in numbers:
        try:
            # تبدیل به عدد
            result.append(float(num))
        except ValueError:
            continue

    return result


def extract_percentages(text: str) -> List[Tuple[float, str]]:
    """
    استخراج تمام درصدها از متن

    Args:
        text: متن ورودی

    Returns:
        لیست (عدد، متن کامل)
    """
    # الگو برای درصدهای فارسی و انگلیسی
    percent_pattern = r'([-+]?(?:\d*\.\d+|\d+))(?:\s*)(?:درصد|%)'
    matches = re.finditer(percent_pattern, text)

    result = []
    for match in matches:
        try:
            num = float(match.group(1))
            full_text = match.group(0)
            result.append((num, full_text))
        except (ValueError, IndexError):
            continue

    return result


def extract_cryptocurrencies(text: str) -> List[str]:
    """
    استخراج نام‌های ارزهای دیجیتال از متن

    Args:
        text: متن ورودی

    Returns:
        لیست نام‌های ارزهای دیجیتال
    """
    # لیست اصلی ارزهای دیجیتال معروف (می‌توان گسترش داد)
    cryptos = [
        r'\bبیت ?کوین\b', r'\bBitcoin\b', r'\bBTC\b',
        r'\bاتریوم\b', r'\bEthereum\b', r'\bETH\b',
        r'\bتتر\b', r'\bTether\b', r'\bUSDT\b',
        r'\bکاردانو\b', r'\bCardano\b', r'\bADA\b',
        r'\bبایننس\b', r'\bBinance\b', r'\bBNB\b',
        r'\bریپل\b', r'\bXRP\b', r'\bXRP\b',
        r'\bدوج ?کوین\b', r'\bDogecoin\b', r'\bDOGE\b',
        r'\bپولکادات\b', r'\bPolkadot\b', r'\bDOT\b',
        r'\bسولانا\b', r'\bSolana\b', r'\bSOL\b',
        r'\bشیبا\b', r'\bShiba\b', r'\bSHIB\b'
    ]

    found = []
    for pattern in cryptos:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found.append(match.group(0))

    return found


def generate_random_string(length: int = 10) -> str:
    """
    تولید یک رشته تصادفی

    Args:
        length: طول رشته

    Returns:
        رشته تصادفی
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def compute_hash(text: str) -> str:
    """
    محاسبه هش SHA256 یک متن

    Args:
        text: متن ورودی

    Returns:
        هش SHA256
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compute_similarity_hash(text: str) -> str:
    """
    محاسبه هش ساده برای مقایسه متن‌ها
    متن ابتدا پاکسازی و نرمال‌سازی می‌شود

    Args:
        text: متن ورودی

    Returns:
        هش ساده
    """
    # پاکسازی و نرمال‌سازی متن
    cleaned_text = clean_text(text)

    # حذف URL‌ها، منشن‌ها و هشتگ‌ها
    cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    cleaned_text = re.sub(r'#\w+', '', cleaned_text)

    # حذف کاراکترهای غیر کلمه‌ای
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # تبدیل به حروف کوچک و حذف فاصله‌های اضافی
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # محاسبه هش
    return hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()


def retry(max_tries: int = 3, delay: int = 1, backoff: int = 2, exceptions: tuple = (Exception,)):
    """
    دکوراتور تلاش مجدد برای توابع

    Args:
        max_tries: حداکثر تعداد تلاش
        delay: تاخیر اولیه بین تلاش‌ها (ثانیه)
        backoff: ضریب افزایش تاخیر
        exceptions: استثناهایی که باید مدیریت شوند

    Returns:
        دکوراتور
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_tries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    mtries -= 1
                    if mtries == 0:
                        raise

                    time.sleep(mdelay)
                    mdelay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator


def chunks(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    تقسیم یک لیست به چند بخش کوچکتر

    Args:
        lst: لیست ورودی
        chunk_size: اندازه هر بخش

    Returns:
        لیست بخش‌ها
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def limit_text_length(text: str, max_length: int = 4000,
                      suffix: str = "...") -> str:
    """
    محدود کردن طول متن به حداکثر مشخص شده

    Args:
        text: متن ورودی
        max_length: حداکثر طول مجاز
        suffix: پسوند برای متن کوتاه شده

    Returns:
        متن محدود شده
    """
    if not text or len(text) <= max_length:
        return text

    # کوتاه کردن متن با حفظ کلمات کامل
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def is_valid_url(url: str) -> bool:
    """
    بررسی اعتبار یک URL

    Args:
        url: URL مورد بررسی

    Returns:
        True اگر URL معتبر باشد
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// یا https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # دامنه
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # آی‌پی
        r'(?::\d+)?'  # پورت اختیاری
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return bool(url_pattern.match(url))


def remove_html_tags(text: str) -> str:
    """
    حذف تگ‌های HTML از متن

    Args:
        text: متن ورودی

    Returns:
        متن بدون تگ‌های HTML
    """
    if not text:
        return ""

    clean_text = re.sub(r'<.*?>', '', text)
    return re.sub(r'\s+', ' ', clean_text).strip()
