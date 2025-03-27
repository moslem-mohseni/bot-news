"""
ماژول جمع‌آوری داده از توییتر برای CryptoNewsBot

این ماژول مسئول جمع‌آوری اخبار از توییتر با استفاده از کتابخانه tweety‑ns است.
این کتابخانه با استفاده از توکن‌های مهمان و مهندسی معکوس API فرانت‌اند توییتر،
بدون نیاز به API رسمی و به صورت رایگان، توییت‌ها را استخراج می‌کند.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import hashlib

# استفاده از کتابخانه tweety‑ns به‌عنوان ابزار اصلی استخراج توییت‌ها
from tweety import TwitterAsync

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import (
    clean_text, extract_urls, extract_hashtags, extract_mentions,
    compute_similarity_hash, now, retry
)
from ..cache.redis_manager import RedisManager
from ..database.query_manager import QueryManager
from ..database.models import NewsSource


class TwitterCollector:
    """
    کلاس جمع‌آوری داده‌ها از توییتر

    این کلاس مسئول جمع‌آوری توییت‌ها از کاربران و هشتگ‌های مرتبط با ارزهای دیجیتال است.
    به جای استفاده از API رسمی توییتر که نیاز به کلید و هزینه دارد، از کتابخانه tweety‑ns استفاده می‌کند
    که یک ابزار اسکرپینگ به‌روز و رایگان برای توییتر محسوب می‌شود.
    """

    def __init__(self) -> None:
        """
        مقداردهی اولیه جمع‌کننده توییتر
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.db = QueryManager()

        # لیست کاربران و هشتگ‌های مورد نظر
        self.accounts = self.config.twitter_accounts or []
        self.hashtags = self.config.twitter_hashtags or []

        # تنظیمات جمع‌آوری
        self.max_tweets_per_query = 100
        self.days_limit = 3  # حداکثر تعداد روز برای جمع‌آوری توییت‌های قدیمی‌تر

        # کلید‌های ذخیره در Redis
        self.last_tweet_id_prefix = "twitter:last_tweet_id:"
        self.processed_tweets_key = "twitter:processed_tweets"

    async def is_processed_tweet(self, tweet_id: str) -> bool:
        """
        بررسی اینکه آیا توییت قبلاً پردازش شده است

        Args:
            tweet_id: شناسه توییت

        Returns:
            وضعیت پردازش قبلی
        """
        return self.redis.set_is_member(self.processed_tweets_key, tweet_id)

    async def mark_tweet_processed(self, tweet_id: str) -> None:
        """
        علامت‌گذاری توییت به عنوان پردازش شده

        Args:
            tweet_id: شناسه توییت
        """
        self.redis.set_add(self.processed_tweets_key, tweet_id)

    async def get_last_tweet_id(self, query_key: str) -> Optional[str]:
        """
        دریافت شناسه آخرین توییت پردازش شده

        Args:
            query_key: کلید جستجو (نام کاربر یا هشتگ)

        Returns:
            شناسه آخرین توییت یا None
        """
        key = f"{self.last_tweet_id_prefix}{query_key}"
        return self.redis.get(key)

    async def set_last_tweet_id(self, query_key: str, tweet_id: str) -> None:
        """
        تنظیم شناسه آخرین توییت پردازش شده

        Args:
            query_key: کلید جستجو (نام کاربر یا هشتگ)
            tweet_id: شناسه توییت
        """
        key = f"{self.last_tweet_id_prefix}{query_key}"
        self.redis.set(key, tweet_id)

    async def process_tweet(self, tweet: Any) -> Optional[Dict[str, Any]]:
        """
        پردازش یک توییت و استخراج اطلاعات آن

        Args:
            tweet: شیء توییت از tweety‑ns

        Returns:
            اطلاعات پردازش شده توییت یا None
        """
        try:
            # استفاده از tweet.text به عنوان محتوای توییت
            tweet_text = getattr(tweet, 'text', None)
            if not tweet_text:
                return None

            # تبدیل شناسه توییت به رشته
            tweet_id = str(getattr(tweet, 'id', None))
            if not tweet_id:
                return None

            # بررسی پردازش قبلی
            if await self.is_processed_tweet(tweet_id):
                return None

            # علامت‌گذاری توییت به عنوان پردازش شده
            await self.mark_tweet_processed(tweet_id)

            # تمیز کردن متن
            cleaned_text = clean_text(tweet_text)

            # رد توییت‌های خیلی کوتاه
            if len(cleaned_text) < 10:
                return None

            # محاسبه هش متن برای تشخیص تکرار
            text_hash = compute_similarity_hash(cleaned_text)

            # بررسی تکراری بودن متن در دیتابیس
            if self.db.is_duplicate_news(cleaned_text):
                return None

            # استخراج URLها، هشتگ‌ها و منشن‌ها (در صورت وجود)
            urls = getattr(tweet, 'urls', [])
            urls = [url for url in urls if url and "twitter.com" not in url]

            hashtags = [tag.lower() for tag in getattr(tweet, 'hashtags', [])]

            mentions = getattr(tweet, 'mentions', [])
            mentioned_usernames = [user.username for user in mentions] if mentions else []

            # بررسی و استخراج رسانه (در صورت وجود)
            media_urls = []
            has_media = False
            if hasattr(tweet, 'media') and tweet.media:
                for item in tweet.media:
                    if hasattr(item, 'url'):
                        media_urls.append(item.url)
                        has_media = True

            # تعیین تاریخ انتشار
            published_at = getattr(tweet, 'date', None)
            if not published_at:
                published_at = getattr(tweet, 'created_at', datetime.utcnow())

            # دریافت اطلاعات کاربری
            user = getattr(tweet, 'user', None)
            if user is None:
                # در صورت عدم وجود اطلاعات کاربری، از مقادیر پیش‌فرض استفاده می‌شود
                user = type('User', (), {})()
                user.username = "unknown"
                user.id = 0
                user.displayname = "unknown"
                user.followersCount = 0
                user.verified = False

            # استخراج آمارهای توییت (در صورت وجود)
            retweet_count = getattr(tweet, 'retweet_count', 0)
            like_count = getattr(tweet, 'like_count', 0)
            reply_count = getattr(tweet, 'reply_count', 0)
            quote_count = getattr(tweet, 'quote_count', 0)
            is_retweet = getattr(tweet, 'is_retweet', False)

            tweet_data = {
                "external_id": tweet_id,
                "source": NewsSource.TWITTER.value,
                "source_name": user.username,
                "text": cleaned_text,
                "title": None,  # توییتر عنوانی ندارد
                "url": f"https://twitter.com/{user.username}/status/{tweet_id}",
                "has_media": has_media,
                "media_urls": media_urls if media_urls else None,
                "published_at": published_at,
                "meta": {
                    "user_id": user.id,
                    "user_name": user.username,
                    "user_display_name": user.displayname,
                    "user_followers": getattr(user, 'followersCount', 0),
                    "user_verified": getattr(user, 'verified', False),
                    "retweet_count": retweet_count,
                    "like_count": like_count,
                    "reply_count": reply_count,
                    "quote_count": quote_count,
                    "is_retweet": is_retweet,
                    "urls": urls,
                    "hashtags": hashtags,
                    "mentions": mentioned_usernames,
                    "text_hash": text_hash
                }
            }

            return tweet_data

        except Exception as e:
            self.logger.error(f"خطا در پردازش توییت: {str(e)}")
            return None

    async def save_tweet(self, tweet_data: Dict[str, Any]) -> bool:
        """
        ذخیره اطلاعات یک توییت در دیتابیس

        Args:
            tweet_data: اطلاعات توییت

        Returns:
            وضعیت ذخیره‌سازی
        """
        try:
            raw_news = self.db.add_raw_news(
                external_id=tweet_data["external_id"],
                source=tweet_data["source"],
                source_name=tweet_data["source_name"],
                text=tweet_data["text"],
                title=tweet_data["title"],
                url=tweet_data["url"],
                has_media=tweet_data["has_media"],
                media_urls=tweet_data["media_urls"],
                published_at=tweet_data["published_at"]
            )

            if raw_news:
                return True
            return False

        except Exception as e:
            self.logger.error(f"خطا در ذخیره توییت {tweet_data['external_id']}: {str(e)}")
            return False

    def _compute_since_date(self, days_limit: Optional[int]) -> str:
        """
        محاسبه تاریخ شروع برای جمع‌آوری توییت‌ها بر اساس محدودیت روز

        Args:
            days_limit: تعداد روز

        Returns:
            تاریخ به فرمت 'YYYY-MM-DD'
        """
        since_date = (now() - timedelta(days=days_limit)).strftime('%Y-%m-%d')
        return since_date

    async def _collect_tweets_with_tweety(self, query: str, max_tweets: int) -> List[Any]:
        """
        جمع‌آوری توییت‌ها با استفاده از کتابخانه tweety‑ns

        Args:
            query: عبارت جستجو (کاربر یا هشتگ)
            max_tweets: حداکثر تعداد توییت

        Returns:
            لیست توییت‌ها
        """
        tweets = []
        app = TwitterAsync()  # ایجاد نمونه TwitterAsync

        since_date = self._compute_since_date(self.days_limit)

        try:
            if query.startswith('@'):
                username = query[1:]
                tweets_data = await app.get_tweets(username=username, since=since_date, limit=max_tweets)
            elif query.startswith('#'):
                hashtag = query[1:]
                tweets_data = await app.get_tweets(hashtag=hashtag, since=since_date, limit=max_tweets)
            else:
                tweets_data = await app.get_tweets(query=query, since=since_date, limit=max_tweets)

            for tweet in tweets_data:
                tweets.append(tweet)

            self.logger.info(f"تعداد {len(tweets)} توییت برای کوئری '{query}' جمع‌آوری شد")
            return tweets

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری توییت‌ها با tweety برای '{query}': {str(e)}")
            return tweets

    @retry(max_tries=3, delay=5, backoff=2)
    async def collect_from_twitter(self, query_term: str) -> Tuple[int, int]:
        """
        جمع‌آوری توییت‌ها از توییتر

        Args:
            query_term: عبارت جستجو (کاربر یا هشتگ)

        Returns:
            تعداد کل توییت‌ها، تعداد توییت‌های ذخیره شده
        """
        processed_count = 0
        saved_count = 0

        try:
            query_key = query_term.lstrip('@#')

            # در tweety‑ns معمولاً استفاده از last_tweet_id برای فیلتر اضافی ضروری نیست؛
            # در اینجا از محدودیت زمانی days_limit بهره می‌بریم.
            tweets = await self._collect_tweets_with_tweety(query_term, self.max_tweets_per_query)

            if not tweets:
                self.logger.info(f"هیچ توییت جدیدی برای '{query_term}' یافت نشد")
                return 0, 0

            tweets.sort(key=lambda x: getattr(x, 'date', getattr(x, 'created_at', datetime.utcnow())), reverse=True)

            newest_id = str(getattr(tweets[0], 'id', ''))
            await self.set_last_tweet_id(query_key, newest_id)

            for tweet in tweets:
                processed_count += 1
                tweet_data = await self.process_tweet(tweet)
                if tweet_data:
                    if await self.save_tweet(tweet_data):
                        saved_count += 1
                await asyncio.sleep(0.1)

            self.logger.info(
                f"جمع‌آوری از '{query_term}' پایان یافت. پردازش‌شده: {processed_count}, ذخیره‌شده: {saved_count}"
            )
            return processed_count, saved_count

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از '{query_term}': {str(e)}")
            return processed_count, saved_count

    async def collect_from_accounts(self, accounts: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        جمع‌آوری توییت‌ها از حساب‌های کاربری

        Args:
            accounts: لیست حساب‌های کاربری (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری برای هر حساب
        """
        accounts_to_collect = accounts or self.accounts
        results = {}

        for account in accounts_to_collect:
            if not account.startswith('@'):
                account = f"@{account}"

            try:
                processed, saved = await self.collect_from_twitter(account)
                results[account] = (processed, saved)
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری از حساب {account}: {str(e)}")
                results[account] = (0, 0)

        return results

    async def collect_from_hashtags(self, hashtags: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        جمع‌آوری توییت‌ها از هشتگ‌ها

        Args:
            hashtags: لیست هشتگ‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری برای هر هشتگ
        """
        hashtags_to_collect = hashtags or self.hashtags
        results = {}

        for hashtag in hashtags_to_collect:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"

            try:
                processed, saved = await self.collect_from_twitter(hashtag)
                results[hashtag] = (processed, saved)
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری از هشتگ {hashtag}: {str(e)}")
                results[hashtag] = (0, 0)

        return results

    async def run(self, accounts: Optional[List[str]] = None, hashtags: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        اجرای جمع‌کننده توییتر

        Args:
            accounts: لیست حساب‌های کاربری (اختیاری)
            hashtags: لیست هشتگ‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری
        """
        results = {
            "accounts": {},
            "hashtags": {}
        }

        try:
            if accounts or self.accounts:
                results["accounts"] = await self.collect_from_accounts(accounts)
            if hashtags or self.hashtags:
                results["hashtags"] = await self.collect_from_hashtags(hashtags)

            return results

        except Exception as e:
            self.logger.error(f"خطا در اجرای جمع‌کننده توییتر: {str(e)}")
            return results
