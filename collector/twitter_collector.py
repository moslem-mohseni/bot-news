"""
ماژول جمع‌آوری داده از توییتر برای CryptoNewsBot

این ماژول مسئول جمع‌آوری اخبار از توییتر با استفاده از کتابخانه snscrape است.
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

# برای استفاده از snscrape بدون نصب به عنوان پکیج
import snscrape.modules.twitter as sntwitter

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
    به جای استفاده از API رسمی توییتر که نیاز به کلید و هزینه دارد، از کتابخانه snscrape
    استفاده می‌کند که یک ابزار اسکرپینگ رایگان است.
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

        # کلید‌های ذخیره در ردیس
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

    async def process_tweet(self, tweet) -> Optional[Dict[str, Any]]:
        """
        پردازش یک توییت و استخراج اطلاعات آن

        Args:
            tweet: شیء توییت از snscrape

        Returns:
            اطلاعات پردازش شده توییت یا None
        """
        try:
            # بررسی توییت خالی
            if not tweet.content:
                return None

            # تبدیل شناسه توییت به رشته
            tweet_id = str(tweet.id)

            # بررسی پردازش قبلی
            if await self.is_processed_tweet(tweet_id):
                return None

            # علامت‌گذاری توییت به عنوان پردازش شده
            await self.mark_tweet_processed(tweet_id)

            # تمیز کردن متن
            cleaned_text = clean_text(tweet.content)

            # بررسی طول متن
            if len(cleaned_text) < 10:  # رد توییت‌های خیلی کوتاه
                return None

            # محاسبه هش متن برای تشخیص تکرار
            text_hash = compute_similarity_hash(cleaned_text)

            # بررسی تکراری بودن متن
            if self.db.is_duplicate_news(cleaned_text):
                return None

            # استخراج URL‌ها، هشتگ‌ها و منشن‌ها
            urls = []
            for url in tweet.outlinks or []:
                if url and "twitter.com" not in url:  # حذف لینک‌های داخلی توییتر
                    urls.append(url)

            # استخراج هشتگ‌ها
            hashtags = [tag.lower() for tag in (tweet.hashtags or [])]

            # استخراج منشن‌ها
            mentions = tweet.mentionedUsers or []
            mentioned_usernames = [user.username for user in mentions] if mentions else []

            # بررسی و استخراج تصاویر
            media_urls = []
            has_media = False

            if hasattr(tweet, 'media'):
                media = tweet.media
                if media:
                    for item in media:
                        if hasattr(item, 'fullUrl'):
                            media_urls.append(item.fullUrl)
                            has_media = True

            # ساخت شیء توییت
            tweet_data = {
                "external_id": tweet_id,
                "source": NewsSource.TWITTER.value,
                "source_name": tweet.user.username,
                "text": cleaned_text,
                "title": None,  # توییتر عنوان ندارد
                "url": f"https://twitter.com/{tweet.user.username}/status/{tweet_id}",
                "has_media": has_media,
                "media_urls": media_urls if media_urls else None,
                "published_at": tweet.date,
                "meta": {
                    "user_id": tweet.user.id,
                    "user_name": tweet.user.username,
                    "user_display_name": tweet.user.displayname,
                    "user_followers": getattr(tweet.user, 'followersCount', 0),
                    "user_verified": getattr(tweet.user, 'verified', False),
                    "retweet_count": getattr(tweet, 'retweetCount', 0),
                    "like_count": getattr(tweet, 'likeCount', 0),
                    "reply_count": getattr(tweet, 'replyCount', 0),
                    "quote_count": getattr(tweet, 'quoteCount', 0),
                    "is_retweet": getattr(tweet, 'isRetweet', False),
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
            # افزودن به دیتابیس
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

    def _build_search_query(self, query_term: str, since_id: Optional[str] = None, days_limit: int = None) -> str:
        """
        ساخت کوئری جستجو برای snscrape

        Args:
            query_term: عبارت جستجو (کاربر یا هشتگ)
            since_id: شناسه آخرین توییت (اختیاری)
            days_limit: محدودیت روز (اختیاری)

        Returns:
            کوئری جستجو
        """
        # بررسی نوع کوئری
        if query_term.startswith('@'):
            # جستجو بر اساس کاربر
            username = query_term[1:]
            query = f"from:{username}"
        elif query_term.startswith('#'):
            # جستجو بر اساس هشتگ
            hashtag = query_term[1:]
            query = f"#{hashtag}"
        else:
            # جستجوی عادی
            query = query_term

        # اضافه کردن محدودیت زمانی
        if days_limit is not None:
            since_date = (now() - timedelta(days=days_limit)).strftime('%Y-%m-%d')
            query += f" since:{since_date}"

        # اضافه کردن محدودیت زبان (اختیاری)
        # query += " lang:en OR lang:fa"

        # حذف ریتوییت‌ها
        query += " -filter:retweets"

        return query

    async def _collect_tweets_with_snscrape(self, query: str, max_tweets: int) -> List[Any]:
        """
        جمع‌آوری توییت‌ها با استفاده از snscrape

        Args:
            query: کوئری جستجو
            max_tweets: حداکثر تعداد توییت

        Returns:
            لیست توییت‌ها
        """
        tweets = []

        # استفاده از scraper توییتر
        scraper = sntwitter.TwitterSearchScraper(query)

        # محدود کردن تعداد توییت‌ها
        for i, tweet in enumerate(scraper.get_items()):
            if i >= max_tweets:
                break
            tweets.append(tweet)

        self.logger.info(f"تعداد {len(tweets)} توییت برای کوئری '{query}' جمع‌آوری شد")
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
            # تبدیل نام کاربری به فرمت مناسب
            query_key = query_term.replace('@', '').replace('#', '')

            # دریافت شناسه آخرین توییت
            last_tweet_id = await self.get_last_tweet_id(query_key)

            # ساخت کوئری جستجو
            search_query = self._build_search_query(query_term, last_tweet_id, self.days_limit)

            # جمع‌آوری توییت‌ها
            tweets = await self._collect_tweets_with_snscrape(search_query, self.max_tweets_per_query)

            if not tweets:
                self.logger.info(f"هیچ توییت جدیدی برای '{query_term}' یافت نشد")
                return 0, 0

            # مرتب‌سازی توییت‌ها بر اساس تاریخ (جدید به قدیم)
            tweets.sort(key=lambda x: x.date, reverse=True)

            # ذخیره شناسه جدیدترین توییت
            newest_id = str(tweets[0].id)
            await self.set_last_tweet_id(query_key, newest_id)

            # پردازش و ذخیره توییت‌ها
            for tweet in tweets:
                processed_count += 1

                # پردازش توییت
                tweet_data = await self.process_tweet(tweet)

                if tweet_data:
                    # ذخیره توییت
                    if await self.save_tweet(tweet_data):
                        saved_count += 1

                # تأخیر کوتاه برای کاهش بار
                await asyncio.sleep(0.1)

            self.logger.info(
                f"جمع‌آوری از '{query_term}' پایان یافت. پردازش‌شده: {processed_count}, ذخیره‌شده: {saved_count}")
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
            # اضافه کردن @ به ابتدای نام کاربری اگر نداشته باشد
            if not account.startswith('@'):
                account = f"@{account}"

            try:
                processed, saved = await self.collect_from_twitter(account)
                results[account] = (processed, saved)

                # تأخیر بین حساب‌ها
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
            # اضافه کردن # به ابتدای هشتگ اگر نداشته باشد
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"

            try:
                processed, saved = await self.collect_from_twitter(hashtag)
                results[hashtag] = (processed, saved)

                # تأخیر بین هشتگ‌ها
                await asyncio.sleep(2)

            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری از هشتگ {hashtag}: {str(e)}")
                results[hashtag] = (0, 0)

        return results

    async def run(self, accounts: Optional[List[str]] = None, hashtags: Optional[List[str]] = None) -> Dict[
        str, Dict[str, Tuple[int, int]]]:
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
            # جمع‌آوری از حساب‌های کاربری
            if accounts or self.accounts:
                results["accounts"] = await self.collect_from_accounts(accounts)

            # جمع‌آوری از هشتگ‌ها
            if hashtags or self.hashtags:
                results["hashtags"] = await self.collect_from_hashtags(hashtags)

            return results

        except Exception as e:
            self.logger.error(f"خطا در اجرای جمع‌کننده توییتر: {str(e)}")
            return results
