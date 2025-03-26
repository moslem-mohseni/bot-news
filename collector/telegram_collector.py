"""
ماژول جمع‌آوری داده از کانال‌های تلگرام برای CryptoNewsBot

این ماژول مسئول جمع‌آوری اخبار از کانال‌های تلگرام و پردازش اولیه آنها است.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging

from telethon import TelegramClient, errors
from telethon.tl.types import Channel, Message, MessageMedia, PeerChannel
from telethon.tl.functions.channels import GetFullChannelRequest

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import (
    clean_text, extract_urls, extract_hashtags, compute_similarity_hash,
    now, retry
)
from ..cache.redis_manager import RedisManager
from ..database.query_manager import QueryManager
from ..database.models import NewsSource


class TelegramCollector:
    """
    کلاس جمع‌آوری داده‌ها از کانال‌های تلگرام

    این کلاس مسئول اتصال به تلگرام و جمع‌آوری پیام‌ها از کانال‌های مختلف است.
    """

    def __init__(self, session_name: Optional[str] = None) -> None:
        """
        مقداردهی اولیه جمع‌کننده تلگرام

        Args:
            session_name: نام سشن تلگرام (اختیاری)
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.db = QueryManager()

        # تنظیمات تلگرام
        self.api_id = self.config.telegram_api_id
        self.api_hash = self.config.telegram_api_hash
        self.phone = self.config.telegram_phone
        self.session_name = session_name or self.config.telegram_session_name

        # لیست کانال‌های مورد نظر برای جمع‌آوری
        self.channels = self.config.telegram_channels

        # مقدار حداکثر پیام برای جمع‌آوری در هر درخواست
        self.default_limit = 100

        # حداکثر تعداد روز برای جمع‌آوری پیام‌های قدیمی‌تر
        self.max_days_back = 3

        # کلید‌های ذخیره در ردیس
        self.last_message_key_prefix = "telegram:last_msg_id:"
        self.channel_info_key_prefix = "telegram:channel_info:"

        # کلاینت تلگرام
        self.client = None
        self.is_connected = False

    async def connect(self) -> bool:
        """
        اتصال به تلگرام

        Returns:
            وضعیت اتصال
        """
        try:
            # ایجاد دایرکتوری سشن اگر وجود نداشته باشد
            session_dir = os.path.dirname(self.session_name)
            if session_dir and not os.path.exists(session_dir):
                os.makedirs(session_dir)

            # ایجاد کلاینت تلگرام
            self.client = TelegramClient(
                self.session_name,
                self.api_id,
                self.api_hash,
                device_model="CryptoNewsBot",
                system_version="1.0",
                app_version="1.0",
                lang_code="fa",
                proxy=None  # می‌توان پروکسی را اینجا تنظیم کرد
            )

            # اتصال به تلگرام
            await self.client.connect()

            # بررسی احراز هویت
            if not await self.client.is_user_authorized():
                # درخواست کد تأیید
                await self.client.send_code_request(self.phone)
                self.logger.info(f"کد تأیید به شماره {self.phone} ارسال شد. لطفاً کد را وارد کنید:")

                verification_code = input("کد تأیید: ")

                # ورود با کد تأیید
                await self.client.sign_in(self.phone, verification_code)

            self.is_connected = True
            self.logger.info("اتصال به تلگرام با موفقیت برقرار شد")
            return True

        except errors.PhoneCodeInvalidError:
            self.logger.error("کد تأیید نامعتبر است")
            return False
        except Exception as e:
            self.logger.error(f"خطا در اتصال به تلگرام: {str(e)}")
            return False

    async def disconnect(self) -> None:
        """
        قطع اتصال از تلگرام
        """
        if self.client:
            await self.client.disconnect()
            self.is_connected = False
            self.logger.info("اتصال به تلگرام قطع شد")

    async def get_channel_info(self, channel_username: str) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات یک کانال

        Args:
            channel_username: نام کاربری کانال

        Returns:
            اطلاعات کانال یا None
        """
        # بررسی کش
        cache_key = f"{self.channel_info_key_prefix}{channel_username}"
        cached_info = self.redis.get_json(cache_key)

        if cached_info:
            return cached_info

        try:
            if not self.is_connected:
                await self.connect()

            # دریافت entity کانال
            channel_entity = await self.client.get_entity(channel_username)

            # دریافت اطلاعات کامل کانال
            full_channel = await self.client(GetFullChannelRequest(channel=channel_entity))

            # استخراج اطلاعات مورد نیاز
            channel_info = {
                "id": channel_entity.id,
                "username": channel_username,
                "title": getattr(channel_entity, "title", channel_username),
                "participants_count": getattr(full_channel.full_chat, "participants_count", 0),
                "description": getattr(full_channel.full_chat, "about", ""),
                "linked_chat_id": getattr(full_channel.full_chat, "linked_chat_id", None),
                "last_checked": now().isoformat()
            }

            # ذخیره در کش (7 روز)
            self.redis.set_json(cache_key, channel_info, expire=7 * 24 * 60 * 60)

            return channel_info

        except Exception as e:
            self.logger.error(f"خطا در دریافت اطلاعات کانال {channel_username}: {str(e)}")
            return None

    async def get_last_message_id(self, channel_username: str) -> int:
        """
        دریافت شناسه آخرین پیام پردازش شده از یک کانال

        Args:
            channel_username: نام کاربری کانال

        Returns:
            شناسه آخرین پیام یا 0
        """
        key = f"{self.last_message_key_prefix}{channel_username}"
        last_id = self.redis.get(key)

        if last_id:
            try:
                return int(last_id)
            except (ValueError, TypeError):
                return 0
        return 0

    async def set_last_message_id(self, channel_username: str, message_id: int) -> None:
        """
        تنظیم شناسه آخرین پیام پردازش شده از یک کانال

        Args:
            channel_username: نام کاربری کانال
            message_id: شناسه پیام
        """
        key = f"{self.last_message_key_prefix}{channel_username}"
        self.redis.set(key, str(message_id))

    async def process_media(self, message: Message) -> Optional[List[str]]:
        """
        پردازش رسانه یک پیام

        Args:
            message: پیام تلگرام

        Returns:
            لیست آدرس‌های رسانه یا None
        """
        if not message.media:
            return None

        media_urls = []
        try:
            # بررسی نوع رسانه و دانلود آن
            # در این پیاده‌سازی فعلی، فقط URL را برمی‌گردانیم
            # می‌توان این بخش را برای دانلود و ذخیره فایل‌ها گسترش داد

            if hasattr(message.media, 'photo'):
                # آدرس عکس
                media_urls.append(f"telegram:photo:{message.id}")

            elif hasattr(message.media, 'document'):
                # آدرس سند
                media_urls.append(f"telegram:document:{message.id}")

            # می‌توان انواع دیگر رسانه را نیز اضافه کرد

            return media_urls if media_urls else None

        except Exception as e:
            self.logger.error(f"خطا در پردازش رسانه پیام {message.id}: {str(e)}")
            return None

    async def process_message(self, message: Message, channel_username: str) -> Optional[Dict[str, Any]]:
        """
        پردازش یک پیام و استخراج اطلاعات آن

        Args:
            message: پیام تلگرام
            channel_username: نام کاربری کانال

        Returns:
            اطلاعات پردازش شده پیام یا None
        """
        try:
            # بررسی پیام خالی
            if not message.text and not message.message:
                return None

            # استخراج متن پیام
            text = message.text or message.message

            # بررسی طول متن (رد پیام‌های کوتاه)
            if len(text) < 10:
                return None

            # تمیز کردن متن
            cleaned_text = clean_text(text)

            # محاسبه هش متن برای تشخیص تکرار
            text_hash = compute_similarity_hash(cleaned_text)

            # استخراج URL‌ها
            urls = extract_urls(text)

            # استخراج هشتگ‌ها
            hashtags = extract_hashtags(text)

            # پردازش رسانه
            media_urls = await self.process_media(message)

            # ایجاد شیء پیام
            message_data = {
                "external_id": str(message.id),
                "source": NewsSource.TELEGRAM.value,
                "source_name": channel_username,
                "text": cleaned_text,
                "title": None,  # تلگرام عنوان ندارد
                "url": None,  # اگر URL در متن باشد، در تحلیل استخراج می‌شود
                "has_media": bool(media_urls),
                "media_urls": media_urls,
                "published_at": message.date,
                "meta": {
                    "urls": urls,
                    "hashtags": hashtags,
                    "text_hash": text_hash,
                    "views": getattr(message, "views", 0),
                    "forwards": getattr(message, "forwards", 0),
                    "replies": getattr(message, "replies", None),
                    "post_author": getattr(message, "post_author", None),
                }
            }

            return message_data

        except Exception as e:
            self.logger.error(f"خطا در پردازش پیام {message.id}: {str(e)}")
            return None

    async def save_message(self, message_data: Dict[str, Any]) -> bool:
        """
        ذخیره اطلاعات یک پیام در دیتابیس

        Args:
            message_data: اطلاعات پیام

        Returns:
            وضعیت ذخیره‌سازی
        """
        try:
            # بررسی تکراری بودن بر اساس منبع و شناسه خارجی
            # این بررسی در QueryManager انجام می‌شود

            # افزودن به دیتابیس
            raw_news = self.db.add_raw_news(
                external_id=message_data["external_id"],
                source=message_data["source"],
                source_name=message_data["source_name"],
                text=message_data["text"],
                title=message_data["title"],
                url=message_data["url"],
                has_media=message_data["has_media"],
                media_urls=message_data["media_urls"],
                published_at=message_data["published_at"]
            )

            if raw_news:
                return True
            return False

        except Exception as e:
            self.logger.error(f"خطا در ذخیره پیام {message_data['external_id']}: {str(e)}")
            return False

    @retry(max_tries=3, delay=5, backoff=2)
    async def collect_from_channel(self, channel_username: str, limit: Optional[int] = None) -> Tuple[int, int]:
        """
        جمع‌آوری پیام‌های یک کانال

        Args:
            channel_username: نام کاربری کانال
            limit: حداکثر تعداد پیام‌های جمع‌آوری شده

        Returns:
            تعداد کل پیام‌ها، تعداد پیام‌های ذخیره شده
        """
        if not self.is_connected:
            if not await self.connect():
                return 0, 0

        limit = limit or self.default_limit
        processed_count = 0
        saved_count = 0

        try:
            # دریافت شناسه آخرین پیام پردازش شده
            last_message_id = await self.get_last_message_id(channel_username)

            # دریافت entity کانال
            channel_entity = await self.client.get_entity(channel_username)

            # تعیین حداکثر زمان برای پیام‌های قدیمی
            min_date = now() - timedelta(days=self.max_days_back)

            # جمع‌آوری پیام‌ها
            messages = []
            newest_id = 0

            # استفاده از iterator برای کنترل بهتر حافظه
            async for message in self.client.iter_messages(
                    channel_entity,
                    limit=limit,
                    min_id=last_message_id,
                    reverse=True  # از قدیمی به جدید
            ):
                # بررسی محدودیت زمانی
                if message.date < min_date:
                    continue

                # پردازش پیام
                message_data = await self.process_message(message, channel_username)
                processed_count += 1

                # به‌روزرسانی جدیدترین شناسه
                if message.id > newest_id:
                    newest_id = message.id

                # ذخیره پیام پردازش شده
                if message_data:
                    if await self.save_message(message_data):
                        saved_count += 1

                # تأخیر کوتاه برای جلوگیری از محدودیت نرخ
                await asyncio.sleep(0.1)

            # به‌روزرسانی آخرین شناسه پیام پردازش شده
            if newest_id > 0:
                await self.set_last_message_id(channel_username, newest_id)

            self.logger.info(
                f"جمع‌آوری از کانال {channel_username} پایان یافت. پردازش‌شده: {processed_count}, ذخیره‌شده: {saved_count}")
            return processed_count, saved_count

        except errors.FloodWaitError as e:
            wait_time = e.seconds
            self.logger.warning(f"محدودیت نرخ تلگرام برای کانال {channel_username}. انتظار برای {wait_time} ثانیه")
            await asyncio.sleep(wait_time)
            # تلاش مجدد بعد از انتظار (توسط دکوراتور retry)
            raise

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از کانال {channel_username}: {str(e)}")
            return processed_count, saved_count

    async def collect_all(self, channels: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        جمع‌آوری پیام‌های تمام کانال‌های تنظیم شده

        Args:
            channels: لیست کانال‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری برای هر کانال
        """
        # اتصال به تلگرام
        if not self.is_connected:
            if not await self.connect():
                return {}

        channels_to_collect = channels or self.channels
        results = {}

        for channel in channels_to_collect:
            try:
                processed, saved = await self.collect_from_channel(channel)
                results[channel] = (processed, saved)

                # تأخیر بین کانال‌ها
                await asyncio.sleep(2)

            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری کانال {channel}: {str(e)}")
                results[channel] = (0, 0)

        return results

    async def run(self, channels: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        اجرای جمع‌کننده تلگرام

        Args:
            channels: لیست کانال‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری
        """
        try:
            # جمع‌آوری از تمام کانال‌ها
            results = await self.collect_all(channels)

            # قطع اتصال در پایان
            await self.disconnect()

            return results

        except Exception as e:
            self.logger.error(f"خطا در اجرای جمع‌کننده تلگرام: {str(e)}")
            await self.disconnect()
            return {}
