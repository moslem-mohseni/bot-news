"""
ماژول مدیریت جمع‌آوری داده برای CryptoNewsBot

این ماژول مسئول هماهنگی و مدیریت کلیه فرآیندهای جمع‌آوری داده از منابع مختلف است.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import now
from ..cache.redis_manager import RedisManager
from ..database.query_manager import QueryManager

from .telegram_collector import TelegramCollector
from .website_collector import WebsiteCollector
from .twitter_collector import TwitterCollector


class CollectionManager:
    """
    کلاس مدیریت جمع‌آوری داده

    این کلاس مسئول مدیریت و هماهنگی تمام فرآیندهای جمع‌آوری داده از منابع مختلف است.
    از الگوی Strategy برای انتزاع مکانیسم‌های جمع‌آوری استفاده می‌کند.
    """

    def __init__(self) -> None:
        """
        مقداردهی اولیه مدیریت جمع‌آوری
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.db = QueryManager()

        # کلید‌های ذخیره در ردیس
        self.last_run_key = "collection:last_run"
        self.collection_stats_key = "collection:stats"

        # فواصل جمع‌آوری (ثانیه)
        self.telegram_interval = self.config._get_env_int("TELEGRAM_COLLECTION_INTERVAL", 3600)  # 1 ساعت
        self.website_interval = self.config._get_env_int("WEBSITE_COLLECTION_INTERVAL", 7200)  # 2 ساعت
        self.twitter_interval = self.config._get_env_int("TWITTER_COLLECTION_INTERVAL", 3600)  # 1 ساعت

        # آخرین زمان اجرای هر منبع
        self.last_runs = self._load_last_runs()

    def _load_last_runs(self) -> Dict[str, datetime]:
        """
        بارگذاری زمان آخرین اجرای هر منبع

        Returns:
            دیکشنری زمان‌های آخرین اجرا
        """
        default_time = now() - timedelta(days=1)  # پیش‌فرض: 1 روز قبل

        last_runs_json = self.redis.get(self.last_run_key)
        if not last_runs_json:
            return {
                "telegram": default_time,
                "website": default_time,
                "twitter": default_time
            }

        try:
            last_runs_dict = json.loads(last_runs_json)
            result = {}

            for source, time_str in last_runs_dict.items():
                try:
                    result[source] = datetime.fromisoformat(time_str)
                except (ValueError, TypeError):
                    result[source] = default_time

            # اطمینان از وجود همه منابع
            for source in ["telegram", "website", "twitter"]:
                if source not in result:
                    result[source] = default_time

            return result

        except Exception as e:
            self.logger.error(f"خطا در بارگذاری زمان‌های آخرین اجرا: {str(e)}")
            return {
                "telegram": default_time,
                "website": default_time,
                "twitter": default_time
            }

    def _save_last_runs(self) -> None:
        """
        ذخیره زمان آخرین اجرای هر منبع
        """
        try:
            last_runs_dict = {
                source: time.isoformat() for source, time in self.last_runs.items()
            }
            self.redis.set(self.last_run_key, json.dumps(last_runs_dict))
        except Exception as e:
            self.logger.error(f"خطا در ذخیره زمان‌های آخرین اجرا: {str(e)}")

    def _update_last_run(self, source: str) -> None:
        """
        به‌روزرسانی زمان آخرین اجرای یک منبع

        Args:
            source: نام منبع
        """
        self.last_runs[source] = now()
        self._save_last_runs()

    def _should_collect(self, source: str) -> bool:
        """
        بررسی اینکه آیا باید از یک منبع جمع‌آوری انجام شود

        Args:
            source: نام منبع

        Returns:
            وضعیت جمع‌آوری
        """
        last_run = self.last_runs.get(source, now() - timedelta(days=1))
        current_time = now()

        if source == "telegram":
            interval = self.telegram_interval
        elif source == "website":
            interval = self.website_interval
        elif source == "twitter":
            interval = self.twitter_interval
        else:
            interval = 3600  # پیش‌فرض: 1 ساعت

        # بررسی گذشت زمان کافی
        return (current_time - last_run).total_seconds() >= interval

    def update_collection_stats(self, source: str, stats: Dict[str, Any]) -> None:
        """
        به‌روزرسانی آمار جمع‌آوری

        Args:
            source: نام منبع
            stats: آمار جمع‌آوری
        """
        try:
            # دریافت آمار قبلی
            stats_json = self.redis.get(self.collection_stats_key)
            if stats_json:
                collection_stats = json.loads(stats_json)
            else:
                collection_stats = {}

            # به‌روزرسانی آمار
            collection_stats[source] = {
                "last_collection": now().isoformat(),
                "stats": stats
            }

            # ذخیره آمار جدید
            self.redis.set(self.collection_stats_key, json.dumps(collection_stats))

        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی آمار جمع‌آوری: {str(e)}")

    async def collect_from_telegram(self, force: bool = False) -> Dict[str, Tuple[int, int]]:
        """
        جمع‌آوری داده از تلگرام

        Args:
            force: اجبار به جمع‌آوری بدون در نظر گرفتن زمان آخرین اجرا

        Returns:
            نتایج جمع‌آوری
        """
        if not force and not self._should_collect("telegram"):
            self.logger.info("فاصله زمانی کافی برای جمع‌آوری از تلگرام سپری نشده است")
            return {}

        try:
            # جمع‌آوری از تلگرام
            self.logger.info("شروع جمع‌آوری داده از تلگرام")
            collector = TelegramCollector()
            results = await collector.run()

            # به‌روزرسانی زمان آخرین اجرا
            self._update_last_run("telegram")

            # به‌روزرسانی آمار
            self.update_collection_stats("telegram", {
                "channels": len(results),
                "total_processed": sum(r[0] for r in results.values()),
                "total_saved": sum(r[1] for r in results.values())
            })

            # گزارش نتایج
            total_processed = sum(r[0] for r in results.values())
            total_saved = sum(r[1] for r in results.values())
            self.logger.info(
                f"جمع‌آوری از تلگرام پایان یافت. کانال‌ها: {len(results)}, کل پردازش‌شده: {total_processed}, کل ذخیره‌شده: {total_saved}")

            return results

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از تلگرام: {str(e)}")
            return {}

    async def collect_from_websites(self, force: bool = False) -> Dict[str, Tuple[int, int, int]]:
        """
        جمع‌آوری داده از وب‌سایت‌ها

        Args:
            force: اجبار به جمع‌آوری بدون در نظر گرفتن زمان آخرین اجرا

        Returns:
            نتایج جمع‌آوری
        """
        if not force and not self._should_collect("website"):
            self.logger.info("فاصله زمانی کافی برای جمع‌آوری از وب‌سایت‌ها سپری نشده است")
            return {}

        try:
            # جمع‌آوری از وب‌سایت‌ها
            self.logger.info("شروع جمع‌آوری داده از وب‌سایت‌ها")
            collector = WebsiteCollector()
            results = await collector.run()

            # به‌روزرسانی زمان آخرین اجرا
            self._update_last_run("website")

            # به‌روزرسانی آمار
            self.update_collection_stats("website", {
                "websites": len(results),
                "total_urls": sum(r[0] for r in results.values()),
                "total_processed": sum(r[1] for r in results.values()),
                "total_saved": sum(r[2] for r in results.values())
            })

            # گزارش نتایج
            total_urls = sum(r[0] for r in results.values())
            total_processed = sum(r[1] for r in results.values())
            total_saved = sum(r[2] for r in results.values())
            self.logger.info(
                f"جمع‌آوری از وب‌سایت‌ها پایان یافت. سایت‌ها: {len(results)}, کل URL‌ها: {total_urls}, کل پردازش‌شده: {total_processed}, کل ذخیره‌شده: {total_saved}")

            return results

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از وب‌سایت‌ها: {str(e)}")
            return {}

    async def collect_from_twitter(self, force: bool = False) -> Dict[str, Dict[str, Tuple[int, int]]]:
        """
        جمع‌آوری داده از توییتر

        Args:
            force: اجبار به جمع‌آوری بدون در نظر گرفتن زمان آخرین اجرا

        Returns:
            نتایج جمع‌آوری
        """
        if not force and not self._should_collect("twitter"):
            self.logger.info("فاصله زمانی کافی برای جمع‌آوری از توییتر سپری نشده است")
            return {"accounts": {}, "hashtags": {}}

        try:
            # جمع‌آوری از توییتر
            self.logger.info("شروع جمع‌آوری داده از توییتر")
            collector = TwitterCollector()
            results = await collector.run()

            # به‌روزرسانی زمان آخرین اجرا
            self._update_last_run("twitter")

            # آمار حساب‌ها
            accounts_count = len(results.get("accounts", {}))
            accounts_processed = sum(r[0] for r in results.get("accounts", {}).values())
            accounts_saved = sum(r[1] for r in results.get("accounts", {}).values())

            # آمار هشتگ‌ها
            hashtags_count = len(results.get("hashtags", {}))
            hashtags_processed = sum(r[0] for r in results.get("hashtags", {}).values())
            hashtags_saved = sum(r[1] for r in results.get("hashtags", {}).values())

            # به‌روزرسانی آمار
            self.update_collection_stats("twitter", {
                "accounts": {
                    "count": accounts_count,
                    "processed": accounts_processed,
                    "saved": accounts_saved
                },
                "hashtags": {
                    "count": hashtags_count,
                    "processed": hashtags_processed,
                    "saved": hashtags_saved
                },
                "total_processed": accounts_processed + hashtags_processed,
                "total_saved": accounts_saved + hashtags_saved
            })

            # گزارش نتایج
            total_processed = accounts_processed + hashtags_processed
            total_saved = accounts_saved + hashtags_saved
            self.logger.info(
                f"جمع‌آوری از توییتر پایان یافت. حساب‌ها: {accounts_count}, هشتگ‌ها: {hashtags_count}, کل پردازش‌شده: {total_processed}, کل ذخیره‌شده: {total_saved}")

            return results

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از توییتر: {str(e)}")
            return {"accounts": {}, "hashtags": {}}

    async def collect_all(self, force: bool = False) -> Dict[str, Any]:
        """
        جمع‌آوری داده از تمام منابع

        Args:
            force: اجبار به جمع‌آوری بدون در نظر گرفتن زمان آخرین اجرا

        Returns:
            نتایج جمع‌آوری
        """
        self.logger.info("شروع جمع‌آوری داده از تمام منابع")
        start_time = time.time()

        # جمع‌آوری از منابع به صورت موازی
        tasks = [
            self.collect_from_telegram(force),
            self.collect_from_websites(force),
            self.collect_from_twitter(force)
        ]

        # اجرای موازی
        telegram_results, website_results, twitter_results = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        # جمع‌بندی نتایج
        results = {
            "telegram": telegram_results,
            "website": website_results,
            "twitter": twitter_results,
            "summary": {
                "telegram": {
                    "channels": len(telegram_results),
                    "processed": sum(r[0] for r in telegram_results.values()),
                    "saved": sum(r[1] for r in telegram_results.values())
                },
                "website": {
                    "websites": len(website_results),
                    "urls": sum(r[0] for r in website_results.values()),
                    "processed": sum(r[1] for r in website_results.values()),
                    "saved": sum(r[2] for r in website_results.values())
                },
                "twitter": {
                    "accounts": len(twitter_results.get("accounts", {})),
                    "hashtags": len(twitter_results.get("hashtags", {})),
                    "processed": sum(r[0] for r in twitter_results.get("accounts", {}).values()) +
                                 sum(r[0] for r in twitter_results.get("hashtags", {}).values()),
                    "saved": sum(r[1] for r in twitter_results.get("accounts", {}).values()) +
                             sum(r[1] for r in twitter_results.get("hashtags", {}).values())
                },
                "elapsed_time": elapsed_time
            }
        }

        # گزارش نتایج
        total_processed = (
                results["summary"]["telegram"]["processed"] +
                results["summary"]["website"]["processed"] +
                results["summary"]["twitter"]["processed"]
        )

        total_saved = (
                results["summary"]["telegram"]["saved"] +
                results["summary"]["website"]["saved"] +
                results["summary"]["twitter"]["saved"]
        )

        self.logger.info(
            f"جمع‌آوری از تمام منابع پایان یافت. کل پردازش‌شده: {total_processed}, کل ذخیره‌شده: {total_saved}, زمان: {elapsed_time:.2f} ثانیه")

        return results

    async def run_once(self, force: bool = False) -> Dict[str, Any]:
        """
        اجرای یک بار جمع‌آوری از تمام منابع

        Args:
            force: اجبار به جمع‌آوری بدون در نظر گرفتن زمان آخرین اجرا

        Returns:
            نتایج جمع‌آوری
        """
        return await self.collect_all(force)

    async def run_continuously(self, interval: int = 600, stop_event: asyncio.Event = None) -> None:
        """
        اجرای مداوم جمع‌آوری با فاصله زمانی مشخص

        Args:
            interval: فاصله زمانی بین اجراها (ثانیه)
            stop_event: رویداد توقف (اختیاری)
        """
        self.logger.info(f"شروع اجرای مداوم جمع‌آوری با فاصله زمانی {interval} ثانیه")

        if stop_event is None:
            stop_event = asyncio.Event()

        while not stop_event.is_set():
            try:
                # اجرای جمع‌آوری
                await self.collect_all()

                # انتظار تا اجرای بعدی
                self.logger.info(f"منتظر {interval} ثانیه تا اجرای بعدی...")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass  # زمان انتظار به پایان رسید

            except Exception as e:
                self.logger.error(f"خطا در اجرای مداوم جمع‌آوری: {str(e)}")
                # انتظار کوتاه قبل از تلاش مجدد
                await asyncio.sleep(60)

        self.logger.info("اجرای مداوم جمع‌آوری متوقف شد")
