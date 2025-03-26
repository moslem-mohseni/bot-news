"""
ماژول جمع‌آوری داده از وب‌سایت‌های خبری برای CryptoNewsBot

این ماژول مسئول جمع‌آوری اخبار از وب‌سایت‌های خبری و پردازش اولیه آنها است.
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import re
import hashlib
import aiohttp
from bs4 import BeautifulSoup, SoupStrainer
from newspaper import Article, ArticleException
import validators

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import (
    clean_text, extract_urls, extract_hashtags, compute_similarity_hash,
    now, retry, is_valid_url
)
from ..cache.redis_manager import RedisManager
from ..database.query_manager import QueryManager
from ..database.models import NewsSource


class WebsiteCollector:
    """
    کلاس جمع‌آوری داده‌ها از وب‌سایت‌های خبری

    این کلاس مسئول جمع‌آوری اخبار از وب‌سایت‌های خبری مختلف با استفاده از
    اسکراپینگ و کتابخانه Newspaper است.
    """

    def __init__(self) -> None:
        """
        مقداردهی اولیه جمع‌کننده وب‌سایت
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.redis = RedisManager()
        self.db = QueryManager()

        # لیست وب‌سایت‌های خبری
        self.websites = self.config.websites or []

        # کلیدهای ذخیره در ردیس
        self.last_fetch_key_prefix = "website:last_fetch:"
        self.processed_urls_key = "website:processed_urls"

        # تنظیمات HTTP
        self.timeout = 30  # زمان انتظار به ثانیه
        self.max_retries = 3  # حداکثر تلاش مجدد
        self.retry_delay = 2  # تأخیر بین تلاش‌ها به ثانیه

        # کش روبات‌ها
        self.robots_cache = {}

        # هدرهای HTTP
        self.headers = [
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            },
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0",
            }
        ]

    def _get_random_headers(self) -> Dict[str, str]:
        """
        انتخاب یک هدر HTTP تصادفی

        Returns:
            هدر تصادفی
        """
        return random.choice(self.headers)

    async def check_robots_allowed(self, session: aiohttp.ClientSession, url: str) -> bool:
        """
        بررسی مجاز بودن URL با توجه به robots.txt

        Args:
            session: جلسه HTTP
            url: آدرس وب‌سایت

        Returns:
            مجاز بودن URL
        """
        try:
            from urllib.parse import urlparse
            from urllib.robotparser import RobotFileParser

            # استخراج دامنه اصلی
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # بررسی کش
            if base_url in self.robots_cache:
                robot_parser = self.robots_cache[base_url]
            else:
                # دریافت فایل robots.txt
                robots_url = f"{base_url}/robots.txt"
                robot_parser = RobotFileParser()

                try:
                    async with session.get(robots_url, headers=self._get_random_headers(), timeout=5) as resp:
                        if resp.status == 200:
                            robots_txt = await resp.text()
                            # تنظیم فایل robots.txt
                            robot_parser.parse(robots_txt.splitlines())
                        else:
                            # در صورت عدم وجود فایل، همه چیز مجاز است
                            robot_parser.allow_all = True
                except Exception:
                    # در صورت خطا، همه چیز مجاز است
                    robot_parser.allow_all = True

                # ذخیره در کش
                self.robots_cache[base_url] = robot_parser

            # بررسی مجاز بودن URL
            return robot_parser.can_fetch("*", url)

        except Exception as e:
            self.logger.warning(f"خطا در بررسی robots.txt برای {url}: {str(e)}")
            return True  # در صورت بروز خطا، مجاز فرض می‌کنیم

    async def is_processed_url(self, url: str) -> bool:
        """
        بررسی اینکه آیا URL قبلاً پردازش شده است

        Args:
            url: آدرس وب‌سایت

        Returns:
            وضعیت پردازش قبلی
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.redis.set_is_member(self.processed_urls_key, url_hash)

    async def mark_url_processed(self, url: str) -> None:
        """
        علامت‌گذاری URL به عنوان پردازش شده

        Args:
            url: آدرس وب‌سایت
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        self.redis.set_add(self.processed_urls_key, url_hash)

    async def clean_expired_processed_urls(self, days: int = 30) -> int:
        """
        پاکسازی URL‌های پردازش شده قدیمی

        Args:
            days: تعداد روزهای نگهداری

        Returns:
            تعداد URL‌های حذف شده
        """
        # در این پیاده‌سازی ساده، تمام URL‌ها را حفظ می‌کنیم
        # می‌توان در نسخه‌های بعدی از مجموعه با کلیدهای انقضادار استفاده کرد
        return 0

    @retry(max_tries=3, delay=2, backoff=2)
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """
        دریافت محتوای یک URL

        Args:
            session: جلسه HTTP
            url: آدرس وب‌سایت

        Returns:
            محتوای HTML یا None
        """
        try:
            # بررسی مجاز بودن بر اساس robots.txt
            if not await self.check_robots_allowed(session, url):
                self.logger.warning(f"URL {url} توسط robots.txt مجاز نیست")
                return None

            # دریافت محتوا
            headers = self._get_random_headers()
            async with session.get(url, headers=headers, timeout=self.timeout) as resp:
                if resp.status != 200:
                    self.logger.warning(f"خطا در دریافت {url}: کد وضعیت {resp.status}")
                    return None

                # بررسی نوع محتوا
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    self.logger.warning(f"نوع محتوای نامعتبر برای {url}: {content_type}")
                    return None

                # دریافت HTML
                html = await resp.text()

                # تأخیر برای کاهش بار روی سرور
                await asyncio.sleep(random.uniform(1.0, 3.0))

                return html

        except asyncio.TimeoutError:
            self.logger.warning(f"تایم‌اوت در دریافت {url}")
            raise  # برای retry
        except Exception as e:
            self.logger.error(f"خطا در دریافت {url}: {str(e)}")
            return None

    async def extract_links(self, html: str, base_url: str) -> List[str]:
        """
        استخراج لینک‌های خبر از HTML

        Args:
            html: محتوای HTML
            base_url: URL پایه

        Returns:
            لیست لینک‌ها
        """
        links = []
        try:
            # استفاده از SoupStrainer برای بهینه‌سازی حافظه
            only_links = SoupStrainer('a')
            soup = BeautifulSoup(html, 'html.parser', parse_only=only_links)

            from urllib.parse import urljoin, urlparse

            # دامنه اصلی برای فیلتر کردن لینک‌های خارجی
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # ساخت URL کامل
                full_url = urljoin(base_url, href)

                # بررسی اعتبار URL
                if not is_valid_url(full_url):
                    continue

                # فیلتر کردن لینک‌های خارجی (اختیاری)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc != base_domain:
                    continue

                # فیلتر کردن URL‌های غیر HTTP
                if not full_url.startswith(('http://', 'https://')):
                    continue

                # فیلتر کردن URL‌های تکراری
                if full_url not in links:
                    links.append(full_url)

            return links

        except Exception as e:
            self.logger.error(f"خطا در استخراج لینک‌ها از {base_url}: {str(e)}")
            return []

    async def filter_news_links(self, links: List[str], website_config: Dict[str, Any]) -> List[str]:
        """
        فیلتر کردن لینک‌های خبر بر اساس الگوهای تعریف شده

        Args:
            links: لیست لینک‌ها
            website_config: تنظیمات وب‌سایت

        Returns:
            لیست لینک‌های فیلتر شده
        """
        # الگوهای شناسایی لینک‌های خبر
        news_patterns = website_config.get('news_patterns', [])
        if not news_patterns:
            return links

        # تبدیل الگوها به عبارات منظم
        compiled_patterns = [re.compile(pattern) for pattern in news_patterns]

        # فیلتر کردن لینک‌ها
        filtered_links = []
        for link in links:
            for pattern in compiled_patterns:
                if pattern.search(link):
                    filtered_links.append(link)
                    break

        return filtered_links

    async def parse_article_with_newspaper(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """
        پارس مقاله با استفاده از کتابخانه Newspaper

        Args:
            html: محتوای HTML
            url: آدرس مقاله

        Returns:
            اطلاعات مقاله یا None
        """
        try:
            # ایجاد شیء مقاله
            article = Article(url)

            # تنظیم HTML (به جای دانلود مجدد)
            article.download_state = 2  # حالت دانلود شده
            article.html = html

            # پارس مقاله
            article.parse()

            # تمیز کردن متن
            text = clean_text(article.text)

            # بررسی طول متن
            if len(text) < 100:  # رد مقالات خیلی کوتاه
                return None

            # استخراج تاریخ انتشار
            publish_date = article.publish_date

            # اطلاعات مقاله
            article_data = {
                "title": article.title,
                "text": text,
                "url": url,
                "authors": article.authors,
                "publish_date": publish_date,
                "top_image": article.top_image,
                "has_media": bool(article.top_image),
                "media_urls": [article.top_image] if article.top_image else None,
            }

            return article_data

        except ArticleException as e:
            self.logger.warning(f"خطای Newspaper برای {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"خطا در پارس مقاله {url}: {str(e)}")
            return None

    async def parse_article_with_custom_parser(self, html: str, url: str, website_config: Dict[str, Any]) -> Optional[
        Dict[str, Any]]:
        """
        پارس مقاله با استفاده از پارسر اختصاصی

        Args:
            html: محتوای HTML
            url: آدرس مقاله
            website_config: تنظیمات وب‌سایت

        Returns:
            اطلاعات مقاله یا None
        """
        try:
            # سلکتورهای CSS از تنظیمات
            selectors = website_config.get('selectors', {})
            if not selectors:
                return None

            # پارس HTML
            soup = BeautifulSoup(html, 'html.parser')

            # استخراج عنوان
            title = None
            if 'title' in selectors:
                title_elem = soup.select_one(selectors['title'])
                if title_elem:
                    title = title_elem.get_text(strip=True)

            # استخراج متن
            text = ""
            if 'content' in selectors:
                content_elem = soup.select_one(selectors['content'])
                if content_elem:
                    # حذف عناصر غیرضروری
                    for selector in selectors.get('remove', []):
                        for elem in content_elem.select(selector):
                            elem.decompose()

                    # استخراج متن
                    text = content_elem.get_text(separator=' ', strip=True)

            # بررسی داده‌های اساسی
            if not title or not text or len(text) < 100:
                return None

            # تمیز کردن متن
            text = clean_text(text)

            # استخراج تاریخ انتشار
            publish_date = None
            if 'date' in selectors:
                date_elem = soup.select_one(selectors['date'])
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    # پارس تاریخ با الگوهای سایت
                    date_formats = website_config.get('date_formats', [])
                    for fmt in date_formats:
                        try:
                            publish_date = datetime.strptime(date_text, fmt)
                            break
                        except ValueError:
                            continue

            # استخراج تصویر اصلی
            top_image = None
            if 'image' in selectors:
                img_elem = soup.select_one(selectors['image'])
                if img_elem:
                    if img_elem.name == 'img':
                        top_image = img_elem.get('src')
                    else:
                        img = img_elem.select_one('img')
                        if img:
                            top_image = img.get('src')

            # استخراج نویسندگان
            authors = []
            if 'author' in selectors:
                author_elem = soup.select_one(selectors['author'])
                if author_elem:
                    authors = [author_elem.get_text(strip=True)]

            # اطلاعات مقاله
            article_data = {
                "title": title,
                "text": text,
                "url": url,
                "authors": authors,
                "publish_date": publish_date,
                "top_image": top_image,
                "has_media": bool(top_image),
                "media_urls": [top_image] if top_image else None,
            }

            return article_data

        except Exception as e:
            self.logger.error(f"خطا در پارس اختصاصی مقاله {url}: {str(e)}")
            return None

    async def process_article(self, html: str, url: str, website_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        پردازش یک مقاله خبری

        Args:
            html: محتوای HTML
            url: آدرس مقاله
            website_config: تنظیمات وب‌سایت

        Returns:
            اطلاعات پردازش شده مقاله یا None
        """
        # بررسی URL تکراری
        if await self.is_processed_url(url):
            return None

        # علامت‌گذاری URL به عنوان پردازش شده
        await self.mark_url_processed(url)

        # انتخاب روش پارس مناسب
        use_custom_parser = website_config.get('use_custom_parser', False)

        if use_custom_parser and website_config.get('selectors'):
            # پارس با پارسر اختصاصی
            article_data = await self.parse_article_with_custom_parser(html, url, website_config)
        else:
            # پارس با Newspaper
            article_data = await self.parse_article_with_newspaper(html, url)

        # بررسی نتیجه پارس
        if not article_data:
            return None

        # محاسبه هش متن برای تشخیص تکرار
        text_hash = compute_similarity_hash(article_data["text"])

        # بررسی تکراری بودن متن
        if self.db.is_duplicate_news(article_data["text"]):
            return None

        # تکمیل اطلاعات مقاله
        source_name = website_config.get('name', urlparse(url).netloc)

        # اطلاعات نهایی
        article_info = {
            "external_id": hashlib.md5(url.encode()).hexdigest(),
            "source": NewsSource.WEBSITE.value,
            "source_name": source_name,
            "text": article_data["text"],
            "title": article_data["title"],
            "url": url,
            "has_media": article_data["has_media"],
            "media_urls": article_data["media_urls"],
            "published_at": article_data["publish_date"] or now(),
            "meta": {
                "authors": article_data["authors"],
                "top_image": article_data["top_image"],
                "text_hash": text_hash
            }
        }

        return article_info

    async def save_article(self, article_info: Dict[str, Any]) -> bool:
        """
        ذخیره اطلاعات یک مقاله در دیتابیس

        Args:
            article_info: اطلاعات مقاله

        Returns:
            وضعیت ذخیره‌سازی
        """
        try:
            # افزودن به دیتابیس
            raw_news = self.db.add_raw_news(
                external_id=article_info["external_id"],
                source=article_info["source"],
                source_name=article_info["source_name"],
                text=article_info["text"],
                title=article_info["title"],
                url=article_info["url"],
                has_media=article_info["has_media"],
                media_urls=article_info["media_urls"],
                published_at=article_info["published_at"]
            )

            if raw_news:
                return True
            return False

        except Exception as e:
            self.logger.error(f"خطا در ذخیره مقاله {article_info['url']}: {str(e)}")
            return False

    async def collect_from_website(self, website_config: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        جمع‌آوری اخبار از یک وب‌سایت

        Args:
            website_config: تنظیمات وب‌سایت

        Returns:
            تعداد URL‌های بررسی شده، تعداد مقالات پردازش شده، تعداد مقالات ذخیره شده
        """
        url_count = 0
        processed_count = 0
        saved_count = 0

        try:
            # دریافت اطلاعات وب‌سایت
            base_url = website_config['url']
            depth = website_config.get('depth', 1)
            max_articles = website_config.get('max_articles', 20)

            # نام وب‌سایت
            name = website_config.get('name', urlparse(base_url).netloc)
            self.logger.info(f"شروع جمع‌آوری از وب‌سایت {name} ({base_url})")

            # ایجاد جلسه HTTP
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # دریافت صفحه اصلی
                html = await self.fetch_url(session, base_url)
                if not html:
                    self.logger.error(f"خطا در دریافت صفحه اصلی {base_url}")
                    return 0, 0, 0

                # استخراج لینک‌ها
                all_links = await self.extract_links(html, base_url)
                url_count = len(all_links)

                # فیلتر کردن لینک‌های خبر
                news_links = await self.filter_news_links(all_links, website_config)
                self.logger.info(f"تعداد {len(news_links)} لینک خبر از {url_count} لینک یافت شد")

                # محدود کردن تعداد مقالات
                news_links = news_links[:max_articles]

                # پردازش مقالات
                for link in news_links:
                    try:
                        # دریافت محتوای مقاله
                        article_html = await self.fetch_url(session, link)
                        if not article_html:
                            continue

                        # پردازش مقاله
                        article_info = await self.process_article(article_html, link, website_config)

                        if article_info:
                            processed_count += 1

                            # ذخیره مقاله
                            if await self.save_article(article_info):
                                saved_count += 1

                        # تأخیر بین درخواست‌ها
                        await asyncio.sleep(random.uniform(1.0, 3.0))

                    except Exception as e:
                        self.logger.error(f"خطا در پردازش مقاله {link}: {str(e)}")
                        continue

            self.logger.info(
                f"جمع‌آوری از وب‌سایت {name} پایان یافت. URL‌ها: {url_count}, پردازش‌شده: {processed_count}, ذخیره‌شده: {saved_count}")
            return url_count, processed_count, saved_count

        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری از وب‌سایت {website_config.get('name', '')}: {str(e)}")
            return url_count, processed_count, saved_count

    async def collect_all(self, websites: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Tuple[int, int, int]]:
        """
        جمع‌آوری اخبار از تمام وب‌سایت‌های تنظیم شده

        Args:
            websites: لیست تنظیمات وب‌سایت‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری برای هر وب‌سایت
        """
        websites_to_collect = websites or self.websites
        results = {}

        for website_config in websites_to_collect:
            try:
                name = website_config.get('name', website_config['url'])
                url_count, processed_count, saved_count = await self.collect_from_website(website_config)
                results[name] = (url_count, processed_count, saved_count)

                # تأخیر بین وب‌سایت‌ها
                await asyncio.sleep(2)

            except Exception as e:
                self.logger.error(f"خطا در جمع‌آوری وب‌سایت {website_config.get('name', '')}: {str(e)}")
                results[website_config.get('name', website_config['url'])] = (0, 0, 0)

        return results

    async def run(self, websites: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Tuple[int, int, int]]:
        """
        اجرای جمع‌کننده وب‌سایت

        Args:
            websites: لیست تنظیمات وب‌سایت‌ها (اختیاری)

        Returns:
            دیکشنری نتایج جمع‌آوری
        """
        try:
            # پاکسازی URL‌های پردازش شده قدیمی
            await self.clean_expired_processed_urls()

            # جمع‌آوری از تمام وب‌سایت‌ها
            results = await self.collect_all(websites)

            return results

        except Exception as e:
            self.logger.error(f"خطا در اجرای جمع‌کننده وب‌سایت: {str(e)}")
            return {}
