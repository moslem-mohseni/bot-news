"""
ماژول مدیریت کوئری برای CryptoNewsBot

این ماژول مسئول اجرای کوئری‌های پیچیده و مدیریت دسترسی به داده‌ها است.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import and_, or_, desc, func, text
from sqlalchemy.orm import Session, contains_eager

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.helpers import compute_similarity_hash, now
from .db_manager import DatabaseManager
from .models import (
    RawNews, NewsAnalysis, GeneratedContent, UserFeedback,
    TrainingData, SourceCredibility, NewsCategory, NewsSource,
    PublishStatus, SentimentType
)


class QueryManager:
    """
    کلاس مدیریت کوئری‌های پیچیده

    این کلاس مسئول اجرای کوئری‌های پیچیده و مدیریت دسترسی به داده‌هاست.
    از الگوی Singleton استفاده می‌کند تا در کل برنامه فقط یک نمونه از آن وجود داشته باشد.
    """

    _instance = None

    def __new__(cls):
        """
        الگوی Singleton برای جلوگیری از ایجاد چندین نمونه از کلاس

        Returns:
            نمونه منحصر به فرد از کلاس
        """
        if cls._instance is None:
            cls._instance = super(QueryManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        راه‌اندازی اولیه کلاس
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()
        self.db_manager = DatabaseManager()

    # === کوئری‌های مربوط به RawNews ===

    def add_raw_news(self,
                     external_id: str,
                     source: Union[NewsSource, str],
                     source_name: str,
                     text: str,
                     title: Optional[str] = None,
                     language_code: Optional[str] = None,
                     url: Optional[str] = None,
                     has_media: bool = False,
                     media_urls: Optional[List[str]] = None,
                     published_at: Optional[datetime] = None) -> Optional[RawNews]:
        """
        افزودن یک خبر خام جدید

        Args:
            external_id: شناسه خبر در منبع اصلی
            source: منبع خبر
            source_name: نام منبع
            text: متن کامل خبر
            title: عنوان خبر (اختیاری)
            language_code: کد زبان (اختیاری)
            url: یوآرال خبر (اختیاری)
            has_media: آیا خبر شامل رسانه است؟
            media_urls: آدرس‌های رسانه (اختیاری)
            published_at: زمان انتشار در منبع اصلی (اختیاری)

        Returns:
            خبر خام ایجاد شده یا None در صورت خطا
        """
        try:
            # تبدیل رشته به Enum در صورت نیاز
            if isinstance(source, str):
                source = NewsSource(source)

            # ایجاد هش متن برای تشخیص تکرار
            text_hash = compute_similarity_hash(text)

            # تبدیل لیست آدرس‌های رسانه به JSON
            media_urls_json = None
            if media_urls:
                media_urls_json = json.dumps(media_urls, ensure_ascii=False)

            # ایجاد خبر خام
            raw_news = RawNews(
                external_id=external_id,
                source=source,
                source_name=source_name,
                title=title,
                text=text,
                language_code=language_code,
                url=url,
                has_media=has_media,
                media_urls=media_urls_json,
                published_at=published_at,
                collected_at=now(),
                text_hash=text_hash
            )

            # افزودن به دیتابیس
            return self.db_manager.add(raw_news)
        except Exception as e:
            self.logger.error(f"خطا در افزودن خبر خام: {str(e)}")
            return None

    def is_duplicate_news(self, text: str, threshold: float = 0.9, hours: int = 24) -> bool:
        """
        بررسی تکراری بودن یک خبر بر اساس متن آن

        Args:
            text: متن خبر
            threshold: آستانه تشخیص تکرار (0 تا 1)
            hours: بازه زمانی بررسی (ساعت)

        Returns:
            True اگر خبر تکراری باشد
        """
        # محاسبه هش متن
        text_hash = compute_similarity_hash(text)

        # محاسبه زمان شروع بررسی
        start_time = now() - timedelta(hours=hours)

        with self.db_manager.session_scope() as session:
            # بررسی تکرار دقیق
            exact_duplicate = session.query(RawNews).filter(
                RawNews.text_hash == text_hash,
                RawNews.collected_at >= start_time
            ).first()

            if exact_duplicate:
                return True

            # در اینجا می‌توان منطق پیچیده‌تری برای تشخیص تکرار معنایی اضافه کرد
            # به عنوان مثال، استفاده از embedding و محاسبه فاصله کسینوسی

            return False

    def get_unanalyzed_news(self, limit: int = 10) -> List[RawNews]:
        """
        دریافت خبرهای تحلیل نشده

        Args:
            limit: محدودیت تعداد نتایج

        Returns:
            لیست خبرهای تحلیل نشده
        """
        with self.db_manager.session_scope() as session:
            # زیرکوئری برای خبرهایی که تحلیل شده‌اند
            analyzed_news = session.query(NewsAnalysis.raw_news_id)

            # کوئری اصلی برای خبرهای تحلیل نشده
            query = session.query(RawNews).filter(
                ~RawNews.id.in_(analyzed_news)
            ).order_by(desc(RawNews.collected_at)).limit(limit)

            return query.all()

    def get_news_by_source(self, source: Union[NewsSource, str], source_name: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> List[RawNews]:
        """
        دریافت خبرها بر اساس منبع

        Args:
            source: منبع خبر
            source_name: نام منبع (اختیاری)
            limit: محدودیت تعداد نتایج
            offset: تعداد نتایج برای پرش از ابتدا

        Returns:
            لیست خبرها
        """
        # تبدیل رشته به Enum در صورت نیاز
        if isinstance(source, str):
            source = NewsSource(source)

        with self.db_manager.session_scope() as session:
            query = session.query(RawNews).filter(RawNews.source == source)

            if source_name:
                query = query.filter(RawNews.source_name == source_name)

            query = query.order_by(desc(RawNews.collected_at)).offset(offset).limit(limit)

            return query.all()

    def search_news(self, query_text: str, limit: int = 100, offset: int = 0) -> List[RawNews]:
        """
        جستجوی خبرها بر اساس متن

        Args:
            query_text: متن جستجو
            limit: محدودیت تعداد نتایج
            offset: تعداد نتایج برای پرش از ابتدا

        Returns:
            لیست خبرهای منطبق
        """
        with self.db_manager.session_scope() as session:
            # جستجو در عنوان و متن
            query = session.query(RawNews).filter(
                or_(
                    RawNews.title.ilike(f"%{query_text}%"),
                    RawNews.text.ilike(f"%{query_text}%")
                )
            ).order_by(desc(RawNews.collected_at)).offset(offset).limit(limit)

            return query.all()

    # === کوئری‌های مربوط به NewsAnalysis ===

    def add_news_analysis(self,
                          raw_news_id: int,
                          primary_category: Union[NewsCategory, str, None] = None,
                          category_confidence: float = 0.0,
                          secondary_categories: Optional[Dict[str, float]] = None,
                          sentiment: Union[SentimentType, str, None] = None,
                          sentiment_score: float = 0.0,
                          importance_score: float = 0.0,
                          importance_factors: Optional[Dict[str, float]] = None,
                          analyzer_version: Optional[str] = None,
                          processing_time: Optional[float] = None) -> Optional[NewsAnalysis]:
        """
        افزودن تحلیل برای یک خبر

        Args:
            raw_news_id: شناسه خبر خام
            primary_category: دسته‌بندی اصلی
            category_confidence: میزان اطمینان دسته‌بندی
            secondary_categories: دسته‌بندی‌های ثانویه
            sentiment: احساس متن
            sentiment_score: امتیاز احساس
            importance_score: امتیاز اهمیت خبر
            importance_factors: فاکتورهای مؤثر در اهمیت
            analyzer_version: نسخه آنالایزر
            processing_time: زمان پردازش

        Returns:
            تحلیل خبر ایجاد شده یا None در صورت خطا
        """
        try:
            # تبدیل رشته به Enum در صورت نیاز
            if isinstance(primary_category, str):
                primary_category = NewsCategory(primary_category)

            if isinstance(sentiment, str):
                sentiment = SentimentType(sentiment)

            # تبدیل دیکشنری‌ها به JSON
            secondary_categories_json = None
            if secondary_categories:
                secondary_categories_json = json.dumps(secondary_categories, ensure_ascii=False)

            importance_factors_json = None
            if importance_factors:
                importance_factors_json = json.dumps(importance_factors, ensure_ascii=False)

            # ایجاد تحلیل خبر
            news_analysis = NewsAnalysis(
                raw_news_id=raw_news_id,
                primary_category=primary_category,
                category_confidence=category_confidence,
                secondary_categories=secondary_categories_json,
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                importance_score=importance_score,
                importance_factors=importance_factors_json,
                analyzer_version=analyzer_version,
                processing_time=processing_time,
                analyzed_at=now()
            )

            # افزودن به دیتابیس
            return self.db_manager.add(news_analysis)
        except Exception as e:
            self.logger.error(f"خطا در افزودن تحلیل خبر: {str(e)}")
            return None

    def get_news_with_analysis(self, limit: int = 100, offset: int = 0) -> List[Tuple[RawNews, NewsAnalysis]]:
        """
        دریافت خبرها به همراه تحلیل آن‌ها

        Args:
            limit: محدودیت تعداد نتایج
            offset: تعداد نتایج برای پرش از ابتدا

        Returns:
            لیست تاپل‌های (خبر، تحلیل)
        """
        with self.db_manager.session_scope() as session:
            query = session.query(RawNews, NewsAnalysis).join(
                NewsAnalysis, RawNews.id == NewsAnalysis.raw_news_id
            ).order_by(desc(RawNews.collected_at)).offset(offset).limit(limit)

            return query.all()

    def get_important_news(self, min_score: float = 0.7, limit: int = 10) -> List[Tuple[RawNews, NewsAnalysis]]:
        """
        دریافت خبرهای مهم بر اساس امتیاز اهمیت

        Args:
            min_score: حداقل امتیاز اهمیت
            limit: محدودیت تعداد نتایج

        Returns:
            لیست تاپل‌های (خبر، تحلیل)
        """
        with self.db_manager.session_scope() as session:
            query = session.query(RawNews, NewsAnalysis).join(
                NewsAnalysis, RawNews.id == NewsAnalysis.raw_news_id
            ).filter(
                NewsAnalysis.importance_score >= min_score
            ).order_by(desc(NewsAnalysis.importance_score), desc(RawNews.collected_at)).limit(limit)

            return query.all()

    def get_news_by_category(self, category: Union[NewsCategory, str], limit: int = 100, offset: int = 0) -> List[
        Tuple[RawNews, NewsAnalysis]]:
        """
        دریافت خبرها بر اساس دسته‌بندی

        Args:
            category: دسته‌بندی
            limit: محدودیت تعداد نتایج
            offset: تعداد نتایج برای پرش از ابتدا

        Returns:
            لیست تاپل‌های (خبر، تحلیل)
        """
        # تبدیل رشته به Enum در صورت نیاز
        if isinstance(category, str):
            category = NewsCategory(category)

        with self.db_manager.session_scope() as session:
            query = session.query(RawNews, NewsAnalysis).join(
                NewsAnalysis, RawNews.id == NewsAnalysis.raw_news_id
            ).filter(
                NewsAnalysis.primary_category == category
            ).order_by(desc(RawNews.collected_at)).offset(offset).limit(limit)

            return query.all()

    # === کوئری‌های مربوط به GeneratedContent ===

    def add_generated_content(self,
                              raw_news_id: int,
                              title: str,
                              summary: str,
                              full_text: str,
                              has_media: bool = False,
                              media_urls: Optional[List[str]] = None,
                              generator_version: Optional[str] = None,
                              publish_status: PublishStatus = PublishStatus.PENDING,
                              scheduled_for: Optional[datetime] = None) -> Optional[GeneratedContent]:
        """
        افزودن محتوای تولید شده برای یک خبر

        Args:
            raw_news_id: شناسه خبر خام
            title: عنوان تولید شده
            summary: خلاصه خبر
            full_text: متن کامل آماده انتشار
            has_media: آیا محتوا شامل رسانه است؟
            media_urls: آدرس‌های رسانه
            generator_version: نسخه تولیدکننده
            publish_status: وضعیت انتشار
            scheduled_for: زمان زمان‌بندی شده برای انتشار

        Returns:
            محتوای تولید شده یا None در صورت خطا
        """
        try:
            # تبدیل لیست آدرس‌های رسانه به JSON
            media_urls_json = None
            if media_urls:
                media_urls_json = json.dumps(media_urls, ensure_ascii=False)

            # ایجاد محتوای تولید شده
            generated_content = GeneratedContent(
                raw_news_id=raw_news_id,
                title=title,
                summary=summary,
                full_text=full_text,
                has_media=has_media,
                media_urls=media_urls_json,
                generator_version=generator_version,
                publish_status=publish_status,
                scheduled_for=scheduled_for,
                generated_at=now()
            )

            # افزودن به دیتابیس
            return self.db_manager.add(generated_content)
        except Exception as e:
            self.logger.error(f"خطا در افزودن محتوای تولید شده: {str(e)}")
            return None

    def get_content_for_publication(self, limit: int = 10) -> List[GeneratedContent]:
        """
        دریافت محتواهای آماده انتشار

        Args:
            limit: محدودیت تعداد نتایج

        Returns:
            لیست محتواهای آماده انتشار
        """
        current_time = now()

        with self.db_manager.session_scope() as session:
            # محتواهای در انتظار
            pending_query = session.query(GeneratedContent).filter(
                GeneratedContent.publish_status == PublishStatus.PENDING
            )

            # محتواهای زمان‌بندی شده که زمان انتشار آن‌ها رسیده است
            scheduled_query = session.query(GeneratedContent).filter(
                GeneratedContent.publish_status == PublishStatus.SCHEDULED,
                GeneratedContent.scheduled_for <= current_time
            )

            # ترکیب کوئری‌ها و مرتب‌سازی بر اساس زمان
            combined_query = pending_query.union(scheduled_query).order_by(
                GeneratedContent.scheduled_for,  # ابتدا زمان‌بندی شده‌ها
                desc(GeneratedContent.generated_at)  # سپس بر اساس زمان تولید
            ).limit(limit)

            return combined_query.all()

    def update_content_status(self, content_id: int, status: PublishStatus,
                              message_id: Optional[str] = None) -> bool:
        """
        به‌روزرسانی وضعیت انتشار یک محتوا

        Args:
            content_id: شناسه محتوا
            status: وضعیت جدید
            message_id: شناسه پیام منتشر شده (برای وضعیت PUBLISHED)

        Returns:
            وضعیت موفقیت عملیات
        """
        try:
            with self.db_manager.session_scope() as session:
                content = session.query(GeneratedContent).filter(
                    GeneratedContent.id == content_id
                ).first()

                if not content:
                    return False

                content.publish_status = status

                if status == PublishStatus.PUBLISHED:
                    content.published_at = now()
                    if message_id:
                        content.publish_message_id = message_id

                return True
        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی وضعیت محتوا: {str(e)}")
            return False

    # === کوئری‌های مربوط به UserFeedback ===

    def add_user_feedback(self,
                          generated_content_id: int,
                          feedback_type: str,
                          feedback_value: int = 0,
                          user_id: Optional[str] = None,
                          user_name: Optional[str] = None,
                          feedback_text: Optional[str] = None) -> Optional[UserFeedback]:
        """
        افزودن بازخورد کاربر برای یک محتوا

        Args:
            generated_content_id: شناسه محتوای تولید شده
            feedback_type: نوع بازخورد
            feedback_value: مقدار بازخورد
            user_id: شناسه کاربر
            user_name: نام کاربر
            feedback_text: متن بازخورد

        Returns:
            بازخورد ایجاد شده یا None در صورت خطا
        """
        try:
            # ایجاد بازخورد
            user_feedback = UserFeedback(
                generated_content_id=generated_content_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                user_id=user_id,
                user_name=user_name,
                feedback_text=feedback_text,
                created_at=now()
            )

            # افزودن به دیتابیس
            return self.db_manager.add(user_feedback)
        except Exception as e:
            self.logger.error(f"خطا در افزودن بازخورد کاربر: {str(e)}")
            return None

    def get_content_feedback(self, content_id: int) -> List[UserFeedback]:
        """
        دریافت بازخوردهای یک محتوا

        Args:
            content_id: شناسه محتوا

        Returns:
            لیست بازخوردها
        """
        with self.db_manager.session_scope() as session:
            query = session.query(UserFeedback).filter(
                UserFeedback.generated_content_id == content_id
            ).order_by(desc(UserFeedback.created_at))

            return query.all()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار بازخوردها

        Returns:
            دیکشنری آمار بازخوردها
        """
        with self.db_manager.session_scope() as session:
            # تعداد کل بازخوردها
            total_count = session.query(func.count(UserFeedback.id)).scalar()

            # تعداد بازخوردها بر اساس نوع
            type_counts = session.query(
                UserFeedback.feedback_type,
                func.count(UserFeedback.id)
            ).group_by(UserFeedback.feedback_type).all()

            # میانگین مقدار بازخوردها
            avg_value = session.query(func.avg(UserFeedback.feedback_value)).scalar()

            return {
                'total_count': total_count,
                'type_counts': dict(type_counts),
                'avg_value': float(avg_value) if avg_value else 0.0
            }

    # === کوئری‌های مربوط به TrainingData ===

    def add_training_data(self,
                          data_type: str,
                          target_field: str,
                          input_text: str,
                          expected_output: str,
                          source_type: Optional[str] = None,
                          source_id: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[TrainingData]:
        """
        افزودن داده آموزشی

        Args:
            data_type: نوع داده
            target_field: فیلد هدف
            input_text: متن ورودی
            expected_output: خروجی مورد انتظار
            source_type: نوع منبع
            source_id: شناسه منبع
            metadata: متادیتای اضافی

        Returns:
            داده آموزشی ایجاد شده یا None در صورت خطا
        """
        try:
            # تبدیل متادیتا به JSON
            metadata_json = None
            if metadata:
                metadata_json = json.dumps(metadata, ensure_ascii=False)

            # ایجاد داده آموزشی
            training_data = TrainingData(
                data_type=data_type,
                target_field=target_field,
                input_text=input_text,
                expected_output=expected_output,
                source_type=source_type,
                source_id=source_id,
                metadata=metadata_json,
                created_at=now(),
                used_for_training=False
            )

            # افزودن به دیتابیس
            return self.db_manager.add(training_data)
        except Exception as e:
            self.logger.error(f"خطا در افزودن داده آموزشی: {str(e)}")
            return None

    def get_training_data(self, data_type: Optional[str] = None,
                          used: Optional[bool] = None, limit: int = 1000) -> List[TrainingData]:
        """
        دریافت داده‌های آموزشی

        Args:
            data_type: نوع داده (اختیاری)
            used: آیا برای آموزش استفاده شده است؟ (اختیاری)
            limit: محدودیت تعداد نتایج

        Returns:
            لیست داده‌های آموزشی
        """
        with self.db_manager.session_scope() as session:
            query = session.query(TrainingData)

            if data_type:
                query = query.filter(TrainingData.data_type == data_type)

            if used is not None:
                query = query.filter(TrainingData.used_for_training == used)

            query = query.order_by(TrainingData.created_at).limit(limit)

            return query.all()

    def mark_training_data_used(self, data_ids: List[int]) -> bool:
        """
        علامت‌گذاری داده‌های آموزشی به عنوان استفاده شده

        Args:
            data_ids: لیست شناسه‌های داده‌ها

        Returns:
            وضعیت موفقیت عملیات
        """
        try:
            with self.db_manager.session_scope() as session:
                session.query(TrainingData).filter(
                    TrainingData.id.in_(data_ids)
                ).update({
                    'used_for_training': True,
                    'training_date': now()
                }, synchronize_session=False)

                return True
        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی وضعیت داده‌های آموزشی: {str(e)}")
            return False

    # === کوئری‌های مربوط به SourceCredibility ===

    def get_source_credibility(self, source: Union[NewsSource, str],
                               source_name: str) -> Optional[SourceCredibility]:
        """
        دریافت اعتبار یک منبع

        Args:
            source: نوع منبع
            source_name: نام منبع

        Returns:
            اعتبار منبع یا None در صورت عدم وجود
        """
        # تبدیل رشته به Enum در صورت نیاز
        if isinstance(source, str):
            source = NewsSource(source)

        with self.db_manager.session_scope() as session:
            query = session.query(SourceCredibility).filter(
                SourceCredibility.source == source,
                SourceCredibility.source_name == source_name
            )

            return query.first()

    def update_source_credibility(self, source: Union[NewsSource, str],
                                  source_name: str,
                                  credibility_score: float,
                                  reliability_score: float = None,
                                  importance_factor: float = None,
                                  description: str = None) -> Optional[SourceCredibility]:
        """
        به‌روزرسانی یا ایجاد اعتبار منبع

        Args:
            source: نوع منبع
            source_name: نام منبع
            credibility_score: امتیاز اعتبار
            reliability_score: امتیاز قابلیت اطمینان (اختیاری)
            importance_factor: ضریب اهمیت (اختیاری)
            description: توضیحات (اختیاری)

        Returns:
            اعتبار منبع به‌روزرسانی شده یا ایجاد شده
        """
        try:
            # تبدیل رشته به Enum در صورت نیاز
            if isinstance(source, str):
                source = NewsSource(source)

            with self.db_manager.session_scope() as session:
                # جستجوی اعتبار موجود
                credibility = session.query(SourceCredibility).filter(
                    SourceCredibility.source == source,
                    SourceCredibility.source_name == source_name
                ).first()

                if credibility:
                    # به‌روزرسانی
                    credibility.credibility_score = credibility_score

                    if reliability_score is not None:
                        credibility.reliability_score = reliability_score

                    if importance_factor is not None:
                        credibility.importance_factor = importance_factor

                    if description is not None:
                        credibility.description = description

                    credibility.last_updated = now()
                else:
                    # ایجاد جدید
                    credibility = SourceCredibility(
                        source=source,
                        source_name=source_name,
                        credibility_score=credibility_score,
                        reliability_score=reliability_score or 0.5,
                        importance_factor=importance_factor or 1.0,
                        description=description,
                        last_updated=now()
                    )
                    session.add(credibility)

                session.flush()
                session.refresh(credibility)
                return credibility
        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی اعتبار منبع: {str(e)}")
            return None

    # === کوئری‌های ترکیبی و گزارش‌گیری ===

    def get_news_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        دریافت آمار خبرها

        Args:
            days: تعداد روزهای گذشته برای محاسبه آمار

        Returns:
            دیکشنری آمار خبرها
        """
        start_time = now() - timedelta(days=days)

        with self.db_manager.session_scope() as session:
            # تعداد کل خبرها
            total_count = session.query(func.count(RawNews.id)).scalar()

            # تعداد خبرهای بازه زمانی
            period_count = session.query(func.count(RawNews.id)).filter(
                RawNews.collected_at >= start_time
            ).scalar()

            # تعداد خبرها بر اساس منبع
            source_counts = session.query(
                RawNews.source,
                func.count(RawNews.id)
            ).group_by(RawNews.source).all()

            # تعداد خبرها بر اساس دسته‌بندی
            category_counts = session.query(
                NewsAnalysis.primary_category,
                func.count(NewsAnalysis.id)
            ).group_by(NewsAnalysis.primary_category).all()

            # میانگین امتیاز اهمیت
            avg_importance = session.query(func.avg(NewsAnalysis.importance_score)).scalar()

            # تعداد خبرهای منتشر شده
            published_count = session.query(func.count(GeneratedContent.id)).filter(
                GeneratedContent.publish_status == PublishStatus.PUBLISHED
            ).scalar()

            return {
                'total_count': total_count,
                'period_count': period_count,
                'source_counts': {s.value: c for s, c in source_counts},
                'category_counts': {c.value if c else 'unknown': count for c, count in category_counts},
                'avg_importance': float(avg_importance) if avg_importance else 0.0,
                'published_count': published_count
            }

    def get_daily_news_counts(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        دریافت تعداد خبرهای روزانه

        Args:
            days: تعداد روزهای گذشته

        Returns:
            لیست دیکشنری‌های آمار روزانه
        """
        start_time = now() - timedelta(days=days)

        # تبدیل تاریخ جمع‌آوری به تاریخ روز در SQL
        date_trunc_sql = "DATE_TRUNC('day', collected_at)"

        with self.db_manager.session_scope() as session:
            # تعداد خبرهای روزانه
            daily_counts = session.query(
                text(date_trunc_sql).label('day'),
                func.count(RawNews.id).label('count')
            ).filter(
                RawNews.collected_at >= start_time
            ).group_by(text('day')).order_by(text('day')).all()

            # تبدیل به لیست دیکشنری‌ها
            result = []
            for day, count in daily_counts:
                result.append({
                    'day': day.strftime('%Y-%m-%d'),
                    'count': count
                })

            return result

    def get_content_generation_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار تولید محتوا

        Returns:
            دیکشنری آمار تولید محتوا
        """
        with self.db_manager.session_scope() as session:
            # تعداد کل محتواها
            total_count = session.query(func.count(GeneratedContent.id)).scalar()

            # تعداد محتواها بر اساس وضعیت
            status_counts = session.query(
                GeneratedContent.publish_status,
                func.count(GeneratedContent.id)
            ).group_by(GeneratedContent.publish_status).all()

            # میانگین زمان بین تولید و انتشار
            avg_time_to_publish = session.query(
                func.avg(func.extract('epoch', GeneratedContent.published_at - GeneratedContent.generated_at))
            ).filter(
                GeneratedContent.publish_status == PublishStatus.PUBLISHED,
                GeneratedContent.published_at.isnot(None)
            ).scalar()

            return {
                'total_count': total_count,
                'status_counts': {s.value: c for s, c in status_counts},
                'avg_time_to_publish': float(avg_time_to_publish) if avg_time_to_publish else 0.0
            }
