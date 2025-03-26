"""
ماژول مدل‌های داده برای CryptoNewsBot

این ماژول شامل تعریف مدل‌های داده‌ای است که در دیتابیس ذخیره می‌شوند.
از SQLAlchemy ORM برای تعریف مدل‌ها استفاده شده است.
"""

import enum
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Enum, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# تعریف کلاس پایه برای مدل‌ها
Base = declarative_base()


class NewsSource(enum.Enum):
    """منابع خبری پشتیبانی شده"""
    TELEGRAM = "telegram"
    WEBSITE = "website"
    TWITTER = "twitter"


class NewsCategory(enum.Enum):
    """دسته‌بندی‌های خبری پشتیبانی شده"""
    PRICE = "price"  # قیمت
    REGULATION = "regulation"  # قوانین
    ADOPTION = "adoption"  # پذیرش
    TECHNOLOGY = "technology"  # تکنولوژی
    SECURITY = "security"  # امنیت
    EXCHANGE = "exchange"  # صرافی
    GENERAL = "general"  # عمومی
    OTHER = "other"  # سایر


class SentimentType(enum.Enum):
    """انواع احساسات متن"""
    POSITIVE = "positive"  # مثبت
    NEGATIVE = "negative"  # منفی
    NEUTRAL = "neutral"  # خنثی


class PublishStatus(enum.Enum):
    """وضعیت‌های انتشار خبر"""
    PENDING = "pending"  # در انتظار
    SCHEDULED = "scheduled"  # زمان‌بندی شده
    PUBLISHED = "published"  # منتشر شده
    REJECTED = "rejected"  # رد شده
    FAILED = "failed"  # خطا در انتشار


class RawNews(Base):
    """مدل خبر خام دریافتی از منابع مختلف"""
    __tablename__ = 'raw_news'

    id = Column(Integer, primary_key=True)
    external_id = Column(String(100), index=True)  # شناسه خبر در منبع اصلی
    source = Column(Enum(NewsSource), index=True)  # منبع خبر
    source_name = Column(String(100), index=True)  # نام منبع (مثلاً نام کانال تلگرام)

    title = Column(String(500), nullable=True)  # عنوان خبر (اگر موجود باشد)
    text = Column(Text)  # متن کامل خبر
    language_code = Column(String(10), nullable=True)  # کد زبان (fa, en, etc.)

    url = Column(String(500), nullable=True)  # یوآرال خبر (اگر موجود باشد)
    has_media = Column(Boolean, default=False)  # آیا خبر شامل رسانه است؟
    media_urls = Column(Text, nullable=True)  # آدرس‌های رسانه (JSON)

    published_at = Column(DateTime, nullable=True)  # زمان انتشار در منبع اصلی
    collected_at = Column(DateTime, default=datetime.utcnow)  # زمان جمع‌آوری

    # فیلد برای تشخیص اخبار تکراری
    text_hash = Column(String(64), index=True)  # هش متن برای تشخیص تکرار

    # ارتباط با تحلیل‌ها
    analyses = relationship("NewsAnalysis", back_populates="raw_news", cascade="all, delete-orphan")

    # ارتباط با محتوای تولیدی
    generated_contents = relationship("GeneratedContent", back_populates="raw_news", cascade="all, delete-orphan")

    # محدودیت یکتایی برای ترکیب منبع و شناسه خارجی
    __table_args__ = (
        UniqueConstraint('external_id', 'source', 'source_name', name='uix_raw_news_source_external_id'),
    )

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        title = self.title if self.title else self.text[:50] + '...'
        return f"<RawNews id={self.id} source={self.source.value}:{self.source_name} title='{title}'>"


class NewsAnalysis(Base):
    """مدل تحلیل محتوای خبر"""
    __tablename__ = 'news_analysis'

    id = Column(Integer, primary_key=True)
    raw_news_id = Column(Integer, ForeignKey('raw_news.id', ondelete='CASCADE'), nullable=False)

    # نتایج تحلیل موضوعی
    primary_category = Column(Enum(NewsCategory), nullable=True)  # دسته‌بندی اصلی
    category_confidence = Column(Float, default=0.0)  # میزان اطمینان دسته‌بندی
    secondary_categories = Column(String(200), nullable=True)  # دسته‌بندی‌های ثانویه (JSON)

    # تحلیل احساسات
    sentiment = Column(Enum(SentimentType), nullable=True)  # احساس متن
    sentiment_score = Column(Float, default=0.0)  # امتیاز احساس (-1 تا 1)

    # تحلیل اهمیت
    importance_score = Column(Float, default=0.0)  # امتیاز اهمیت خبر
    importance_factors = Column(Text, nullable=True)  # فاکتورهای مؤثر در اهمیت (JSON)

    # اطلاعات پردازش
    analyzed_at = Column(DateTime, default=datetime.utcnow)  # زمان تحلیل
    analyzer_version = Column(String(50), nullable=True)  # نسخه آنالایزر
    processing_time = Column(Float, nullable=True)  # زمان پردازش (ثانیه)

    # ارتباط با خبر خام
    raw_news = relationship("RawNews", back_populates="analyses")

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        return f"<NewsAnalysis id={self.id} news_id={self.raw_news_id} importance={self.importance_score:.2f}>"


class GeneratedContent(Base):
    """مدل محتوای تولید شده برای انتشار"""
    __tablename__ = 'generated_content'

    id = Column(Integer, primary_key=True)
    raw_news_id = Column(Integer, ForeignKey('raw_news.id', ondelete='CASCADE'), nullable=False)

    # محتوای تولید شده
    title = Column(String(500), nullable=False)  # عنوان تولید شده
    summary = Column(Text, nullable=False)  # خلاصه خبر
    full_text = Column(Text, nullable=False)  # متن کامل آماده انتشار

    # اطلاعات رسانه
    has_media = Column(Boolean, default=False)  # آیا محتوا شامل رسانه است؟
    media_urls = Column(Text, nullable=True)  # آدرس‌های رسانه (JSON)

    # اطلاعات تولید
    generated_at = Column(DateTime, default=datetime.utcnow)  # زمان تولید
    generator_version = Column(String(50), nullable=True)  # نسخه تولیدکننده

    # اطلاعات انتشار
    publish_status = Column(Enum(PublishStatus), default=PublishStatus.PENDING)  # وضعیت انتشار
    scheduled_for = Column(DateTime, nullable=True)  # زمان زمان‌بندی شده برای انتشار
    published_at = Column(DateTime, nullable=True)  # زمان انتشار
    publish_message_id = Column(String(100), nullable=True)  # شناسه پیام منتشر شده

    # ارتباط با خبر خام
    raw_news = relationship("RawNews", back_populates="generated_contents")

    # ارتباط با بازخوردها
    feedbacks = relationship("UserFeedback", back_populates="generated_content", cascade="all, delete-orphan")

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        return f"<GeneratedContent id={self.id} news_id={self.raw_news_id} status={self.publish_status.value}>"


class UserFeedback(Base):
    """مدل بازخورد کاربران برای خبرهای منتشر شده"""
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True)
    generated_content_id = Column(Integer, ForeignKey('generated_content.id', ondelete='CASCADE'), nullable=False)

    # اطلاعات کاربر
    user_id = Column(String(100), nullable=True)  # شناسه کاربر (اگر موجود باشد)
    user_name = Column(String(100), nullable=True)  # نام کاربر (اگر موجود باشد)

    # اطلاعات بازخورد
    feedback_type = Column(String(50), nullable=False)  # نوع بازخورد (like, dislike, report, etc.)
    feedback_value = Column(Integer, default=0)  # مقدار بازخورد (مثبت یا منفی)
    feedback_text = Column(Text, nullable=True)  # متن بازخورد (اگر موجود باشد)

    # زمان
    created_at = Column(DateTime, default=datetime.utcnow)  # زمان ایجاد بازخورد

    # ارتباط با محتوای تولید شده
    generated_content = relationship("GeneratedContent", back_populates="feedbacks")

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        return f"<UserFeedback id={self.id} content_id={self.generated_content_id} type={self.feedback_type}>"


class TrainingData(Base):
    """مدل داده‌های آموزشی جمع‌آوری شده"""
    __tablename__ = 'training_data'

    id = Column(Integer, primary_key=True)

    # نوع داده آموزشی
    data_type = Column(String(50), nullable=False)  # نوع داده (importance, summary, title, sentiment, etc.)
    target_field = Column(String(50), nullable=False)  # فیلد هدف برای آموزش

    # داده‌ها
    input_text = Column(Text, nullable=False)  # متن ورودی
    expected_output = Column(Text, nullable=False)  # خروجی مورد انتظار

    # منبع داده
    source_type = Column(String(50), nullable=True)  # نوع منبع (feedback, manual, etc.)
    source_id = Column(Integer, nullable=True)  # شناسه منبع (مثلاً شناسه بازخورد)

    # اطلاعات دیگر
    metadata = Column(Text, nullable=True)  # متادیتای اضافی (JSON)
    created_at = Column(DateTime, default=datetime.utcnow)  # زمان ایجاد
    used_for_training = Column(Boolean, default=False)  # آیا برای آموزش استفاده شده است؟
    training_date = Column(DateTime, nullable=True)  # زمان استفاده برای آموزش

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        return f"<TrainingData id={self.id} type={self.data_type} field={self.target_field}>"


class SourceCredibility(Base):
    """مدل اعتبار منابع خبری"""
    __tablename__ = 'source_credibility'

    id = Column(Integer, primary_key=True)

    # اطلاعات منبع
    source = Column(Enum(NewsSource), nullable=False)  # نوع منبع
    source_name = Column(String(100), nullable=False)  # نام منبع

    # اطلاعات اعتبار
    credibility_score = Column(Float, default=0.5)  # امتیاز اعتبار (0 تا 1)
    reliability_score = Column(Float, default=0.5)  # امتیاز قابلیت اطمینان (0 تا 1)
    importance_factor = Column(Float, default=1.0)  # ضریب اهمیت (برای محاسبه اهمیت خبر)

    # اطلاعات دیگر
    description = Column(Text, nullable=True)  # توضیحات
    last_updated = Column(DateTime, default=datetime.utcnow)  # آخرین به‌روزرسانی

    # محدودیت یکتایی برای ترکیب نوع و نام منبع
    __table_args__ = (
        UniqueConstraint('source', 'source_name', name='uix_source_credibility_source_name'),
    )

    def __repr__(self):
        """نمایش رشته‌ای مدل"""
        return f"<SourceCredibility id={self.id} source={self.source.value}:{self.source_name} score={self.credibility_score:.2f}>"
