# اسکریپت اصلی اجرای پروژه CryptoNewsBot
# تاریخ ایجاد: 2025-03-26

import asyncio
from utils.config import Config
from utils.logger import Logger
from collector.collection_manager import CollectionManager
from analyzer.analyzer_manager import AnalyzerManager
from generator.content_generator import ContentGenerator
from publisher.publisher_manager import PublisherManager

async def main():
    """تابع اصلی اجرای پروژه"""
    # راه‌اندازی لاگر
    logger = Logger()
    logger.info("شروع اجرای بات خبری ارزهای دیجیتال")
    
    # بارگذاری تنظیمات
    config = Config()
    
    # راه‌اندازی مدیریت‌کننده‌ها
    collection_manager = CollectionManager()
    analyzer_manager = AnalyzerManager()
    content_generator = ContentGenerator()
    publisher_manager = PublisherManager()
    
    # شروع فرآیند اصلی
    try:
        # اجرای چرخه اصلی برنامه
        while True:
            # جمع‌آوری اخبار جدید
            raw_news = await collection_manager.collect_news()
            
            # تحلیل اخبار
            analyzed_news = await analyzer_manager.analyze_news(raw_news)
            
            # تولید محتوا
            generated_content = await content_generator.generate_content(analyzed_news)
            
            # انتشار اخبار
            await publisher_manager.publish_news(generated_content)
            
            # انتظار برای دور بعدی
            await asyncio.sleep(config.collection_interval)
    
    except KeyboardInterrupt:
        logger.info("توقف اجرای برنامه توسط کاربر")
    except Exception as e:
        logger.error(f"خطا در اجرای برنامه: {str(e)}")
    finally:
        logger.info("پایان اجرای بات خبری ارزهای دیجیتال")

if __name__ == "__main__":
    asyncio.run(main())
