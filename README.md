﻿# بات خبری ارزهای دیجیتال (CryptoNewsBot)

پروژه CryptoNewsBot یک سیستم هوشمند برای جمع‌آوری، تحلیل، پردازش و انتشار اخبار ارزهای دیجیتال است. این سیستم با استفاده از مدل‌های پیش‌آموزش دیده هوش مصنوعی (از جمله ParsBERT)، اخبار مرتبط با ارزهای دیجیتال را از منابع مختلف (کانال‌های تلگرام، وب‌سایت‌های خبری و توییتر) جمع‌آوری می‌کند و پس از تحلیل محتوا، اهمیت و احساسات متن، خبرهای مهم را شناسایی، پردازش و در قالبی استاندارد در کانال تلگرام منتشر می‌کند.

## پیش‌نیازها

- Python 3.8 یا بالاتر
- PostgreSQL 14 یا بالاتر
- Redis 6 یا بالاتر
- Docker و Docker Compose (برای اجرا با داکر)

## راه‌اندازی

### با استفاده از Docker

1. فایل .env.example را به .env تغییر نام دهید و مقادیر را تنظیم کنید
2. دستور زیر را اجرا کنید:

`ash
docker-compose up -d
`

### بدون Docker

1. فایل .env.example را به .env تغییر نام دهید و مقادیر را تنظیم کنید
2. وابستگی‌ها را نصب کنید:

`ash
python -m pip install -r requirements.txt
`

3. برنامه را اجرا کنید:

`ash
python main.py
`

## ساختار پروژه

- collector/: ماژول جمع‌آوری داده
- nalyzer/: ماژول تحلیل محتوا
- generator/: ماژول تولید محتوا
- publisher/: ماژول انتشار
- database/: ماژول دیتابیس
- cache/: ماژول کش
- 	raining/: ماژول آموزش و بهبود مدل‌ها
- models/: ماژول مدل‌های پیش‌آموزش‌دیده
- utils/: ابزارهای کمکی
- 	ests/: تست‌ها

## فازهای اجرایی

1. **فاز اول**: راه‌اندازی هسته اصلی با قابلیت‌های پایه
2. **فاز دوم**: توسعه قابلیت‌های پیشرفته و بهبود کیفیت خروجی
3. **فاز سوم**: بهینه‌سازی عملکرد، افزایش مقیاس‌پذیری و افزودن قابلیت‌های پیشرفته

## ابزارهای مدیریتی

- PostgreSQL Adminer: http://localhost:8080
- Redis Commander: http://localhost:8081
