# مستندات پروژه بات خبری ارزهای دیجیتال (CryptoNewsBot)

## معرفی پروژه

پروژه CryptoNewsBot یک سیستم هوشمند برای جمع‌آوری، تحلیل، پردازش و انتشار اخبار ارزهای دیجیتال است. این سیستم با استفاده از مدل‌های پیش‌آموزش دیده هوش مصنوعی (از جمله ParsBERT)، اخبار مرتبط با ارزهای دیجیتال را از منابع مختلف (کانال‌های تلگرام، وب‌سایت‌های خبری و توییتر) جمع‌آوری می‌کند و پس از تحلیل محتوا، اهمیت و احساسات متن، خبرهای مهم را شناسایی، پردازش و در قالبی استاندارد در کانال تلگرام منتشر می‌کند.

### اهداف اصلی پروژه

1. **جمع‌آوری خودکار اخبار**: جمع‌آوری مستمر و خودکار اخبار از منابع معتبر فارسی و انگلیسی
2. **تحلیل هوشمند محتوا**: تشخیص موضوع، اهمیت و احساسات متن
3. **فیلترینگ و اولویت‌بندی**: تشخیص اخبار مهم و فیلتر کردن اخبار کم‌اهمیت یا تکراری
4. **ترجمه و بومی‌سازی**: ترجمه اخبار انگلیسی به فارسی روان و قابل فهم
5. **خلاصه‌سازی هوشمند**: خلاصه کردن اخبار بلند با حفظ نکات کلیدی
6. **تولید محتوای جذاب**: ایجاد عناوین و متن‌های جذاب برای جلب توجه مخاطب
7. **انتشار استاندارد**: انتشار اخبار با قالب استاندارد و یکپارچه در کانال تلگرام
8. **بهبود مستمر**: بهبود عملکرد مدل‌ها با استفاده از بازخوردهای کاربران و مدیران

### رویکرد اصلی

پروژه با دو اصل اساسی پیاده‌سازی می‌شود:
1. **کمترین پیچیدگی**: طراحی ساده و قابل فهم برای توسعه و نگهداری آسان
2. **بیشترین کارایی عملیاتی**: استفاده بهینه از منابع و عملکرد سریع

## ساختار پروژه

ساختار کلی پوشه‌های پروژه:

```
crypto_news_bot/
│
├── collector/                  # ماژول جمع‌آوری داده
│
├── analyzer/                   # ماژول تحلیل محتوا
│
├── generator/                  # ماژول تولید محتوا
│
├── publisher/                  # ماژول انتشار
│
├── database/                   # ماژول دیتابیس
│
├── cache/                      # ماژول کش
│
├── training/                   # ماژول آموزش و بهبود مدل‌ها
│
├── models/                     # ماژول مدل‌های پیش‌آموزش‌دیده
│
├── utils/                      # ابزارهای کمکی
│
├── tests/                      # تست‌ها
```

## ماژول‌های پیاده‌سازی شده

### 1. ماژول ابزارها (utils)

ماژول `utils` شامل ابزارها و توابع کمکی است که در سراسر پروژه استفاده می‌شوند.

#### مسیر
```
crypto_news_bot/utils/
```

#### ساختار فایل‌بندی
```
utils/
├── __init__.py                # تعریف پکیج و اکسپورت‌های اصلی
├── config.py                  # مدیریت تنظیمات پروژه
├── logger.py                  # مدیریت لاگینگ
└── helpers.py                 # توابع کمکی متنوع
```

#### فایل‌ها و کلاس‌ها

##### `config.py`

**کلاس `Config`**

این کلاس مسئول مدیریت تنظیمات پروژه است و با استفاده از الگوی Singleton پیاده‌سازی شده است.

| متد | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `__new__(cls)` | کلاس | نمونه منحصر به فرد کلاس | الگوی Singleton |
| `_load_configs()` | - | - | بارگذاری تنظیمات از متغیرهای محیطی |
| `_get_env(key, default=None)` | کلید، مقدار پیش‌فرض | مقدار متغیر محیطی | خواندن متغیر محیطی با مقدار پیش‌فرض |
| `_get_env_int(key, default=0)` | کلید، مقدار پیش‌فرض | عدد صحیح | خواندن متغیر محیطی عددی صحیح |
| `_get_env_float(key, default=0.0)` | کلید، مقدار پیش‌فرض | عدد اعشاری | خواندن متغیر محیطی عددی اعشاری |
| `_get_env_bool(key, default=False)` | کلید، مقدار پیش‌فرض | بولین | خواندن متغیر محیطی بولین |
| `_get_env_list(key, default=None)` | کلید، مقدار پیش‌فرض | لیست | خواندن متغیر محیطی به صورت لیست با جداکننده کاما |
| `_get_env_json(key, default=None)` | کلید، مقدار پیش‌فرض | شیء JSON | خواندن متغیر محیطی به صورت JSON |
| `get_db_url()` | - | رشته URL | ساخت URL اتصال به دیتابیس |
| `set(key, value)` | کلید، مقدار | - | تنظیم یک مقدار پیکربندی در زمان اجرا |
| `to_dict()` | - | دیکشنری | تبدیل تمام تنظیمات به دیکشنری |
| `__str__()` | - | رشته | نمایش رشته‌ای تنظیمات |

##### `logger.py`

**کلاس `Logger`**

این کلاس سیستم لاگینگ یکپارچه را برای کل پروژه فراهم می‌کند و با استفاده از الگوی Singleton پیاده‌سازی شده است.

| متد | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `__new__(cls, name='crypto_news_bot')` | کلاس، نام لاگر | نمونه منحصر به فرد کلاس | الگوی Singleton |
| `_configure(name)` | نام لاگر | - | پیکربندی اولیه لاگر |
| `_get_logger(name)` | نام لاگر | شیء لاگر | ایجاد یا بازیابی یک لاگر با نام مشخص |
| `get_logger(name=None)` | نام لاگر | شیء لاگر | دریافت یک لاگر با نام مشخص |
| `debug(message, *args, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام در سطح DEBUG |
| `info(message, *args, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام در سطح INFO |
| `warning(message, *args, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام در سطح WARNING |
| `error(message, *args, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام در سطح ERROR |
| `critical(message, *args, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام در سطح CRITICAL |
| `exception(message, *args, exc_info=True, **kwargs)` | پیام، آرگومان‌ها | - | ثبت پیام استثنا با جزئیات خطا |

##### `helpers.py`

این فایل شامل توابع کمکی متنوعی است که در سراسر پروژه استفاده می‌شوند.

| تابع | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `now()` | - | datetime | زمان فعلی با منطقه زمانی تهران |
| `to_jalali(date)` | تاریخ میلادی | تاریخ شمسی | تبدیل تاریخ میلادی به شمسی |
| `format_jalali_date(date, format_str='%Y/%m/%d')` | تاریخ، قالب | رشته | فرمت‌بندی تاریخ شمسی |
| `format_jalali_datetime(date, format_str='%Y/%m/%d %H:%M:%S')` | تاریخ، قالب | رشته | فرمت‌بندی تاریخ و زمان شمسی |
| `human_readable_time(seconds)` | ثانیه | رشته | تبدیل ثانیه به فرمت قابل خواندن انسان |
| `time_ago(date)` | تاریخ | رشته | زمان سپری شده از یک تاریخ تا کنون |
| `clean_text(text)` | متن | متن | پاکسازی متن از کاراکترهای اضافی |
| `extract_urls(text)` | متن | لیست URL‌ها | استخراج URL‌ها از متن |
| `extract_hashtags(text)` | متن | لیست هشتگ‌ها | استخراج هشتگ‌ها از متن |
| `extract_mentions(text)` | متن | لیست منشن‌ها | استخراج منشن‌ها از متن |
| `extract_numbers(text)` | متن | لیست اعداد | استخراج اعداد از متن |
| `extract_percentages(text)` | متن | لیست درصدها | استخراج درصدها از متن |
| `extract_cryptocurrencies(text)` | متن | لیست ارزها | استخراج نام‌های ارزهای دیجیتال |
| `generate_random_string(length=10)` | طول | رشته | تولید یک رشته تصادفی |
| `compute_hash(text)` | متن | هش | محاسبه هش SHA256 یک متن |
| `compute_similarity_hash(text)` | متن | هش | محاسبه هش ساده برای مقایسه متن‌ها |
| `retry(max_tries=3, delay=1, backoff=2, exceptions=(Exception,))` | پارامترها | دکوراتور | دکوراتور تلاش مجدد برای توابع |
| `chunks(lst, chunk_size)` | لیست، اندازه | لیست بخش‌ها | تقسیم یک لیست به چند بخش کوچکتر |
| `limit_text_length(text, max_length=4000, suffix="...")` | متن، طول، پسوند | متن | محدود کردن طول متن |
| `is_valid_url(url)` | URL | بولین | بررسی اعتبار یک URL |
| `remove_html_tags(text)` | متن | متن | حذف تگ‌های HTML از متن |

### 2. ماژول کش (cache)

ماژول `cache` مسئول مدیریت کش با استفاده از Redis است.

#### مسیر
```
crypto_news_bot/cache/
```

#### ساختار فایل‌بندی
```
cache/
├── __init__.py                # تعریف پکیج و اکسپورت‌های اصلی
└── redis_manager.py           # مدیریت کش Redis
```

#### فایل‌ها و کلاس‌ها

##### `redis_manager.py`

**کلاس `RedisManager`**

این کلاس مسئول مدیریت ارتباط با Redis و انجام عملیات کش‌گذاری است و با استفاده از الگوی Singleton پیاده‌سازی شده است.

| متد | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `__new__(cls)` | کلاس | نمونه منحصر به فرد کلاس | الگوی Singleton |
| `_initialize()` | - | - | راه‌اندازی اولیه کلاس و اتصال به Redis |
| `reconnect()` | - | بولین | تلاش مجدد برای اتصال به Redis |
| `_ensure_connected()` | - | بولین | اطمینان از اتصال به Redis |
| **روش‌های اصلی کش** |
| `get(key)` | کلید | مقدار | دریافت مقدار یک کلید |
| `set(key, value, expire=None)` | کلید، مقدار، انقضا | بولین | تنظیم مقدار یک کلید |
| `delete(key)` | کلید | بولین | حذف یک کلید |
| `exists(key)` | کلید | بولین | بررسی وجود یک کلید |
| `expire(key, time)` | کلید، زمان | بولین | تنظیم زمان انقضا برای یک کلید |
| **روش‌های کار با اشیاء پایتون** |
| `get_object(key)` | کلید | شیء | دریافت یک شیء پایتون از کش |
| `set_object(key, value, expire=None)` | کلید، شیء، انقضا | بولین | ذخیره یک شیء پایتون در کش |
| **روش‌های کار با JSON** |
| `get_json(key)` | کلید | دیکشنری | دریافت یک شیء JSON از کش |
| `set_json(key, value, expire=None)` | کلید، دیکشنری، انقضا | بولین | ذخیره یک شیء JSON در کش |
| **روش‌های کار با لیست** |
| `list_push(key, value, left=False)` | کلید، مقدار، چپ | بولین | افزودن یک مقدار به لیست |
| `list_pop(key, left=False)` | کلید، چپ | مقدار | حذف و بازگرداندن یک مقدار از لیست |
| `list_length(key)` | کلید | عدد | طول یک لیست |
| `list_range(key, start=0, end=-1)` | کلید، شروع، پایان | لیست | دریافت محدوده‌ای از لیست |
| **روش‌های کار با مجموعه** |
| `set_add(key, value)` | کلید، مقدار | بولین | افزودن یک مقدار به مجموعه |
| `set_remove(key, value)` | کلید، مقدار | بولین | حذف یک مقدار از مجموعه |
| `set_members(key)` | کلید | مجموعه | دریافت تمام اعضای مجموعه |
| `set_is_member(key, value)` | کلید، مقدار | بولین | بررسی عضویت یک مقدار در مجموعه |
| **روش‌های کار با Hash** |
| `hash_set(key, field, value)` | کلید، فیلد، مقدار | بولین | تنظیم مقدار یک فیلد در هش |
| `hash_get(key, field)` | کلید، فیلد | مقدار | دریافت مقدار یک فیلد از هش |
| `hash_delete(key, field)` | کلید، فیلد | بولین | حذف یک فیلد از هش |
| `hash_get_all(key)` | کلید | دیکشنری | دریافت تمام فیلدها و مقادیر یک هش |
| **روش‌های کار با Sorted Set** |
| `zset_add(key, value, score)` | کلید، مقدار، امتیاز | بولین | افزودن یک مقدار به مجموعه مرتب‌شده |
| `zset_range(key, start=0, end=-1, with_scores=False, desc=False)` | کلید، شروع، پایان، با امتیاز، نزولی | لیست | دریافت محدوده‌ای از مجموعه مرتب‌شده |
| `zset_remove(key, value)` | کلید، مقدار | بولین | حذف یک مقدار از مجموعه مرتب‌شده |
| **روش‌های مدیریت کلیدها** |
| `delete_pattern(pattern)` | الگو | تعداد | حذف کلیدهای منطبق با یک الگو |
| `flush_db()` | - | بولین | پاک کردن کل دیتابیس |
| `get_keys(pattern="*")` | الگو | لیست | دریافت کلیدهای منطبق با یک الگو |
| `get_info()` | - | دیکشنری | دریافت اطلاعات Redis |

### 3. ماژول دیتابیس (database)

ماژول `database` مسئول مدیریت دسترسی به دیتابیس و مدل‌های داده است.

#### مسیر
```
crypto_news_bot/database/
```

#### ساختار فایل‌بندی
```
database/
├── __init__.py                # تعریف پکیج و اکسپورت‌های اصلی
├── models.py                  # تعریف مدل‌های داده
├── db_manager.py              # مدیریت ارتباط با دیتابیس
└── query_manager.py           # مدیریت کوئری‌های پیچیده
```

#### فایل‌ها و کلاس‌ها

##### `models.py`

این فایل شامل تعریف مدل‌های داده‌ای است که در دیتابیس ذخیره می‌شوند.

**انواع داده‌ای (Enums):**

| کلاس | توضیحات |
|------|---------|
| `NewsSource` | منابع خبری پشتیبانی شده (TELEGRAM, WEBSITE, TWITTER) |
| `NewsCategory` | دسته‌بندی‌های خبری (PRICE, REGULATION, ADOPTION, TECHNOLOGY, SECURITY, EXCHANGE, GENERAL, OTHER) |
| `SentimentType` | انواع احساسات متن (POSITIVE, NEGATIVE, NEUTRAL) |
| `PublishStatus` | وضعیت‌های انتشار خبر (PENDING, SCHEDULED, PUBLISHED, REJECTED, FAILED) |

**مدل‌های داده:**

| کلاس | توضیحات | فیلدهای اصلی |
|------|---------|--------------|
| `RawNews` | خبر خام دریافتی از منابع | id, external_id, source, source_name, title, text, language_code, url, has_media, media_urls, published_at, collected_at, text_hash |
| `NewsAnalysis` | تحلیل محتوای خبر | id, raw_news_id, primary_category, category_confidence, secondary_categories, sentiment, sentiment_score, importance_score, importance_factors, analyzed_at |
| `GeneratedContent` | محتوای تولید شده برای انتشار | id, raw_news_id, title, summary, full_text, has_media, media_urls, generated_at, publish_status, scheduled_for, published_at, publish_message_id |
| `UserFeedback` | بازخورد کاربران | id, generated_content_id, user_id, user_name, feedback_type, feedback_value, feedback_text, created_at |
| `TrainingData` | داده‌های آموزشی | id, data_type, target_field, input_text, expected_output, source_type, source_id, metadata, created_at, used_for_training, training_date |
| `SourceCredibility` | اعتبار منابع خبری | id, source, source_name, credibility_score, reliability_score, importance_factor, description, last_updated |

##### `db_manager.py`

**کلاس `DatabaseManager`**

این کلاس مسئول مدیریت ارتباط با دیتابیس و انجام عملیات پایه است و با استفاده از الگوی Singleton پیاده‌سازی شده است.

| متد | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `__new__(cls)` | کلاس | نمونه منحصر به فرد کلاس | الگوی Singleton |
| `_initialize()` | - | - | راه‌اندازی اولیه کلاس و اتصال به دیتابیس |
| `create_tables()` | - | - | ایجاد جداول دیتابیس |
| `drop_tables()` | - | - | حذف تمام جداول دیتابیس |
| `get_session()` | - | جلسه | دریافت یک جلسه دیتابیس |
| `session_scope()` | - | جلسه | مدیریت جلسه دیتابیس با context manager |
| `execute_raw_sql(sql, params=None)` | SQL، پارامترها | نتایج | اجرای یک دستور SQL خام |
| **روش‌های عمومی دسترسی به داده** |
| `add(obj)` | شیء | شیء | افزودن یک شیء به دیتابیس |
| `add_all(objects)` | لیست اشیاء | لیست اشیاء | افزودن چندین شیء به دیتابیس |
| `get_by_id(model_class, id)` | کلاس مدل، شناسه | شیء | دریافت یک شیء با شناسه آن |
| `get_all(model_class, limit=None, offset=0)` | کلاس مدل، محدودیت، آفست | لیست اشیاء | دریافت تمام اشیاء یک مدل |
| `update(obj)` | شیء | شیء | به‌روزرسانی یک شیء در دیتابیس |
| `delete(obj)` | شیء | بولین | حذف یک شیء از دیتابیس |
| `delete_by_id(model_class, id)` | کلاس مدل، شناسه | بولین | حذف یک شیء با شناسه آن |
| `count(model_class)` | کلاس مدل | عدد | شمارش تعداد اشیاء یک مدل |
| `truncate(model_class)` | کلاس مدل | - | حذف تمام اشیاء یک مدل |
| `get_table_names()` | - | لیست | دریافت نام تمام جداول دیتابیس |
| `get_db_size()` | - | دیکشنری | دریافت اندازه دیتابیس و جداول |

##### `query_manager.py`

**کلاس `QueryManager`**

این کلاس مسئول اجرای کوئری‌های پیچیده و مدیریت دسترسی به داده‌هاست و با استفاده از الگوی Singleton پیاده‌سازی شده است.

| متد | ورودی | خروجی | توضیحات |
|-----|------|-------|---------|
| `__new__(cls)` | کلاس | نمونه منحصر به فرد کلاس | الگوی Singleton |
| `_initialize()` | - | - | راه‌اندازی اولیه کلاس |
| **کوئری‌های مربوط به RawNews** |
| `add_raw_news(external_id, source, source_name, text, ...)` | پارامترها | RawNews | افزودن یک خبر خام جدید |
| `is_duplicate_news(text, threshold=0.9, hours=24)` | متن، آستانه، ساعت | بولین | بررسی تکراری بودن یک خبر |
| `get_unanalyzed_news(limit=10)` | محدودیت | لیست | دریافت خبرهای تحلیل نشده |
| `get_news_by_source(source, source_name=None, limit=100, offset=0)` | منبع، نام منبع، محدودیت، آفست | لیست | دریافت خبرها بر اساس منبع |
| `search_news(query_text, limit=100, offset=0)` | متن جستجو، محدودیت، آفست | لیست | جستجوی خبرها بر اساس متن |
| **کوئری‌های مربوط به NewsAnalysis** |
| `add_news_analysis(raw_news_id, ...)` | پارامترها | NewsAnalysis | افزودن تحلیل برای یک خبر |
| `get_news_with_analysis(limit=100, offset=0)` | محدودیت، آفست | لیست تاپل‌ها | دریافت خبرها به همراه تحلیل |
| `get_important_news(min_score=0.7, limit=10)` | حداقل امتیاز، محدودیت | لیست تاپل‌ها | دریافت خبرهای مهم |
| `get_news_by_category(category, limit=100, offset=0)` | دسته‌بندی، محدودیت، آفست | لیست تاپل‌ها | دریافت خبرها بر اساس دسته‌بندی |
| **کوئری‌های مربوط به GeneratedContent** |
| `add_generated_content(raw_news_id, ...)` | پارامترها | GeneratedContent | افزودن محتوای تولید شده |
| `get_content_for_publication(limit=10)` | محدودیت | لیست | دریافت محتواهای آماده انتشار |
| `update_content_status(content_id, status, message_id=None)` | شناسه، وضعیت، شناسه پیام | بولین | به‌روزرسانی وضعیت انتشار |
| **کوئری‌های مربوط به UserFeedback** |
| `add_user_feedback(generated_content_id, ...)` | پارامترها | UserFeedback | افزودن بازخورد کاربر |
| `get_content_feedback(content_id)` | شناسه محتوا | لیست | دریافت بازخوردهای یک محتوا |
| `get_feedback_stats()` | - | دیکشنری | دریافت آمار بازخوردها |
| **کوئری‌های مربوط به TrainingData** |
| `add_training_data(data_type, ...)` | پارامترها | TrainingData | افزودن داده آموزشی |
| `get_training_data(data_type=None, used=None, limit=1000)` | نوع داده، استفاده شده، محدودیت | لیست | دریافت داده‌های آموزشی |
| `mark_training_data_used(data_ids)` | لیست شناسه‌ها | بولین | علامت‌گذاری داده‌های آموزشی |
| **کوئری‌های مربوط به SourceCredibility** |
| `get_source_credibility(source, source_name)` | منبع، نام منبع | SourceCredibility | دریافت اعتبار یک منبع |
| `update_source_credibility(source, source_name, ...)` | پارامترها | SourceCredibility | به‌روزرسانی اعتبار منبع |
| **کوئری‌های ترکیبی و گزارش‌گیری** |
| `get_news_stats(days=30)` | روزها | دیکشنری | دریافت آمار خبرها |
| `get_daily_news_counts(days=30)` | روزها | لیست | دریافت تعداد خبرهای روزانه |
| `get_content_generation_stats()` | - | دیکشنری | دریافت آمار تولید محتوا |


## ماژول جمع‌آوری داده (collector)

ماژول `collector` مسئول جمع‌آوری داده‌ها از منابع مختلف شامل کانال‌های تلگرام، وب‌سایت‌های خبری و توییتر است. هدف اصلی این ماژول، دریافت مستمر و خودکار اخبار به منظور پردازش‌های بعدی است.

### اهداف ماژول

1. **جمع‌آوری مستمر اخبار** از منابع مختلف
2. **پردازش اولیه داده‌ها** و آماده‌سازی آنها برای تحلیل و ذخیره‌سازی

### مسیر

```
crypto_news_bot/collector/
```

### ساختار فایل‌بندی

```
collector/
├── __init__.py                  # تعریف پکیج و کلاس‌های قابل استفاده
├── collection_manager.py        # مدیریت جمع‌آوری داده‌ها
├── telegram_collector.py        # جمع‌آوری داده از تلگرام
├── twitter_collector.py         # جمع‌آوری داده از توییتر
└── website_collector.py         # جمع‌آوری داده از وب‌سایت‌ها
```

### فایل‌ها و کلاس‌ها

#### `collection_manager.py`

**کلاس `CollectionManager`**

این کلاس مسئول هماهنگی و مدیریت فرآیندهای جمع‌آوری داده از منابع مختلف است.

| متد | ورودی | خروجی | توضیحات |
|-----|-------|-------|----------|
| `__init__()` | - | - | راه‌اندازی اولیه مدیریت |
| `_load_last_runs()` | - | دیکشنری | بارگذاری زمان آخرین اجرای جمع‌آوری |
| `_save_last_runs()` | - | - | ذخیره‌سازی زمان‌های آخرین اجرا |
| `_should_collect(source)` | نام منبع | بولین | بررسی نیاز به اجرای جمع‌آوری داده |
| `update_collection_stats(source, stats)` | نام منبع، آمار | - | به‌روزرسانی آمار جمع‌آوری |
| `collect_from_telegram(force=False)` | اجبار | دیکشنری | جمع‌آوری داده از تلگرام |
| `collect_from_websites(force=False)` | اجبار | دیکشنری | جمع‌آوری داده از وب‌سایت‌ها |
| `collect_from_twitter(force=False)` | اجبار | دیکشنری | جمع‌آوری داده از توییتر |
| `collect_all(force=False)` | اجبار | دیکشنری | جمع‌آوری همزمان از تمام منابع |

#### `telegram_collector.py`

**کلاس `TelegramCollector`**

این کلاس مسئول جمع‌آوری اخبار از کانال‌های تلگرام است.

| متد | ورودی | خروجی | توضیحات |
|-----|-------|-------|----------|
| `__init__(session_name=None)` | نام سشن | - | راه‌اندازی اولیه |
| `connect()` | - | بولین | اتصال به تلگرام |
| `disconnect()` | - | - | قطع اتصال از تلگرام |
| `get_channel_info(channel_username)` | نام کانال | دیکشنری | دریافت اطلاعات کانال |
| `get_last_message_id(channel_username)` | نام کانال | عدد | دریافت آخرین شناسه پیام پردازش شده |
| `set_last_message_id(channel_username, message_id)` | نام کانال، شناسه پیام | - | تنظیم آخرین شناسه پیام پردازش شده |
| `process_media(message)` | پیام | لیست | پردازش رسانه پیام |
| `process_message(message, channel_username)` | پیام، نام کانال | دیکشنری | پردازش پیام |
| `save_message(message_data)` | اطلاعات پیام | بولین | ذخیره پیام در دیتابیس |
| `collect_from_channel(channel_username, limit=None)` | نام کانال، محدودیت | tuple | جمع‌آوری پیام‌ها از یک کانال |
| `run()` | - | دیکشنری | اجرای جمع‌آوری از کانال‌ها |

#### `twitter_collector.py`

**کلاس `TwitterCollector`**

این کلاس مسئول جمع‌آوری اخبار از توییتر است.

| متد | ورودی | خروجی | توضیحات |
|-----|-------|-------|----------|
| `__init__()` | - | - | راه‌اندازی اولیه |
| `process_tweet(tweet)` | توییت | دیکشنری | پردازش توییت |
| `save_tweet(tweet_data)` | اطلاعات توییت | بولین | ذخیره توییت در دیتابیس |
| `_build_search_query(query_term, since_id=None, days_limit=None)` | عبارت جستجو | رشته | ساخت کوئری جستجو |
| `_collect_tweets_with_snscrape(query, max_tweets)` | کوئری، حداکثر تعداد | لیست | جمع‌آوری توییت‌ها |
| `collect_from_twitter(query_term)` | عبارت جستجو | tuple | جمع‌آوری توییت‌ها |
| `collect_from_accounts(accounts=None)` | لیست حساب‌ها | دیکشنری | جمع‌آوری توییت‌ها از حساب‌ها |
| `collect_from_hashtags(hashtags=None)` | لیست هشتگ‌ها | دیکشنری | جمع‌آوری توییت‌ها از هشتگ‌ها |
| `run()` | - | دیکشنری | اجرای جمع‌آوری |

#### `website_collector.py`

**کلاس `WebsiteCollector`**

این کلاس مسئول جمع‌آوری اخبار از وب‌سایت‌های خبری است.

| متد | ورودی | خروجی | توضیحات |
|-----|-------|-------|----------|
| `__init__()` | - | - | راه‌اندازی اولیه |
| `check_robots_allowed(session, url)` | جلسه، آدرس | بولین | بررسی دسترسی از طریق robots.txt |
| `fetch_url(session, url)` | جلسه، آدرس | رشته | دریافت محتوای URL |
| `extract_links(html, base_url)` | محتوای HTML، آدرس پایه | لیست | استخراج لینک‌ها |
| `filter_news_links(links, website_config)` | لیست لینک‌ها، تنظیمات | لیست | فیلتر کردن لینک‌ها |
| `process_article(html, url, website_config)` | HTML، آدرس، تنظیمات | دیکشنری | پردازش مقاله |
| `save_article(article_info)` | اطلاعات مقاله | بولین | ذخیره مقاله در دیتابیس |
| `collect_from_website(website_config)` | تنظیمات وب‌سایت | tuple | جمع‌آوری از یک وب‌سایت |
| `run()` | - | دیکشنری | اجرای جمع‌آوری از وب‌سایت‌ها |

