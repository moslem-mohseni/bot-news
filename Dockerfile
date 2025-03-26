# Dockerfile for CryptoNewsBot
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تنظیم دایرکتوری کاری
WORKDIR /app

# نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# کپی کردن کدها
COPY . .

# اجرای برنامه
CMD ["python", "main.py"]
