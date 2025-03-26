"""
ماژول تحلیل‌گر بازار ارزهای دیجیتال برای CryptoNewsBot

این ماژول تحلیل‌های پیشرفته بازار را با ترکیب داده‌های قیمتی و اخبار فراهم می‌کند.
قابلیت‌های اصلی آن شامل: پیش‌بینی روند بازار بر اساس اخبار، تعیین ضریب تأثیر خبر بر ارزهای مختلف،
تولید گزارش تحلیلی از تأثیر خبر، و ارائه توصیه‌های سرمایه‌گذاری است.
"""

import re
import torch
import numpy as np
import json
import time
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from collections import defaultdict, deque

from .base_model import BaseModel, retry_on_error


def _generate_risk_analysis_text(analysis_result: Dict[str, Any]) -> str:
    """
    تولید متن تحلیلی برای تحلیل ریسک

    Args:
        analysis_result: نتایج تحلیل ریسک

    Returns:
        متن تحلیلی
    """
    symbol = analysis_result.get("symbol", "")
    risk_level = analysis_result.get("risk_level", "")
    risk_fa = {"high": "بالا", "medium": "متوسط", "low": "پایین"}.get(risk_level, "")

    volatility = analysis_result.get("volatility", 0)
    max_drawdown = analysis_result.get("max_drawdown", 0)
    sharpe_ratio = analysis_result.get("sharpe_ratio", 0)

    # سناریوها
    scenarios = analysis_result.get("scenarios", {})
    current_price = scenarios.get("current_price", 0)
    base_scenario = scenarios.get("base", 0)
    optimistic_scenario = scenarios.get("optimistic", 0)
    pessimistic_scenario = scenarios.get("pessimistic", 0)

    # ساخت متن تحلیل
    text = f"تحلیل ریسک برای {symbol}:\n\n"
    text += f"سطح ریسک کلی: {risk_fa}\n"
    text += f"نوسانات (انحراف معیار): {volatility}٪\n"
    text += f"حداکثر افت (Drawdown): {max_drawdown}٪\n"
    text += f"نسبت شارپ: {sharpe_ratio}\n"

    # ارزش در معرض خطر
    var_95 = analysis_result.get("var_95", 0)
    var_99 = analysis_result.get("var_99", 0)

    text += f"ارزش در معرض خطر (95٪): {abs(var_95)}٪\n"
    text += f"ارزش در معرض خطر (99٪): {abs(var_99)}٪\n"

    # سناریوها
    text += f"\nسناریوهای قیمت (یک‌ماهه):\n"
    text += f"قیمت فعلی: {current_price}\n"

    # محاسبه درصد تغییر
    base_change = ((base_scenario / current_price) - 1) * 100
    optimistic_change = ((optimistic_scenario / current_price) - 1) * 100
    pessimistic_change = ((pessimistic_scenario / current_price) - 1) * 100

    base_direction = "افزایش" if base_change > 0 else "کاهش"
    optimistic_direction = "افزایش" if optimistic_change > 0 else "کاهش"
    pessimistic_direction = "افزایش" if pessimistic_change > 0 else "کاهش"

    text += f"سناریوی پایه: {base_scenario} ({abs(base_change):.1f}٪ {base_direction})\n"
    text += f"سناریوی خوش‌بینانه: {optimistic_scenario} ({abs(optimistic_change):.1f}٪ {optimistic_direction})\n"
    text += f"سناریوی بدبینانه: {pessimistic_scenario} ({abs(pessimistic_change):.1f}٪ {pessimistic_direction})\n"

    # توضیحات تکمیلی بر اساس سطح ریسک
    text += f"\nتفسیر: "

    if risk_level == "high":
        text += "این ارز دارای نوسانات بالا و ریسک زیادی است. سرمایه‌گذاری در آن مناسب افراد ریسک‌پذیر است و توصیه می‌شود حجم معاملات پایین نگه داشته شود."
    elif risk_level == "medium":
        text += "این ارز دارای نوسانات و ریسک متوسطی است. برای سرمایه‌گذاری میان‌مدت مناسب است اما همچنان نیاز به مدیریت ریسک دارد."
    else:
        text += "این ارز نسبت به سایر ارزهای دیجیتال نوسانات کمتری دارد و برای سرمایه‌گذاری بلندمدت یا افراد محتاط مناسب‌تر است."

    return text


def _generate_market_report_text(analysis_result: Dict[str, Any]) -> str:
    """
    تولید متن تحلیلی برای گزارش بازار

    Args:
        analysis_result: نتایج تحلیل گزارش بازار

    Returns:
        متن تحلیلی
    """
    symbol = analysis_result.get("symbol", "")
    current_price = analysis_result.get("current_price", 0)
    price_change_24h = analysis_result.get("price_change_24h", 0)
    price_change_7d = analysis_result.get("price_change_7d", 0)

    price_trend = analysis_result.get("price_trend", {})
    trend = price_trend.get("trend_fa", "")

    predicted_trend = analysis_result.get("predicted_trend", {})
    predicted_trend_text = predicted_trend.get("trend_fa", "")
    predicted_timeframe = predicted_trend.get("timeframe_fa", "")

    # الگوهای تکنیکال
    technical_patterns = analysis_result.get("technical_patterns", [])
    patterns_text = ""

    if technical_patterns:
        patterns_text = "الگوهای تکنیکال شناسایی شده:\n"
        for pattern in technical_patterns:
            patterns_text += f"- {pattern.get('name', '')}: {pattern.get('description', '')}\n"

    # سطوح حمایت و مقاومت
    support_resistance = analysis_result.get("support_resistance", {})
    levels_text = ""

    resistance_levels = support_resistance.get("resistance_levels", [])
    if resistance_levels:
        levels_text += "سطوح مقاومت:\n"
        for level in resistance_levels:
            levels_text += f"- {level.get('price', 0)} (فاصله: {level.get('distance', 0)}٪، قدرت: {level.get('strength', 0)})\n"

    support_levels = support_resistance.get("support_levels", [])
    if support_levels:
        levels_text += "سطوح حمایت:\n"
        for level in support_levels:
            levels_text += f"- {level.get('price', 0)} (فاصله: {level.get('distance', 0)}٪، قدرت: {level.get('strength', 0)})\n"

    # ساخت متن نهایی
    text = f"گزارش بازار برای {symbol}:\n\n"
    text += f"قیمت فعلی: {current_price}\n"

    # تغییرات قیمت
    change_24h_text = "افزایش" if price_change_24h > 0 else "کاهش"
    text += f"تغییرات 24 ساعته: {abs(price_change_24h)}٪ {change_24h_text}\n"

    change_7d_text = "افزایش" if price_change_7d > 0 else "کاهش"
    text += f"تغییرات 7 روزه: {abs(price_change_7d)}٪ {change_7d_text}\n\n"

    # روند
    text += f"روند فعلی: {trend}\n"
    if predicted_trend_text and predicted_timeframe:
        text += f"پیش‌بینی روند در {predicted_timeframe}: {predicted_trend_text}\n"
        if "description" in predicted_trend:
            text += f"{predicted_trend['description']}\n"

    # الگوهای تکنیکال
    if patterns_text:
        text += f"\n{patterns_text}\n"

    # سطوح حمایت و مقاومت
    if levels_text:
        text += f"\n{levels_text}\n"

    # تأثیر اخبار
    news_impact = analysis_result.get("news_impact")
    if news_impact:
        sentiment = news_impact.get("dominant_sentiment")
        sentiment_fa = {"positive": "مثبت", "negative": "منفی", "neutral": "خنثی"}.get(sentiment, "")
        text += f"\nتحلیل اخبار اخیر ({news_impact.get('count', 0)} خبر):\n"
        text += f"احساس غالب: {sentiment_fa}\n"
        text += f"میانگین تأثیر: {news_impact.get('avg_impact_score', 0)}\n"

    return text


class MarketAnalyzer(BaseModel):
    """
    تحلیل‌گر بازار ارزهای دیجیتال

    این کلاس با ترکیب تحلیل محتوای خبری، داده‌های قیمتی و الگوهای بازار،
    تحلیل‌های پیشرفته و پیش‌بینی روند بازار ارزهای دیجیتال را ارائه می‌دهد.

    کاربردهای اصلی این تحلیل‌گر شامل:
    - تشخیص پتانسیل تأثیر اخبار بر قیمت ارزهای مختلف
    - تعیین ضریب تأثیر خبر بر هر ارز
    - پیش‌بینی کوتاه‌مدت و میان‌مدت روند بازار
    - تولید گزارش‌های تحلیلی قابل فهم برای کاربران
    """

    # انواع روندهای بازار
    MARKET_TRENDS = {
        'strong_bullish': 'صعودی قوی',
        'bullish': 'صعودی',
        'weak_bullish': 'صعودی ضعیف',
        'neutral': 'خنثی',
        'weak_bearish': 'نزولی ضعیف',
        'bearish': 'نزولی',
        'strong_bearish': 'نزولی قوی',
        'volatile': 'پرنوسان'
    }

    # بازه‌های زمانی پیش‌بینی
    TIME_FRAMES = {
        'immediate': 'فوری (24 ساعت)',
        'short_term': 'کوتاه‌مدت (1 هفته)',
        'mid_term': 'میان‌مدت (1 ماه)',
        'long_term': 'بلندمدت (3 ماه)'
    }

    # سطوح اطمینان تحلیل
    CONFIDENCE_LEVELS = {
        'very_high': 'بسیار بالا',
        'high': 'بالا',
        'moderate': 'متوسط',
        'low': 'پایین',
        'very_low': 'بسیار پایین'
    }

    def __init__(self) -> None:
        """راه‌اندازی اولیه تحلیل‌گر بازار"""
        super().__init__(model_name="MarketAnalyzer", priority=7)  # اولویت پایین‌تر از financial_model

        # پیکربندی‌های خاص مدل
        self.model_config.update({
            'default_time_window': 30,  # پنجره زمانی پیش‌فرض (روز)
            'default_confidence_threshold': 0.65,  # حداقل اطمینان قابل قبول
            'impact_calculation_weights': {
                'sentiment_weight': 0.3,  # وزن احساسات خبر
                'importance_weight': 0.25,  # وزن اهمیت خبر
                'specificity_weight': 0.25,  # وزن اختصاصی بودن خبر برای ارز
                'source_weight': 0.2,  # وزن اعتبار منبع خبر
            },
            'trend_memory_size': 50,  # تعداد تحلیل‌های قبلی برای یادگیری
            'market_correlation_window': 14,  # پنجره زمانی برای همبستگی بازار (روز)
            'volatility_threshold': 0.1,  # آستانه تشخیص بازار پرنوسان (تغییرات 10%)
            'min_news_for_analysis': 3,  # حداقل تعداد خبر برای تحلیل معتبر
        })

        # وضعیت مدل‌های مختلف و داده‌ها
        self.price_data_loaded = False
        self.correlation_matrix_loaded = False
        self.market_patterns_loaded = False
        self.historical_impact_loaded = False

        # داده‌های داخلی
        self._price_data = {}
        self._correlation_matrix = {}
        self._market_patterns = {}
        self._historical_impact = {}
        self._trend_memory = defaultdict(lambda: deque(maxlen=self.model_config['trend_memory_size']))

        # نگاشت‌های داخلی
        self._crypto_importance = {}  # اهمیت نسبی هر ارز
        self._source_credibility = {}  # اعتبار منابع خبری

        self.logger.debug("تحلیل‌گر بازار راه‌اندازی شد")

    def _setup_model(self) -> None:
        """تنظیمات اولیه مدل - هیچ کاری انجام نمی‌دهد (بارگیری تنبل)"""
        pass

    def load_model(self) -> None:
        """
        بارگیری اولیه مدل‌های مورد نیاز

        این متد برای بارگیری پایه و اطمینان از عملکرد صحیح است.
        هر مجموعه داده به صورت جداگانه با بارگیری تنبل مدیریت می‌شود.
        """
        if self.model_loaded:
            return

        try:
            # بارگیری حداقل داده‌های مورد نیاز
            self._load_crypto_importance()
            self._load_source_credibility()

            self.model_loaded = True
            self.logger.info("تحلیل‌گر بازار با موفقیت بارگیری شد")

        except Exception as e:
            self.logger.error(f"خطا در بارگیری تحلیل‌گر بازار: {str(e)}")
            raise

    def _load_crypto_importance(self) -> None:
        """بارگیری اهمیت نسبی ارزهای دیجیتال"""
        if self._crypto_importance:
            return

        self.logger.debug("در حال بارگیری اطلاعات اهمیت ارزهای دیجیتال...")

        # دریافت داده‌های ارزها از مخزن داده
        crypto_data = self.data_repo.get_data('crypto')
        if not crypto_data:
            self.logger.warning("داده‌های ارزهای دیجیتال یافت نشد، استفاده از مقادیر پیش‌فرض")
            # مقادیر پیش‌فرض برای ارزهای اصلی
            self._crypto_importance = {
                "BTC": 1.0,
                "ETH": 0.9,
                "BNB": 0.8,
                "XRP": 0.75,
                "ADA": 0.7,
                "SOL": 0.7,
                "DOGE": 0.65,
                "DOT": 0.65,
                "SHIB": 0.6
            }
        else:
            # استخراج مقادیر اهمیت از داده‌های مخزن
            for symbol, data in crypto_data.items():
                self._crypto_importance[symbol] = data.get('importance', 0.5)

        self.logger.debug(f"اطلاعات اهمیت {len(self._crypto_importance)} ارز دیجیتال بارگیری شد")

    def _load_source_credibility(self) -> None:
        """بارگیری اعتبار منابع خبری"""
        if self._source_credibility:
            return

        self.logger.debug("در حال بارگیری اطلاعات اعتبار منابع خبری...")

        # مقادیر پیش‌فرض اعتبار منابع
        # در نسخه‌های آینده می‌تواند از دیتابیس بارگیری شود
        self._source_credibility = {
            # منابع خبری معتبر انگلیسی
            "coindesk": 0.95,
            "cointelegraph": 0.9,
            "theblock": 0.9,
            "decrypt": 0.85,
            "cryptoslate": 0.8,
            # منابع فارسی
            "arzdigital": 0.85,
            "ramzarz": 0.8,
            "wallex": 0.8,
            # کانال‌های تلگرام معتبر
            "telegram:cryptomohem": 0.85,
            "telegram:arzcoinews": 0.8,
            "telegram:coiniran": 0.8,
            # توییتر
            "twitter:binance": 0.9,
            "twitter:coinbase": 0.9,
            "twitter:krakenfx": 0.85,
            # مقادیر پیش‌فرض برای منابع ناشناخته
            "default_website": 0.7,
            "default_telegram": 0.65,
            "default_twitter": 0.6,
            "default_unknown": 0.5
        }

        self.logger.debug(f"اطلاعات اعتبار {len(self._source_credibility)} منبع خبری بارگیری شد")

    def _load_price_data(self) -> None:
        """بارگیری داده‌های قیمتی ارزهای دیجیتال"""
        if self.price_data_loaded:
            return

        self.logger.info("در حال بارگیری داده‌های قیمتی ارزهای دیجیتال...")

        try:
            # در نسخه واقعی، داده‌ها از API یا دیتابیس بارگیری می‌شوند
            # برای نمونه، داده‌های ساختگی ایجاد می‌کنیم
            self._price_data = self._generate_sample_price_data()

            self.price_data_loaded = True
            self.logger.info("داده‌های قیمتی ارزهای دیجیتال با موفقیت بارگیری شد")

        except Exception as e:
            self.logger.error(f"خطا در بارگیری داده‌های قیمتی: {str(e)}")
            raise

    def _generate_sample_price_data(self) -> Dict[str, Any]:
        """
        تولید داده‌های قیمتی نمونه برای استفاده در توسعه و تست

        در محیط تولید، این متد با دریافت داده‌های واقعی از API یا دیتابیس جایگزین می‌شود.

        Returns:
            دیکشنری داده‌های قیمتی نمونه
        """
        price_data = {}
        now = datetime.datetime.now()

        # ارزهای مورد نظر برای تولید داده نمونه
        symbols = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT"]

        # قیمت‌های پایه
        base_prices = {
            "BTC": 51000.0,
            "ETH": 2800.0,
            "BNB": 450.0,
            "XRP": 0.55,
            "ADA": 0.45,
            "SOL": 110.0,
            "DOGE": 0.08,
            "DOT": 7.5,
        }

        # تولید داده‌های قیمتی روزانه برای 90 روز گذشته
        for symbol in symbols:
            base_price = base_prices[symbol]
            daily_data = []

            # الگوی قیمت با نوسانات تصادفی و روند کلی
            trend = np.random.choice([-1, 0, 1])  # روند نزولی، خنثی یا صعودی
            volatility = base_price * 0.02  # نوسان 2%

            for day in range(90):
                date = now - datetime.timedelta(days=90 - day)

                # نوسان روزانه
                daily_change = np.random.normal(0, 1) * volatility
                # روند کلی
                trend_change = day * trend * base_price * 0.0005

                # محاسبه قیمت
                price = base_price + trend_change + daily_change
                price = max(price, base_price * 0.5)  # حداقل قیمت

                # اطلاعات کامل روز
                daily_info = {
                    "date": date.strftime("%Y-%m-%d"),
                    "timestamp": int(date.timestamp()),
                    "open": price,
                    "high": price * (1 + np.random.random() * 0.015),
                    "low": price * (1 - np.random.random() * 0.015),
                    "close": price * (1 + np.random.normal(0, 0.005)),
                    "volume": int(base_price * 1000 * (1 + np.random.random())),
                    "market_cap": int(price * base_price * 10000000)
                }

                daily_data.append(daily_info)

            price_data[symbol] = {
                "name": symbol,
                "data": daily_data,
                "last_updated": now.isoformat()
            }

        return price_data

    def _load_correlation_matrix(self) -> None:
        """بارگیری یا محاسبه ماتریس همبستگی ارزهای دیجیتال"""
        if self.correlation_matrix_loaded:
            return

        self.logger.info("در حال محاسبه ماتریس همبستگی ارزهای دیجیتال...")

        # اطمینان از بارگیری داده‌های قیمتی
        self._load_price_data()

        try:
            # محاسبه ماتریس همبستگی
            correlation = {}
            symbols = list(self._price_data.keys())

            window = self.model_config['market_correlation_window']

            for base_symbol in symbols:
                correlation[base_symbol] = {}
                base_prices = [day["close"] for day in self._price_data[base_symbol]["data"][-window:]]

                for target_symbol in symbols:
                    if base_symbol == target_symbol:
                        correlation[base_symbol][target_symbol] = 1.0
                        continue

                    target_prices = [day["close"] for day in self._price_data[target_symbol]["data"][-window:]]

                    # همبستگی پیرسون
                    try:
                        corr = np.corrcoef(base_prices, target_prices)[0, 1]
                        correlation[base_symbol][target_symbol] = float(corr)
                    except:
                        correlation[base_symbol][target_symbol] = 0.0

            self._correlation_matrix = correlation
            self.correlation_matrix_loaded = True
            self.logger.info("ماتریس همبستگی ارزهای دیجیتال با موفقیت محاسبه شد")

        except Exception as e:
            self.logger.error(f"خطا در محاسبه ماتریس همبستگی: {str(e)}")
            self._correlation_matrix = {}

    def _load_market_patterns(self) -> None:
        """بارگیری الگوهای بازار"""
        if self.market_patterns_loaded:
            return

        self.logger.info("در حال بارگیری الگوهای بازار...")

        # الگوهای شناخته شده بازار
        # در نسخه‌های آینده می‌تواند از دیتابیس بارگیری شود یا با یادگیری ماشین استخراج شود
        self._market_patterns = {
            "double_top": {
                "name": "دابل تاپ (قله دوگانه)",
                "description": "الگوی قله دوگانه که معمولاً نشانه معکوس شدن روند صعودی است",
                "confidence": 0.75,
                "trend_indication": "bearish",
                "timeframe": "mid_term"
            },
            "double_bottom": {
                "name": "دابل باتم (کف دوگانه)",
                "description": "الگوی کف دوگانه که معمولاً نشانه معکوس شدن روند نزولی است",
                "confidence": 0.75,
                "trend_indication": "bullish",
                "timeframe": "mid_term"
            },
            "head_and_shoulders": {
                "name": "سر و شانه",
                "description": "الگوی سر و شانه که معمولاً نشانه معکوس شدن روند صعودی است",
                "confidence": 0.8,
                "trend_indication": "bearish",
                "timeframe": "mid_term"
            },
            "inverse_head_and_shoulders": {
                "name": "سر و شانه معکوس",
                "description": "الگوی سر و شانه معکوس که معمولاً نشانه معکوس شدن روند نزولی است",
                "confidence": 0.8,
                "trend_indication": "bullish",
                "timeframe": "mid_term"
            },
            "ascending_triangle": {
                "name": "مثلث صعودی",
                "description": "الگوی مثلث صعودی که معمولاً نشانه ادامه روند صعودی است",
                "confidence": 0.7,
                "trend_indication": "bullish",
                "timeframe": "short_term"
            },
            "descending_triangle": {
                "name": "مثلث نزولی",
                "description": "الگوی مثلث نزولی که معمولاً نشانه ادامه روند نزولی است",
                "confidence": 0.7,
                "trend_indication": "bearish",
                "timeframe": "short_term"
            },
            "bullish_flag": {
                "name": "پرچم صعودی",
                "description": "الگوی پرچم صعودی که نشانه ادامه روند صعودی پس از استراحت کوتاه است",
                "confidence": 0.65,
                "trend_indication": "bullish",
                "timeframe": "short_term"
            },
            "bearish_flag": {
                "name": "پرچم نزولی",
                "description": "الگوی پرچم نزولی که نشانه ادامه روند نزولی پس از استراحت کوتاه است",
                "confidence": 0.65,
                "trend_indication": "bearish",
                "timeframe": "short_term"
            },
            "golden_cross": {
                "name": "گلدن کراس",
                "description": "تقاطع میانگین متحرک کوتاه‌مدت از بالای میانگین متحرک بلندمدت",
                "confidence": 0.75,
                "trend_indication": "bullish",
                "timeframe": "mid_term"
            },
            "death_cross": {
                "name": "دث کراس",
                "description": "تقاطع میانگین متحرک کوتاه‌مدت از پایین میانگین متحرک بلندمدت",
                "confidence": 0.75,
                "trend_indication": "bearish",
                "timeframe": "mid_term"
            }
        }

        self.market_patterns_loaded = True
        self.logger.info(f"{len(self._market_patterns)} الگوی بازار با موفقیت بارگیری شد")

    def _load_historical_impact(self) -> None:
        """بارگیری داده‌های تاریخی تأثیر اخبار بر بازار"""
        if self.historical_impact_loaded:
            return

        self.logger.info("در حال بارگیری داده‌های تاریخی تأثیر اخبار...")

        # در نسخه واقعی، این داده‌ها از دیتابیس بارگیری می‌شوند
        # برای نمونه، داده‌های پیش‌فرض ایجاد می‌کنیم

        self._historical_impact = {
            "regulation": {
                "impact_weight": 0.85,
                "avg_price_change": 0.12,
                "duration": 48,  # ساعت
                "volatility_increase": 0.20,
                "affected_coins": ["BTC", "ETH", "XRP", "BNB"],
                "sentiment_correlation": 0.8
            },
            "adoption": {
                "impact_weight": 0.80,
                "avg_price_change": 0.08,
                "duration": 72,  # ساعت
                "volatility_increase": 0.10,
                "affected_coins": ["BTC", "ETH"],
                "sentiment_correlation": 0.75
            },
            "security": {
                "impact_weight": 0.75,
                "avg_price_change": -0.15,
                "duration": 36,  # ساعت
                "volatility_increase": 0.30,
                "affected_coins": ["BTC", "ETH", "SOL", "BNB"],
                "sentiment_correlation": 0.85
            },
            "partnership": {
                "impact_weight": 0.65,
                "avg_price_change": 0.05,
                "duration": 24,  # ساعت
                "volatility_increase": 0.05,
                "affected_coins": ["specific"],
                "sentiment_correlation": 0.6
            },
            "listing": {
                "impact_weight": 0.60,
                "avg_price_change": 0.15,
                "duration": 12,  # ساعت
                "volatility_increase": 0.25,
                "affected_coins": ["specific"],
                "sentiment_correlation": 0.7
            },
            "price_movement": {
                "impact_weight": 0.55,
                "avg_price_change": 0.03,
                "duration": 6,  # ساعت
                "volatility_increase": 0.02,
                "affected_coins": ["BTC"],
                "sentiment_correlation": 0.5
            }
        }

        self.historical_impact_loaded = True
        self.logger.info(f"داده‌های تاریخی تأثیر اخبار بر {len(self._historical_impact)} دسته خبری بارگیری شد")

    @retry_on_error()
    def analyze_news_impact(self, news_content: str, news_title: Optional[str] = None,
                            source_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        تحلیل تأثیر یک خبر بر بازار ارزهای دیجیتال

        Args:
            news_content: متن خبر
            news_title: عنوان خبر (اختیاری)
            source_info: اطلاعات منبع خبر (اختیاری)

        Returns:
            دیکشنری نتایج تحلیل
        """
        return self.cached_call(
            "analyze_news_impact",
            f"{news_title or ''}|{news_content[:100]}",  # کلید کش
            lambda x: self._analyze_news_impact_impl(news_content, news_title, source_info)
        )

    def _analyze_news_impact_impl(self, news_content: str, news_title: Optional[str] = None,
                                  source_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی تحلیل تأثیر خبر بر بازار

        Args:
            news_content: متن خبر
            news_title: عنوان خبر (اختیاری)
            source_info: اطلاعات منبع خبر (اختیاری)

        Returns:
            دیکشنری نتایج تحلیل
        """
        # بررسی متن خالی
        if not news_content:
            return {
                "error": "متن خبر خالی است",
                "has_impact": False,
                "impact_level": "low",
                "affected_coins": [],
                "prediction": None
            }

        # اطمینان از بارگیری داده‌های پایه
        self.load_model()

        # ترکیب عنوان و متن برای تحلیل
        full_text = news_content
        if news_title:
            full_text = f"{news_title}\n{news_content}"

        # دریافت تحلیل مالی و استخراج ارزهای مذکور
        from .financial_model import financial
        finance_analysis = financial.analyze_complete_news(full_text)

        # دریافت تحلیل متنی
        from .parsbert_model import parsbert
        text_analysis = parsbert.analyze_news(full_text, news_title)

        # محاسبه ضریب تأثیر خبر
        impact_result = self._calculate_news_impact(
            finance_analysis,
            text_analysis,
            source_info
        )

        # محاسبه پیش‌بینی روند بازار
        trend_prediction = self._predict_market_trend(
            impact_result["affected_coins"],
            finance_analysis,
            impact_result["impact_level"]
        )

        # ترکیب نتایج
        result = {
            "has_impact": impact_result["has_impact"],
            "impact_level": impact_result["impact_level"],
            "impact_score": impact_result["impact_score"],
            "affected_coins": impact_result["affected_coins"],
            "related_coins": impact_result["related_coins"],
            "sentiment": finance_analysis["sentiment"],
            "category": finance_analysis["category"],
            "prediction": trend_prediction,
            "confidence": impact_result["confidence"],
            "analysis_time": datetime.datetime.now().isoformat()
        }

        # ذخیره تحلیل در حافظه روند
        self._store_trend_analysis(result)

        return result

    def _calculate_news_impact(self, finance_analysis: Dict[str, Any],
                               text_analysis: Dict[str, Any],
                               source_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        محاسبه ضریب تأثیر خبر بر بازار و ارزهای متأثر

        Args:
            finance_analysis: نتایج تحلیل مالی
            text_analysis: نتایج تحلیل متنی
            source_info: اطلاعات منبع خبر

        Returns:
            دیکشنری نتایج ضریب تأثیر
        """
        # دریافت وزن‌های محاسبه تأثیر
        weights = self.model_config['impact_calculation_weights']

        # 1. محاسبه تأثیر احساسات
        sentiment_score = 0.0
        sentiment_info = text_analysis["sentiment"]

        if sentiment_info["sentiment"] == "positive":
            sentiment_score = sentiment_info["positive"]
        elif sentiment_info["sentiment"] == "negative":
            sentiment_score = sentiment_info["negative"]
        else:
            sentiment_score = 0.3  # خنثی بودن تأثیر کمتری دارد

        # 2. محاسبه تأثیر اهمیت خبر
        importance_score = 0.0
        if "importance" in finance_analysis and "importance_score" in finance_analysis["importance"]:
            importance_score = finance_analysis["importance"]["importance_score"]
        else:
            # محاسبه تقریبی اهمیت
            keywords = text_analysis["keywords"]
            importance_score = min(1.0, len(keywords) * 0.05)

        # 3. محاسبه اختصاصی بودن خبر برای ارزها
        specificity_score = 0.0
        affected_coins = []
        related_coins = []

        # استخراج ارزهای مذکور در خبر
        if "cryptocurrencies" in finance_analysis and finance_analysis["cryptocurrencies"]["found"]:
            crypto_data = finance_analysis["cryptocurrencies"]["cryptocurrencies"]

            # افزودن اصلی‌ترین ارز به لیست ارزهای متأثر
            for crypto in crypto_data:
                symbol = crypto["symbol"]
                mentions = len(crypto["mentions"])
                importance = self._crypto_importance.get(symbol, 0.5)

                # ارزهای مهم‌تر و با تکرار بیشتر، تأثیر بیشتری می‌پذیرند
                if mentions >= 2 or importance >= 0.7:
                    affected_coins.append({
                        "symbol": symbol,
                        "impact_coefficient": min(1.0, importance * (1.0 + mentions * 0.1)),
                        "mentions": mentions
                    })
                else:
                    related_coins.append({
                        "symbol": symbol,
                        "impact_coefficient": importance * 0.5,
                        "mentions": mentions
                    })

            # محاسبه امتیاز اختصاصی بودن
            if affected_coins:
                # خبرهای مختص یک یا دو ارز، اختصاصی‌تر و تأثیرگذارتر هستند
                if len(affected_coins) <= 2:
                    specificity_score = 0.9
                elif len(affected_coins) <= 5:
                    specificity_score = 0.7
                else:
                    specificity_score = 0.5

            # اضافه کردن ارزهای مرتبط بر اساس همبستگی
            if affected_coins and not self.correlation_matrix_loaded:
                try:
                    self._load_correlation_matrix()
                except:
                    pass

            if affected_coins and self.correlation_matrix_loaded:
                primary_crypto = affected_coins[0]["symbol"]

                if primary_crypto in self._correlation_matrix:
                    correlations = self._correlation_matrix[primary_crypto]

                    for symbol, correlation in correlations.items():
                        # افزودن ارزهای دارای همبستگی بالا که در خبر ذکر نشده‌اند
                        if correlation > 0.7 and symbol not in [c["symbol"] for c in affected_coins] and symbol not in [
                            c["symbol"] for c in related_coins]:
                            related_coins.append({
                                "symbol": symbol,
                                "impact_coefficient": correlation * 0.5,
                                "mentions": 0,
                                "correlation": correlation
                            })

        # اگر هیچ ارزی در خبر ذکر نشده، بیت‌کوین را به عنوان پیش‌فرض در نظر می‌گیریم
        if not affected_coins and not related_coins:
            affected_coins.append({
                "symbol": "BTC",
                "impact_coefficient": 0.7,
                "mentions": 0
            })
            specificity_score = 0.4  # خبر عمومی است

        # مرتب‌سازی ارزها بر اساس ضریب تأثیر
        affected_coins.sort(key=lambda x: x["impact_coefficient"], reverse=True)
        related_coins.sort(key=lambda x: x["impact_coefficient"], reverse=True)

        # 4. محاسبه تأثیر اعتبار منبع
        source_score = 0.5  # پیش‌فرض

        if source_info and "source" in source_info:
            source = source_info["source"].lower()

            # جستجو در مقادیر اعتبار منبع
            if source in self._source_credibility:
                source_score = self._source_credibility[source]
            else:
                # تلاش برای تطبیق جزئی
                for key, value in self._source_credibility.items():
                    if key in source:
                        source_score = value
                        break

                # استفاده از مقادیر پیش‌فرض بر اساس نوع منبع
                if "telegram" in source:
                    source_score = self._source_credibility["default_telegram"]
                elif "twitter" in source:
                    source_score = self._source_credibility["default_twitter"]
                elif any(domain in source for domain in [".com", ".org", ".io", ".net"]):
                    source_score = self._source_credibility["default_website"]

        # 5. محاسبه امتیاز تأثیر کلی
        impact_score = (
                sentiment_score * weights["sentiment_weight"] +
                importance_score * weights["importance_weight"] +
                specificity_score * weights["specificity_weight"] +
                source_score * weights["source_weight"]
        )

        # تعیین سطح تأثیر
        impact_level = "low"
        if impact_score >= 0.8:
            impact_level = "very_high"
        elif impact_score >= 0.65:
            impact_level = "high"
        elif impact_score >= 0.5:
            impact_level = "moderate"
        elif impact_score >= 0.3:
            impact_level = "low"
        else:
            impact_level = "very_low"

        # محاسبه اطمینان تحلیل
        confidence = min(1.0, (impact_score + importance_score + source_score) / 3)

        # تشخیص آیا خبر تأثیر قابل توجهی دارد
        has_impact = impact_score >= self.model_config['default_confidence_threshold']

        return {
            "has_impact": has_impact,
            "impact_level": impact_level,
            "impact_score": round(float(impact_score), 2),
            "affected_coins": affected_coins,
            "related_coins": related_coins,
            "sentiment_score": round(float(sentiment_score), 2),
            "importance_score": round(float(importance_score), 2),
            "specificity_score": round(float(specificity_score), 2),
            "source_score": round(float(source_score), 2),
            "confidence": round(float(confidence), 2)
        }

    def _predict_market_trend(self, affected_coins: List[Dict[str, Any]],
                              finance_analysis: Dict[str, Any],
                              impact_level: str) -> Dict[str, Any]:
        """
        پیش‌بینی روند بازار بر اساس تحلیل خبر

        Args:
            affected_coins: لیست ارزهای متأثر از خبر
            finance_analysis: نتایج تحلیل مالی
            impact_level: سطح تأثیر خبر

        Returns:
            دیکشنری پیش‌بینی روند
        """
        if not affected_coins:
            return None

        # بارگیری داده‌های تاریخی تأثیر
        if not self.historical_impact_loaded:
            self._load_historical_impact()

        # دریافت اطلاعات دسته‌بندی خبر
        category = "general"
        if "category" in finance_analysis and "category" in finance_analysis["category"]:
            category = finance_analysis["category"]["category"].lower()

        # مپینگ دسته‌بندی به دسته‌های تاریخی
        category_mapping = {
            "price": "price_movement",
            "regulation": "regulation",
            "adoption": "adoption",
            "technology": "adoption",
            "security": "security",
            "exchange": "listing",
            "general": "price_movement"
        }

        historic_category = category_mapping.get(category, "price_movement")

        # دریافت اطلاعات تاریخی
        historic_data = self._historical_impact.get(historic_category, {
            "impact_weight": 0.5,
            "avg_price_change": 0.05,
            "duration": 24,
            "volatility_increase": 0.1,
            "sentiment_correlation": 0.6
        })

        # دریافت اطلاعات احساسات
        sentiment = "neutral"
        if "sentiment" in finance_analysis and "sentiment" in finance_analysis["sentiment"]:
            sentiment = finance_analysis["sentiment"]["sentiment"]

        # تعیین جهت احتمالی حرکت قیمت بر اساس احساسات
        price_direction = 0.0
        if sentiment == "positive":
            price_direction = 1.0
        elif sentiment == "negative":
            price_direction = -1.0

        # محاسبه ضریب تأثیر بر اساس سطح تأثیر
        impact_multiplier = {
            "very_high": 1.5,
            "high": 1.2,
            "moderate": 1.0,
            "low": 0.7,
            "very_low": 0.5
        }.get(impact_level, 1.0)

        # محاسبه درصد تغییر احتمالی
        estimated_change = historic_data["avg_price_change"] * price_direction * impact_multiplier

        # محاسبه مدت تأثیر
        impact_duration = int(historic_data["duration"] * impact_multiplier)

        # تعیین روند بازار
        trend = "neutral"
        if estimated_change > 0.1:
            trend = "strong_bullish"
        elif estimated_change > 0.05:
            trend = "bullish"
        elif estimated_change > 0.02:
            trend = "weak_bullish"
        elif estimated_change < -0.1:
            trend = "strong_bearish"
        elif estimated_change < -0.05:
            trend = "bearish"
        elif estimated_change < -0.02:
            trend = "weak_bearish"

        # تعیین بازه زمانی
        timeframe = "immediate"
        if impact_duration > 72:
            timeframe = "mid_term"
        elif impact_duration > 24:
            timeframe = "short_term"

        # تعیین سطح اطمینان
        confidence_level = "moderate"
        confidence_score = historic_data["sentiment_correlation"] * impact_multiplier

        if confidence_score > 0.85:
            confidence_level = "very_high"
        elif confidence_score > 0.7:
            confidence_level = "high"
        elif confidence_score > 0.5:
            confidence_level = "moderate"
        elif confidence_score > 0.3:
            confidence_level = "low"
        else:
            confidence_level = "very_low"

        # ایجاد نتیجه پیش‌بینی
        prediction = {
            "trend": trend,
            "trend_fa": self.MARKET_TRENDS[trend],
            "estimated_change": round(float(estimated_change * 100), 2),  # تبدیل به درصد
            "timeframe": timeframe,
            "timeframe_fa": self.TIME_FRAMES[timeframe],
            "confidence": confidence_level,
            "confidence_fa": self.CONFIDENCE_LEVELS[confidence_level],
            "confidence_score": round(float(confidence_score), 2),
            "impact_duration_hours": impact_duration,
            "volatility_increase": round(float(historic_data["volatility_increase"] * impact_multiplier), 2)
        }

        return prediction

    def _store_trend_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """
        ذخیره تحلیل در حافظه روند برای یادگیری و بهبود پیش‌بینی‌های آینده

        Args:
            analysis_result: نتیجه تحلیل
        """
        # ذخیره تحلیل برای هر ارز متأثر
        for coin in analysis_result["affected_coins"]:
            symbol = coin["symbol"]

            # افزودن به حافظه روند
            trend_data = {
                "timestamp": time.time(),
                "impact_score": analysis_result["impact_score"],
                "impact_level": analysis_result["impact_level"],
                "trend": analysis_result["prediction"]["trend"] if analysis_result["prediction"] else "neutral",
                "sentiment": analysis_result["sentiment"]["sentiment"],
                "category": analysis_result["category"]["category"],
                "confidence": analysis_result["confidence"]
            }

            self._trend_memory[symbol].append(trend_data)

    @retry_on_error()
    def analyze_news_batch(self, news_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تحلیل دسته‌ای چندین خبر و ارائه تحلیل ترکیبی

        Args:
            news_batch: لیست خبرها (هر خبر دارای 'content', 'title', 'source')

        Returns:
            دیکشنری نتایج تحلیل ترکیبی
        """
        # بررسی ورودی خالی
        if not news_batch:
            return {
                "error": "لیست خبرها خالی است",
                "has_impact": False
            }

        # تحلیل تک تک خبرها
        news_analyses = []
        for news in news_batch:
            content = news.get("content", "")
            title = news.get("title", "")
            source = news.get("source", {})

            if content:
                analysis = self.analyze_news_impact(content, title, source)
                if not "error" in analysis:
                    news_analyses.append(analysis)

        # بررسی تعداد تحلیل‌ها
        if len(news_analyses) < self.model_config['min_news_for_analysis']:
            return {
                "error": "تعداد خبرهای معتبر برای تحلیل کافی نیست",
                "has_impact": False,
                "individual_analyses": news_analyses
            }

        # ترکیب تحلیل‌ها
        combined_result = self._combine_news_analyses(news_analyses)

        # ایجاد نتیجه نهایی
        result = {
            "has_impact": combined_result["has_impact"],
            "impact_level": combined_result["impact_level"],
            "impact_score": combined_result["impact_score"],
            "affected_coins": combined_result["affected_coins"],
            "prediction": combined_result["prediction"],
            "sentiment_summary": combined_result["sentiment_summary"],
            "confidence": combined_result["confidence"],
            "news_count": len(news_analyses),
            "analysis_time": datetime.datetime.now().isoformat(),
            "individual_analyses": news_analyses
        }

        return result

    def _combine_news_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ترکیب نتایج تحلیل چندین خبر

        Args:
            analyses: لیست نتایج تحلیل خبرها

        Returns:
            دیکشنری نتیجه ترکیبی
        """
        # جمع‌آوری ارزهای متأثر از تمام خبرها
        all_affected_coins = defaultdict(list)
        for analysis in analyses:
            for coin in analysis.get("affected_coins", []):
                all_affected_coins[coin["symbol"]].append({
                    "impact_coefficient": coin["impact_coefficient"],
                    "analysis_impact": analysis["impact_score"]
                })

        # محاسبه میانگین وزنی ضریب تأثیر برای هر ارز
        combined_affected_coins = []
        for symbol, impacts in all_affected_coins.items():
            avg_impact = np.mean([impact["impact_coefficient"] * impact["analysis_impact"] for impact in impacts])
            mentions = len(impacts)

            combined_affected_coins.append({
                "symbol": symbol,
                "impact_coefficient": round(float(avg_impact), 2),
                "mentions": mentions,
                "news_count": len(impacts)
            })

        # مرتب‌سازی ارزها بر اساس ضریب تأثیر
        combined_affected_coins.sort(key=lambda x: x["impact_coefficient"], reverse=True)

        # محاسبه میانگین امتیاز تأثیر
        impact_scores = [analysis.get("impact_score", 0) for analysis in analyses]
        avg_impact_score = round(float(np.mean(impact_scores)), 2)

        # تعیین سطح تأثیر ترکیبی
        impact_level = "low"
        if avg_impact_score >= 0.8:
            impact_level = "very_high"
        elif avg_impact_score >= 0.65:
            impact_level = "high"
        elif avg_impact_score >= 0.5:
            impact_level = "moderate"
        elif avg_impact_score >= 0.3:
            impact_level = "low"
        else:
            impact_level = "very_low"

        # خلاصه احساسات
        sentiments = [analysis.get("sentiment", {}).get("sentiment", "neutral") for analysis in analyses]
        sentiment_counts = {
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral")
        }

        # تعیین احساس غالب
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        sentiment_ratio = sentiment_counts[dominant_sentiment] / len(sentiments)

        # ترکیب پیش‌بینی‌ها
        trends = [analysis.get("prediction", {}).get("trend", "neutral") for analysis in analyses if
                  analysis.get("prediction")]

        # شمارش روندها
        trend_counts = {}
        for trend in trends:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        # تعیین روند غالب
        dominant_trend = max(trend_counts.items(), key=lambda x: x[1])[0] if trend_counts else "neutral"
        trend_ratio = trend_counts.get(dominant_trend, 0) / len(trends) if trends else 0

        # محاسبه تغییر قیمت احتمالی
        estimated_changes = [analysis.get("prediction", {}).get("estimated_change", 0) for analysis in analyses if
                             analysis.get("prediction")]
        avg_estimated_change = round(float(np.mean(estimated_changes)) if estimated_changes else 0, 2)

        # تعیین اطمینان ترکیبی
        confidence_scores = [analysis.get("confidence", 0) for analysis in analyses]
        avg_confidence = round(float(np.mean(confidence_scores)) if confidence_scores else 0, 2)

        # تعیین سطح اطمینان
        confidence_level = "moderate"
        if avg_confidence > 0.85:
            confidence_level = "very_high"
        elif avg_confidence > 0.7:
            confidence_level = "high"
        elif avg_confidence > 0.5:
            confidence_level = "moderate"
        elif avg_confidence > 0.3:
            confidence_level = "low"
        else:
            confidence_level = "very_low"

        # ایجاد پیش‌بینی ترکیبی
        combined_prediction = {
            "trend": dominant_trend,
            "trend_fa": self.MARKET_TRENDS.get(dominant_trend, "خنثی"),
            "trend_ratio": round(float(trend_ratio), 2),
            "estimated_change": avg_estimated_change,
            "timeframe": "short_term" if avg_impact_score > 0.6 else "immediate",
            "timeframe_fa": self.TIME_FRAMES["short_term"] if avg_impact_score > 0.6 else self.TIME_FRAMES["immediate"],
            "confidence": confidence_level,
            "confidence_fa": self.CONFIDENCE_LEVELS[confidence_level],
            "confidence_score": avg_confidence
        }

        # ایجاد نتیجه ترکیبی
        combined_result = {
            "has_impact": avg_impact_score >= self.model_config['default_confidence_threshold'],
            "impact_level": impact_level,
            "impact_score": avg_impact_score,
            "affected_coins": combined_affected_coins,
            "prediction": combined_prediction,
            "sentiment_summary": {
                "dominant_sentiment": dominant_sentiment,
                "sentiment_ratio": round(float(sentiment_ratio), 2),
                "sentiment_counts": sentiment_counts
            },
            "confidence": avg_confidence
        }

        return combined_result

    @retry_on_error()
    def generate_market_report(self, symbol: str, time_window: int = None) -> Dict[str, Any]:
        """
        تولید گزارش تحلیلی از وضعیت بازار یک ارز

        Args:
            symbol: نماد ارز
            time_window: پنجره زمانی به روز (اختیاری)

        Returns:
            دیکشنری گزارش تحلیلی
        """
        # تنظیم پنجره زمانی
        if time_window is None:
            time_window = self.model_config['default_time_window']

        # اطمینان از بارگیری داده‌های قیمتی
        self._load_price_data()

        # بررسی وجود داده‌های ارز
        if symbol not in self._price_data:
            return {
                "error": f"داده‌های قیمتی برای ارز {symbol} یافت نشد",
                "symbol": symbol,
                "has_data": False
            }

        # استخراج داده‌های قیمتی در بازه زمانی مورد نظر
        price_data = self._price_data[symbol]["data"]
        recent_data = price_data[-min(time_window, len(price_data)):]

        # محاسبه روند قیمت
        price_trend = self._calculate_price_trend(recent_data)

        # تشخیص الگوهای تکنیکال
        technical_patterns = self._detect_technical_patterns(recent_data)

        # تحلیل نوسانات
        volatility_analysis = self._analyze_volatility(recent_data)

        # جمع‌آوری تحلیل‌های اخبار مرتبط
        news_analysis = self._get_recent_news_analysis(symbol, time_window)

        # محاسبه روند پیش‌بینی شده
        predicted_trend = self._calculate_predicted_trend(symbol, news_analysis, price_trend)

        # محاسبه سطوح حمایت/مقاومت
        support_resistance = self._calculate_support_resistance(recent_data)

        # ایجاد گزارش نهایی
        report = {
            "symbol": symbol,
            "has_data": True,
            "time_window": time_window,
            "analysis_time": datetime.datetime.now().isoformat(),
            "current_price": recent_data[-1]["close"],
            "price_change_24h": self._calculate_price_change(recent_data, 1),
            "price_change_7d": self._calculate_price_change(recent_data, 7),
            "price_change_30d": self._calculate_price_change(recent_data, 30),
            "price_trend": price_trend,
            "predicted_trend": predicted_trend,
            "technical_patterns": technical_patterns,
            "volatility_analysis": volatility_analysis,
            "support_resistance": support_resistance,
            "news_impact": news_analysis["summary"] if news_analysis else None,
            "news_count": news_analysis["count"] if news_analysis else 0
        }

        return report

    def _calculate_price_trend(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        محاسبه روند قیمت

        Args:
            price_data: داده‌های قیمتی

        Returns:
            دیکشنری روند قیمت
        """
        if not price_data or len(price_data) < 2:
            return {
                "trend": "neutral",
                "trend_fa": self.MARKET_TRENDS["neutral"],
                "strength": 0.0
            }

        # محاسبه میانگین‌های متحرک
        closes = [day["close"] for day in price_data]

        ma7 = np.mean(closes[-7:]) if len(closes) >= 7 else np.mean(closes)
        ma25 = np.mean(closes[-25:]) if len(closes) >= 25 else ma7

        # قیمت فعلی
        current_price = closes[-1]

        # محاسبه تغییرات
        short_term_change = (current_price / closes[-min(7, len(closes))] - 1) if len(closes) >= 2 else 0
        long_term_change = (current_price / closes[0] - 1) if len(closes) >= 2 else 0

        # تعیین روند
        trend = "neutral"
        strength = 0.0

        # بررسی روند بر اساس میانگین‌های متحرک و تغییرات قیمت
        if current_price > ma7 and ma7 > ma25:
            # روند صعودی
            if short_term_change > 0.1 and long_term_change > 0.2:
                trend = "strong_bullish"
                strength = 0.9
            elif short_term_change > 0.05 or long_term_change > 0.1:
                trend = "bullish"
                strength = 0.7
            else:
                trend = "weak_bullish"
                strength = 0.5
        elif current_price < ma7 and ma7 < ma25:
            # روند نزولی
            if short_term_change < -0.1 and long_term_change < -0.2:
                trend = "strong_bearish"
                strength = 0.9
            elif short_term_change < -0.05 or long_term_change < -0.1:
                trend = "bearish"
                strength = 0.7
            else:
                trend = "weak_bearish"
                strength = 0.5
        else:
            # روند خنثی/رنج
            if abs(short_term_change) > 0.05:
                trend = "volatile"
                strength = 0.6
            else:
                trend = "neutral"
                strength = 0.4

        return {
            "trend": trend,
            "trend_fa": self.MARKET_TRENDS[trend],
            "strength": round(float(strength), 2),
            "short_term_change": round(float(short_term_change * 100), 2),  # تبدیل به درصد
            "long_term_change": round(float(long_term_change * 100), 2),  # تبدیل به درصد
            "current_price": round(float(current_price), 2),
            "ma7": round(float(ma7), 2),
            "ma25": round(float(ma25), 2)
        }

    def _calculate_price_change(self, price_data: List[Dict[str, Any]], days: int) -> float:
        """
        محاسبه درصد تغییر قیمت در بازه زمانی مشخص

        Args:
            price_data: داده‌های قیمتی
            days: تعداد روز

        Returns:
            درصد تغییر قیمت
        """
        if not price_data or len(price_data) < 2:
            return 0.0

        current_price = price_data[-1]["close"]

        if len(price_data) <= days:
            old_price = price_data[0]["close"]
        else:
            old_price = price_data[-days - 1]["close"]

        price_change = ((current_price / old_price) - 1) * 100
        return round(float(price_change), 2)

    def _detect_technical_patterns(self, price_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        تشخیص الگوهای تکنیکال در نمودار قیمت

        Args:
            price_data: داده‌های قیمتی

        Returns:
            لیست الگوهای تشخیص داده شده
        """
        # اطمینان از بارگیری الگوهای بازار
        self._load_market_patterns()

        # برای پیاده‌سازی کامل، الگوریتم‌های تشخیص الگو مورد نیاز است
        # در اینجا یک نمونه ساده پیاده‌سازی می‌کنیم

        # داده‌های قیمتی نیاز به حداقل تعداد روز دارند
        if not price_data or len(price_data) < 10:
            return []

        detected_patterns = []

        # محاسبه میانگین‌های متحرک
        closes = np.array([day["close"] for day in price_data])
        highs = np.array([day["high"] for day in price_data])
        lows = np.array([day["low"] for day in price_data])

        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
        ma200 = np.mean(closes[-200:]) if len(closes) >= 200 else ma50

        # تشخیص گلدن کراس / دث کراس (تقاطع میانگین‌های متحرک)
        if len(closes) >= 50:
            # محاسبه میانگین‌های متحرک برای دو روز آخر
            ma50_now = np.mean(closes[-50:])
            ma50_prev = np.mean(closes[-51:-1])
            ma200_now = np.mean(closes[-200:]) if len(closes) >= 200 else np.mean(closes)
            ma200_prev = np.mean(closes[-201:-1]) if len(closes) >= 201 else np.mean(closes[:-1])

            # تشخیص گلدن کراس
            if ma50_prev < ma200_prev and ma50_now > ma200_now:
                detected_patterns.append({
                    "pattern": "golden_cross",
                    "name": self._market_patterns["golden_cross"]["name"],
                    "description": self._market_patterns["golden_cross"]["description"],
                    "confidence": self._market_patterns["golden_cross"]["confidence"],
                    "trend_indication": self._market_patterns["golden_cross"]["trend_indication"]
                })

            # تشخیص دث کراس
            elif ma50_prev > ma200_prev and ma50_now < ma200_now:
                detected_patterns.append({
                    "pattern": "death_cross",
                    "name": self._market_patterns["death_cross"]["name"],
                    "description": self._market_patterns["death_cross"]["description"],
                    "confidence": self._market_patterns["death_cross"]["confidence"],
                    "trend_indication": self._market_patterns["death_cross"]["trend_indication"]
                })

        # تشخیص ساده دابل تاپ / دابل باتم
        if len(highs) >= 20:
            # شناسایی قله‌ها و دره‌ها
            is_peak = np.logical_and(
                np.concatenate(([False], highs[1:-1] > highs[:-2])),
                np.concatenate((highs[1:-1] > highs[2:], [False]))
            )
            peak_indices = np.where(is_peak)[0]

            is_valley = np.logical_and(
                np.concatenate(([False], lows[1:-1] < lows[:-2])),
                np.concatenate((lows[1:-1] < lows[2:], [False]))
            )
            valley_indices = np.where(is_valley)[0]

            # تشخیص دابل تاپ - دو قله مشابه با فاصله کم
            if len(peak_indices) >= 2:
                peaks = peak_indices[-2:]
                if 3 <= peaks[1] - peaks[0] <= 10:
                    peak_prices = highs[peaks]
                    if abs(peak_prices[1] - peak_prices[0]) / peak_prices[0] < 0.03:
                        detected_patterns.append({
                            "pattern": "double_top",
                            "name": self._market_patterns["double_top"]["name"],
                            "description": self._market_patterns["double_top"]["description"],
                            "confidence": self._market_patterns["double_top"]["confidence"],
                            "trend_indication": self._market_patterns["double_top"]["trend_indication"]
                        })

            # تشخیص دابل باتم - دو دره مشابه با فاصله کم
            if len(valley_indices) >= 2:
                valleys = valley_indices[-2:]
                if 3 <= valleys[1] - valleys[0] <= 10:
                    valley_prices = lows[valleys]
                    if abs(valley_prices[1] - valley_prices[0]) / valley_prices[0] < 0.03:
                        detected_patterns.append({
                            "pattern": "double_bottom",
                            "name": self._market_patterns["double_bottom"]["name"],
                            "description": self._market_patterns["double_bottom"]["description"],
                            "confidence": self._market_patterns["double_bottom"]["confidence"],
                            "trend_indication": self._market_patterns["double_bottom"]["trend_indication"]
                        })

        return detected_patterns

    def _analyze_volatility(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تحلیل نوسانات قیمت

        Args:
            price_data: داده‌های قیمتی

        Returns:
            دیکشنری تحلیل نوسانات
        """
        if not price_data or len(price_data) < 2:
            return {
                "volatility": 0.0,
                "is_volatile": False
            }

        # محاسبه نوسانات روزانه
        closes = np.array([day["close"] for day in price_data])
        daily_returns = np.diff(closes) / closes[:-1]

        # محاسبه انحراف معیار
        volatility = np.std(daily_returns)

        # محاسبه دامنه نوسان
        highs = np.array([day["high"] for day in price_data])
        lows = np.array([day["low"] for day in price_data])

        true_ranges = []
        for i in range(1, len(price_data)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range / prev_close)  # نسبی کردن

        avg_true_range = np.mean(true_ranges) if true_ranges else 0

        # محاسبه میانگین متحرک 14 روزه
        atr_14 = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else avg_true_range

        # تعیین آیا بازار پرنوسان است
        is_volatile = volatility > self.model_config['volatility_threshold'] or atr_14 > self.model_config[
            'volatility_threshold']

        return {
            "volatility": round(float(volatility * 100), 2),  # تبدیل به درصد
            "is_volatile": is_volatile,
            "avg_true_range": round(float(avg_true_range * 100), 2),  # تبدیل به درصد
            "atr_14": round(float(atr_14 * 100), 2),  # تبدیل به درصد
            "daily_range_percent": round(float(np.mean((highs - lows) / lows) * 100), 2)  # میانگین دامنه روزانه
        }

    def _get_recent_news_analysis(self, symbol: str, time_window: int) -> Optional[Dict[str, Any]]:
        """
        دریافت خلاصه تحلیل‌های اخبار اخیر برای یک ارز

        Args:
            symbol: نماد ارز
            time_window: پنجره زمانی (روز)

        Returns:
            دیکشنری خلاصه تحلیل‌ها یا None
        """
        # استفاده از حافظه روند برای دریافت تحلیل‌های اخیر
        if symbol not in self._trend_memory or not self._trend_memory[symbol]:
            return None

        # محاسبه زمان آستانه
        threshold_time = time.time() - (time_window * 24 * 3600)

        # فیلتر کردن تحلیل‌های در بازه زمانی
        recent_analyses = [
            analysis for analysis in self._trend_memory[symbol]
            if analysis["timestamp"] >= threshold_time
        ]

        if not recent_analyses:
            return None

        # محاسبه میانگین امتیاز تأثیر
        avg_impact_score = np.mean([analysis["impact_score"] for analysis in recent_analyses])

        # شمارش احساسات
        sentiment_counts = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        for analysis in recent_analyses:
            sentiment = analysis["sentiment"]
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # تعیین احساس غالب
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]

        # شمارش روندها
        trend_counts = {}
        for analysis in recent_analyses:
            trend = analysis["trend"]
            trend_counts[trend] = trend_counts.get(trend, 0) + 1

        # تعیین روند غالب
        dominant_trend = max(trend_counts.items(), key=lambda x: x[1])[0] if trend_counts else "neutral"

        # ایجاد خلاصه
        summary = {
            "count": len(recent_analyses),
            "avg_impact_score": round(float(avg_impact_score), 2),
            "dominant_sentiment": dominant_sentiment,
            "sentiment_counts": sentiment_counts,
            "dominant_trend": dominant_trend,
            "trend_counts": trend_counts
        }

        return summary

    def _calculate_predicted_trend(self, symbol: str, news_analysis: Optional[Dict[str, Any]],
                                   price_trend: Dict[str, Any]) -> Dict[str, Any]:
        """
        محاسبه روند پیش‌بینی شده بر اساس تحلیل‌های اخبار و روند قیمت

        Args:
            symbol: نماد ارز
            news_analysis: نتایج تحلیل اخبار
            price_trend: روند فعلی قیمت

        Returns:
            دیکشنری روند پیش‌بینی شده
        """
        if not news_analysis:
            # اگر تحلیل خبری نداشته باشیم، فقط بر اساس روند تکنیکال پیش‌بینی می‌کنیم
            return {
                "trend": price_trend["trend"],
                "trend_fa": price_trend["trend_fa"],
                "confidence": "moderate",
                "confidence_fa": self.CONFIDENCE_LEVELS["moderate"],
                "confidence_score": price_trend["strength"],
                "timeframe": "short_term",
                "timeframe_fa": self.TIME_FRAMES["short_term"],
                "based_on": "technical",
                "description": "این پیش‌بینی فقط بر اساس تحلیل تکنیکال است و اخبار اخیر در آن لحاظ نشده است."
            }

        # ترکیب روند تکنیکال و روند خبری
        technical_trend = price_trend["trend"]
        news_trend = news_analysis["dominant_trend"]

        # اگر روندها همسو باشند، اطمینان بیشتر است
        if self._are_trends_aligned(technical_trend, news_trend):
            confidence = "high"
            confidence_score = 0.8
            trend = technical_trend  # استفاده از روند تکنیکال
            description = "روندهای تکنیکال و خبری همسو هستند، بنابراین اطمینان تحلیل بالاست."
        else:
            # وزن‌دهی به روندها بر اساس قدرت تکنیکال و تعداد اخبار
            technical_weight = price_trend["strength"]
            news_weight = min(0.8, news_analysis["avg_impact_score"] * (news_analysis["count"] / 5))

            # انتخاب روند غالب
            if technical_weight > news_weight:
                trend = technical_trend
                confidence = "moderate"
                confidence_score = technical_weight
                description = "روند تکنیکال قوی‌تر از تأثیر اخبار اخیر است."
            else:
                trend = news_trend
                confidence = "moderate"
                confidence_score = news_weight
                description = "تأثیر اخبار اخیر بر روند قوی‌تر از الگوهای تکنیکال است."

        # تعیین بازه زمانی
        if confidence == "high":
            timeframe = "mid_term"
        else:
            timeframe = "short_term"

        # افزودن توضیحات بیشتر بر اساس روند
        if trend.endswith("bullish"):
            description += " احتمال رشد قیمت وجود دارد."
        elif trend.endswith("bearish"):
            description += " احتمال کاهش قیمت وجود دارد."
        elif trend == "volatile":
            description += " احتمال نوسانات شدید قیمت وجود دارد."
        else:
            description += " قیمت احتمالاً در محدوده فعلی باقی خواهد ماند."

        return {
            "trend": trend,
            "trend_fa": self.MARKET_TRENDS[trend],
            "confidence": confidence,
            "confidence_fa": self.CONFIDENCE_LEVELS[confidence],
            "confidence_score": round(float(confidence_score), 2),
            "timeframe": timeframe,
            "timeframe_fa": self.TIME_FRAMES[timeframe],
            "based_on": "technical_and_news",
            "description": description
        }

    def _are_trends_aligned(self, trend1: str, trend2: str) -> bool:
        """
        بررسی همسو بودن دو روند

        Args:
            trend1: روند اول
            trend2: روند دوم

        Returns:
            True اگر روندها همسو باشند، در غیر این صورت False
        """
        bullish_trends = ["strong_bullish", "bullish", "weak_bullish"]
        bearish_trends = ["strong_bearish", "bearish", "weak_bearish"]
        neutral_trends = ["neutral"]
        volatile_trends = ["volatile"]

        # بررسی همسو بودن
        if trend1 in bullish_trends and trend2 in bullish_trends:
            return True
        elif trend1 in bearish_trends and trend2 in bearish_trends:
            return True
        elif trend1 in neutral_trends and trend2 in neutral_trends:
            return True
        elif trend1 in volatile_trends and trend2 in volatile_trends:
            return True

        return False

    def _calculate_support_resistance(self, price_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        محاسبه سطوح حمایت و مقاومت

        Args:
            price_data: داده‌های قیمتی

        Returns:
            دیکشنری سطوح حمایت و مقاومت
        """
        if not price_data or len(price_data) < 10:
            return {
                "support_levels": [],
                "resistance_levels": []
            }

        # استخراج داده‌های قیمتی
        highs = np.array([day["high"] for day in price_data])
        lows = np.array([day["low"] for day in price_data])
        closes = np.array([day["close"] for day in price_data])

        # قیمت فعلی
        current_price = closes[-1]

        # یافتن قله‌ها و دره‌ها برای تعیین سطوح مقاومت و حمایت
        peak_indices = self._find_peaks(highs)
        valley_indices = self._find_valleys(lows)

        # استخراج سطوح مقاومت (بالاتر از قیمت فعلی)
        resistance_levels = []
        for idx in peak_indices:
            level = highs[idx]
            if level > current_price:
                strength = self._calculate_level_strength(price_data, level, "resistance")
                resistance_levels.append({
                    "price": round(float(level), 2),
                    "distance": round(float(((level / current_price) - 1) * 100), 2),  # فاصله به درصد
                    "strength": strength
                })

        # استخراج سطوح حمایت (پایین‌تر از قیمت فعلی)
        support_levels = []
        for idx in valley_indices:
            level = lows[idx]
            if level < current_price:
                strength = self._calculate_level_strength(price_data, level, "support")
                support_levels.append({
                    "price": round(float(level), 2),
                    "distance": round(float(((current_price / level) - 1) * 100), 2),  # فاصله به درصد
                    "strength": strength
                })

        # محدود کردن تعداد سطوح به 3 سطح قوی‌تر
        resistance_levels.sort(key=lambda x: (x["strength"], -x["distance"]), reverse=True)
        support_levels.sort(key=lambda x: (x["strength"], -x["distance"]), reverse=True)

        resistance_levels = resistance_levels[:3] if len(resistance_levels) > 3 else resistance_levels
        support_levels = support_levels[:3] if len(support_levels) > 3 else support_levels

        # مرتب‌سازی بر اساس فاصله از قیمت فعلی
        resistance_levels.sort(key=lambda x: x["distance"])
        support_levels.sort(key=lambda x: x["distance"])

        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "current_price": round(float(current_price), 2)
        }

    def _find_peaks(self, values: np.ndarray, min_distance: int = 5) -> List[int]:
        """
        یافتن قله‌ها در یک سری زمانی

        Args:
            values: آرایه مقادیر
            min_distance: حداقل فاصله بین قله‌ها

        Returns:
            لیست ایندکس‌های قله‌ها
        """
        if len(values) < 3:
            return []

        # شناسایی قله‌ها
        is_peak = np.logical_and(
            np.concatenate(([False], values[1:-1] >= values[:-2])),
            np.concatenate((values[1:-1] >= values[2:], [False]))
        )

        peak_indices = np.where(is_peak)[0]

        # حذف قله‌های خیلی نزدیک به هم
        filtered_peaks = []

        for i, peak in enumerate(peak_indices):
            if i == 0 or peak - peak_indices[i - 1] >= min_distance:
                filtered_peaks.append(peak)

        return filtered_peaks

    def _find_valleys(self, values: np.ndarray, min_distance: int = 5) -> List[int]:
        """
        یافتن دره‌ها در یک سری زمانی

        Args:
            values: آرایه مقادیر
            min_distance: حداقل فاصله بین دره‌ها

        Returns:
            لیست ایندکس‌های دره‌ها
        """
        if len(values) < 3:
            return []

        # شناسایی دره‌ها
        is_valley = np.logical_and(
            np.concatenate(([False], values[1:-1] <= values[:-2])),
            np.concatenate((values[1:-1] <= values[2:], [False]))
        )

        valley_indices = np.where(is_valley)[0]

        # حذف دره‌های خیلی نزدیک به هم
        filtered_valleys = []

        for i, valley in enumerate(valley_indices):
            if i == 0 or valley - valley_indices[i - 1] >= min_distance:
                filtered_valleys.append(valley)

        return filtered_valleys

    def _calculate_level_strength(self, price_data: List[Dict[str, Any]], level: float, level_type: str) -> float:
        """
        محاسبه قدرت یک سطح حمایت یا مقاومت

        Args:
            price_data: داده‌های قیمتی
            level: سطح مورد نظر
            level_type: نوع سطح ('support' یا 'resistance')

        Returns:
            قدرت سطح (0 تا 1)
        """
        # تعداد برخوردهای قیمت با این سطح
        touches = 0

        # محدوده سطح (با در نظر گرفتن 0.5% محدوده)
        level_min = level * 0.995
        level_max = level * 1.005

        # بررسی برخوردهای قیمت
        for day in price_data:
            high = day["high"]
            low = day["low"]

            if level_type == "resistance" and low <= level_max and high >= level_min:
                touches += 1
            elif level_type == "support" and high >= level_min and low <= level_max:
                touches += 1

        # محاسبه قدرت بر اساس تعداد برخوردها
        strength = min(1.0, 0.3 + (touches * 0.15))

        return round(float(strength), 2)

    def analyze_market_risk(self, symbol: str, time_window: int = None) -> Dict[str, Any]:
        """
        تحلیل ریسک بازار برای یک ارز

        Args:
            symbol: نماد ارز
            time_window: پنجره زمانی به روز (اختیاری)

        Returns:
            دیکشنری تحلیل ریسک
        """
        # تنظیم پنجره زمانی
        if time_window is None:
            time_window = self.model_config['default_time_window']

        # اطمینان از بارگیری داده‌های قیمتی
        self._load_price_data()

        # بررسی وجود داده‌های ارز
        if symbol not in self._price_data:
            return {
                "error": f"داده‌های قیمتی برای ارز {symbol} یافت نشد",
                "symbol": symbol,
                "has_data": False
            }

        # استخراج داده‌های قیمتی در بازه زمانی مورد نظر
        price_data = self._price_data[symbol]["data"]
        recent_data = price_data[-min(time_window, len(price_data)):]

        # محاسبه نوسانات
        volatility_analysis = self._analyze_volatility(recent_data)

        # محاسبه شاخص‌های ریسک
        closes = np.array([day["close"] for day in recent_data])
        returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0])

        # انحراف معیار (نوسان)
        std_dev = float(np.std(returns))

        # حداکثر افت (Drawdown)
        cumulative_returns = np.cumprod(1 + returns)
        max_drawdown = float(1 - min(cumulative_returns / np.maximum.accumulate(cumulative_returns)))

        # نسبت شارپ ساده (بدون نرخ بدون ریسک)
        avg_return = float(np.mean(returns))
        sharpe_ratio = float(avg_return / std_dev) if std_dev > 0 else 0

        # محاسبه سناریوهای ریسک
        current_price = closes[-1]

        # سناریوی پایه (میانه)
        base_scenario = avg_return * 30  # تخمین یک‌ماهه

        # سناریوی خوش‌بینانه
        optimistic_scenario = base_scenario + (std_dev * 2)

        # سناریوی بدبینانه
        pessimistic_scenario = base_scenario - (std_dev * 2)

        # محاسبه ارزش در معرض خطر (Value at Risk)
        var_95 = float(np.percentile(returns, 5))  # 5٪ بدترین حالت (95٪ VaR)
        var_99 = float(np.percentile(returns, 1))  # 1٪ بدترین حالت (99٪ VaR)

        # تعیین سطح کلی ریسک
        risk_level = "medium"
        if std_dev > 0.05 or max_drawdown > 0.3:
            risk_level = "high"
        elif std_dev < 0.02 and max_drawdown < 0.1:
            risk_level = "low"

        # ایجاد نتیجه نهایی
        result = {
            "symbol": symbol,
            "has_data": True,
            "risk_level": risk_level,
            "volatility": round(float(std_dev * 100), 2),  # تبدیل به درصد
            "max_drawdown": round(float(max_drawdown * 100), 2),  # تبدیل به درصد
            "sharpe_ratio": round(float(sharpe_ratio), 2),
            "var_95": round(float(var_95 * 100), 2),  # تبدیل به درصد
            "var_99": round(float(var_99 * 100), 2),  # تبدیل به درصد
            "is_volatile": volatility_analysis["is_volatile"],
            "scenarios": {
                "current_price": round(float(current_price), 2),
                "base": round(float(current_price * (1 + base_scenario)), 2),
                "optimistic": round(float(current_price * (1 + optimistic_scenario)), 2),
                "pessimistic": round(float(current_price * (1 + pessimistic_scenario)), 2)
            },
            "analysis_time": datetime.datetime.now().isoformat()
        }

        return result

    def compare_coins(self, symbols: List[str], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        مقایسه چندین ارز دیجیتال بر اساس معیارهای مختلف

        Args:
            symbols: لیست نمادهای ارزها
            metrics: لیست معیارهای مقایسه (اختیاری)

        Returns:
            دیکشنری نتایج مقایسه
        """
        # بررسی ورودی خالی
        if not symbols:
            return {"error": "لیست ارزها خالی است"}

        # معیارهای پیش‌فرض اگر ارائه نشده باشند
        if not metrics:
            metrics = ["price_change_24h", "price_change_7d", "volatility", "sentiment"]

        # اطمینان از بارگیری داده‌های قیمتی
        self._load_price_data()

        # جمع‌آوری اطلاعات هر ارز
        coins_data = []

        for symbol in symbols:
            # بررسی وجود داده‌های ارز
            if symbol not in self._price_data:
                coins_data.append({
                    "symbol": symbol,
                    "has_data": False
                })
                continue

            # استخراج داده‌های قیمتی
            price_data = self._price_data[symbol]["data"]
            recent_data = price_data[-30:]  # 30 روز اخیر

            # محاسبه معیارها
            coin_metrics = {
                "symbol": symbol,
                "has_data": True,
                "current_price": recent_data[-1]["close"],
                "price_change_24h": self._calculate_price_change(recent_data, 1),
                "price_change_7d": self._calculate_price_change(recent_data, 7),
                "price_change_30d": self._calculate_price_change(recent_data, 30),
                "market_cap": recent_data[-1].get("market_cap", 0),
                "volume_24h": recent_data[-1].get("volume", 0)
            }

            # محاسبه نوسانات
            volatility_analysis = self._analyze_volatility(recent_data)
            coin_metrics["volatility"] = volatility_analysis["volatility"]
            coin_metrics["is_volatile"] = volatility_analysis["is_volatile"]

            # دریافت تحلیل‌های اخبار
            news_analysis = self._get_recent_news_analysis(symbol, 7)  # 7 روز اخیر

            if news_analysis:
                coin_metrics["sentiment"] = news_analysis["dominant_sentiment"]
                coin_metrics["news_impact"] = news_analysis["avg_impact_score"]
                coin_metrics["news_count"] = news_analysis["count"]
            else:
                coin_metrics["sentiment"] = "neutral"
                coin_metrics["news_impact"] = 0.0
                coin_metrics["news_count"] = 0

            coins_data.append(coin_metrics)

        # مرتب‌سازی ارزها بر اساس هر معیار
        rankings = {}

        for metric in metrics:
            # اطمینان از وجود معیار در همه ارزها
            valid_coins = [coin for coin in coins_data if coin["has_data"] and metric in coin]

            if not valid_coins:
                continue

            # تعیین مرتب‌سازی صعودی یا نزولی
            reverse = True  # پیش‌فرض: نزولی (بزرگتر بهتر)

            # برخی معیارها باید صعودی مرتب شوند (کوچکتر بهتر)
            if metric == "volatility" and "risk" in metric.lower():
                reverse = False

            # مرتب‌سازی
            sorted_coins = sorted(valid_coins, key=lambda x: x[metric], reverse=reverse)

            # افزودن رتبه‌بندی
            rankings[metric] = [{"symbol": coin["symbol"], "value": coin[metric]} for coin in sorted_coins]

        # محاسبه رتبه کلی (میانگین رتبه‌ها)
        overall_scores = {}

        for coin in coins_data:
            if not coin["has_data"]:
                continue

            symbol = coin["symbol"]
            overall_scores[symbol] = 0
            metric_count = 0

            for metric in metrics:
                if metric in rankings:
                    # یافتن رتبه ارز در این معیار
                    for i, ranked_coin in enumerate(rankings[metric]):
                        if ranked_coin["symbol"] == symbol:
                            # محاسبه امتیاز معکوس رتبه (رتبه پایین‌تر = امتیاز بیشتر)
                            rank_score = len(rankings[metric]) - i
                            overall_scores[symbol] += rank_score
                            metric_count += 1
                            break

            # محاسبه میانگین
            if metric_count > 0:
                overall_scores[symbol] /= metric_count

        # مرتب‌سازی بر اساس امتیاز کلی
        overall_ranking = [{"symbol": symbol, "score": score}
                           for symbol, score in overall_scores.items()]
        overall_ranking.sort(key=lambda x: x["score"], reverse=True)

        # ایجاد نتیجه نهایی
        result = {
            "coins": coins_data,
            "rankings": rankings,
            "overall_ranking": overall_ranking,
            "analysis_time": datetime.datetime.now().isoformat()
        }

        return result

    def generate_investment_recommendation(self, symbol: str) -> Dict[str, Any]:
        """
        تولید توصیه‌های سرمایه‌گذاری برای یک ارز

        Args:
            symbol: نماد ارز

        Returns:
            دیکشنری توصیه‌های سرمایه‌گذاری
        """
        # تولید گزارش بازار
        market_report = self.generate_market_report(symbol)

        # بررسی خطا
        if "error" in market_report:
            return {"error": market_report["error"], "symbol": symbol}

        # تحلیل ریسک
        risk_analysis = self.analyze_market_risk(symbol)

        # تعیین استراتژی سرمایه‌گذاری
        strategy = self._determine_investment_strategy(market_report, risk_analysis)

        # محاسبه قیمت‌های ورود، هدف و حد ضرر
        price_targets = self._calculate_price_targets(market_report)

        # محاسبه زمان‌بندی توصیه‌شده
        timing = self._determine_investment_timing(market_report)

        # ایجاد نتیجه نهایی
        result = {
            "symbol": symbol,
            "recommendation": strategy["recommendation"],
            "confidence": strategy["confidence"],
            "reasoning": strategy["reasoning"],
            "price_targets": price_targets,
            "timing": timing,
            "risk_level": risk_analysis["risk_level"],
            "warning": f"این تحلیل صرفاً جنبه آموزشی دارد و توصیه سرمایه‌گذاری محسوب نمی‌شود.",
            "analysis_time": datetime.datetime.now().isoformat()
        }

        return result

    def _determine_investment_strategy(self, market_report: Dict[str, Any],
                                       risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        تعیین استراتژی سرمایه‌گذاری بر اساس تحلیل بازار و ریسک

        Args:
            market_report: گزارش بازار
            risk_analysis: تحلیل ریسک

        Returns:
            دیکشنری استراتژی سرمایه‌گذاری
        """
        # بررسی روندها
        price_trend = market_report["price_trend"]["trend"]
        predicted_trend = market_report["predicted_trend"]["trend"]

        # تحلیل ریسک
        risk_level = risk_analysis["risk_level"]
        volatility = risk_analysis["volatility"]

        # تأثیر اخبار
        news_impact = None
        if market_report["news_impact"]:
            news_impact = market_report["news_impact"]["dominant_sentiment"]

        # تصمیم‌گیری بر اساس شرایط مختلف
        recommendation = "hold"  # پیش‌فرض: نگه‌داری
        confidence = "moderate"
        reasoning = ""

        # روند صعودی قوی
        if (price_trend in ["strong_bullish", "bullish"] and
                predicted_trend in ["strong_bullish", "bullish"]):
            recommendation = "buy"
            confidence = "high"
            reasoning = "روند قیمت و پیش‌بینی روند آینده هر دو صعودی هستند."

            if news_impact == "positive":
                confidence = "very_high"
                reasoning += " اخبار اخیر نیز مثبت ارزیابی شده‌اند."

        # روند نزولی قوی
        elif (price_trend in ["strong_bearish", "bearish"] and
              predicted_trend in ["strong_bearish", "bearish"]):
            recommendation = "sell"
            confidence = "high"
            reasoning = "روند قیمت و پیش‌بینی روند آینده هر دو نزولی هستند."

            if news_impact == "negative":
                confidence = "very_high"
                reasoning += " اخبار اخیر نیز منفی ارزیابی شده‌اند."

        # روند صعودی کند یا نامشخص
        elif (price_trend in ["weak_bullish", "neutral"] and
              predicted_trend in ["bullish", "strong_bullish"]):
            recommendation = "buy"
            confidence = "moderate"
            reasoning = "روند فعلی ضعیف یا خنثی است، اما پیش‌بینی روند آینده صعودی است."

        # روند نزولی کند یا نامشخص
        elif (price_trend in ["weak_bearish", "neutral"] and
              predicted_trend in ["bearish", "strong_bearish"]):
            recommendation = "sell"
            confidence = "moderate"
            reasoning = "روند فعلی ضعیف یا خنثی است، اما پیش‌بینی روند آینده نزولی است."

        # بازار پرنوسان
        elif price_trend == "volatile" or predicted_trend == "volatile":
            recommendation = "hold"
            confidence = "moderate"
            reasoning = "بازار پرنوسان است و ریسک معاملات افزایش می‌یابد."

            if risk_level == "high":
                confidence = "high"
                reasoning += " سطح ریسک نیز بالاست."

        # حالت پیش‌فرض - نگه‌داری
        else:
            reasoning = "شرایط بازار مبهم است و روند مشخصی قابل تشخیص نیست."

        # اصلاح توصیه بر اساس ریسک
        if risk_level == "high" and volatility > 15:
            if recommendation == "buy":
                reasoning += " با توجه به ریسک بالا، خرید تدریجی پیشنهاد می‌شود."
            elif recommendation == "sell":
                reasoning += " با توجه به ریسک بالا، فروش تدریجی پیشنهاد می‌شود."

        # تبدیل توصیه به فارسی
        recommendation_fa = {
            "buy": "خرید",
            "sell": "فروش",
            "hold": "نگهداری"
        }.get(recommendation, "نگهداری")

        return {
            "recommendation": recommendation,
            "recommendation_fa": recommendation_fa,
            "confidence": confidence,
            "confidence_fa": self.CONFIDENCE_LEVELS[confidence],
            "reasoning": reasoning
        }

    def _calculate_price_targets(self, market_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        محاسبه قیمت‌های ورود، هدف و حد ضرر

        Args:
            market_report: گزارش بازار

        Returns:
            دیکشنری قیمت‌های هدف
        """
        # دریافت قیمت فعلی
        current_price = market_report["current_price"]

        # دریافت سطوح حمایت و مقاومت
        support_levels = market_report["support_resistance"]["support_levels"]
        resistance_levels = market_report["support_resistance"]["resistance_levels"]

        # محاسبه تلاطم و نوسانات
        volatility = market_report.get("volatility_analysis", {}).get("volatility", 5)
        daily_range = market_report.get("volatility_analysis", {}).get("daily_range_percent", 2)

        # محاسبه قیمت‌های ورود بر اساس نوسانات روزانه
        # برای استراتژی خرید، ورود: قیمت فعلی منهای 1/2 دامنه روزانه
        entry_buy = round(float(current_price * (1 - daily_range / 200)), 2)

        # برای استراتژی فروش، ورود: قیمت فعلی به‌علاوه 1/2 دامنه روزانه
        entry_sell = round(float(current_price * (1 + daily_range / 200)), 2)

        # محاسبه قیمت‌های هدف
        # هدف خرید: مقاومت نزدیک یا افزایش بر اساس نوسانات
        target_buy1 = None
        if resistance_levels:
            target_buy1 = round(float(resistance_levels[0]["price"]), 2)
        else:
            target_buy1 = round(float(current_price * (1 + volatility / 100)), 2)

        # هدف خرید دوم (سود بیشتر)
        target_buy2 = round(float(target_buy1 * 1.05), 2)

        # هدف فروش: حمایت نزدیک یا کاهش بر اساس نوسانات
        target_sell1 = None
        if support_levels:
            target_sell1 = round(float(support_levels[0]["price"]), 2)
        else:
            target_sell1 = round(float(current_price * (1 - volatility / 100)), 2)

        # هدف فروش دوم (سود بیشتر)
        target_sell2 = round(float(target_sell1 * 0.95), 2)

        # محاسبه حد ضرر
        # حد ضرر خرید: سطح حمایت نزدیک یا کاهش 2 برابر نوسانات روزانه
        stop_loss_buy = None
        if support_levels:
            stop_loss_buy = round(float(support_levels[0]["price"] * 0.98), 2)
        else:
            stop_loss_buy = round(float(current_price * (1 - daily_range / 50)), 2)

        # حد ضرر فروش: سطح مقاومت نزدیک یا افزایش 2 برابر نوسانات روزانه
        stop_loss_sell = None
        if resistance_levels:
            stop_loss_sell = round(float(resistance_levels[0]["price"] * 1.02), 2)
        else:
            stop_loss_sell = round(float(current_price * (1 + daily_range / 50)), 2)

        return {
            "current_price": current_price,
            "buy": {
                "entry": entry_buy,
                "target1": target_buy1,
                "target2": target_buy2,
                "stop_loss": stop_loss_buy,
                "risk_reward1": round(float((target_buy1 - entry_buy) / (entry_buy - stop_loss_buy)),
                                      2) if stop_loss_buy < entry_buy else None
            },
            "sell": {
                "entry": entry_sell,
                "target1": target_sell1,
                "target2": target_sell2,
                "stop_loss": stop_loss_sell,
                "risk_reward1": round(float((entry_sell - target_sell1) / (stop_loss_sell - entry_sell)),
                                      2) if stop_loss_sell > entry_sell else None
            }
        }

    def _determine_investment_timing(self, market_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        تعیین زمان‌بندی مناسب برای سرمایه‌گذاری

        Args:
            market_report: گزارش بازار

        Returns:
            دیکشنری زمان‌بندی توصیه‌شده
        """
        # روند پیش‌بینی شده
        predicted_trend = market_report["predicted_trend"]["trend"]
        timeframe = market_report["predicted_trend"]["timeframe"]

        # اطلاعات تکنیکال
        technical_patterns = market_report["technical_patterns"]

        # تعیین فوریت بر اساس روند و الگوها
        urgency = "medium"  # پیش‌فرض

        if predicted_trend in ["strong_bullish", "strong_bearish"]:
            urgency = "high"
        elif predicted_trend in ["weak_bullish", "weak_bearish", "neutral"]:
            urgency = "low"

        # بررسی الگوهای تکنیکال
        has_significant_pattern = False
        for pattern in technical_patterns:
            if pattern["confidence"] > 0.7:
                has_significant_pattern = True
                if pattern["trend_indication"] in ["bullish", "bearish"]:
                    urgency = "high"
                break

        # تعیین محدوده زمانی مناسب
        if urgency == "high":
            time_description = "در اسرع وقت (1-2 روز آینده)"
        elif urgency == "medium":
            time_description = "در کوتاه‌مدت (هفته جاری)"
        else:
            time_description = "با صبر و بررسی بیشتر (هفته‌های آینده)"

        # ایجاد نتیجه نهایی
        result = {
            "urgency": urgency,
            "time_description": time_description,
            "timeframe": timeframe,
            "timeframe_fa": self.TIME_FRAMES[timeframe],
            "has_significant_pattern": has_significant_pattern
        }

        return result

    def generate_text_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """
        تولید متن تحلیلی قابل فهم بر اساس نتایج تحلیل

        Args:
            analysis_result: نتایج تحلیل

        Returns:
            متن تحلیلی
        """
        # این متد می‌تواند متن تحلیلی مناسب برای کاربران غیرمتخصص تولید کند

        # بررسی نوع تحلیل
        if "affected_coins" in analysis_result:
            # تحلیل تأثیر خبر
            return self._generate_news_impact_text(analysis_result)
        elif "recommendation" in analysis_result:
            # توصیه سرمایه‌گذاری
            return self._generate_investment_recommendation_text(analysis_result)
        elif "support_levels" in analysis_result:
            # گزارش بازار
            return _generate_market_report_text(analysis_result)
        elif "risk_level" in analysis_result and "scenarios" in analysis_result:
            # تحلیل ریسک
            return _generate_risk_analysis_text(analysis_result)
        else:
            # متن پیش‌فرض
            return "تحلیل انجام شده است اما نوع تحلیل قابل تشخیص نیست."

    def _generate_news_impact_text(self, analysis_result: Dict[str, Any]) -> str:
        """
        تولید متن تحلیلی برای تأثیر خبر

        Args:
            analysis_result: نتایج تحلیل تأثیر خبر

        Returns:
            متن تحلیلی
        """
        # ساخت متن بر اساس نتایج تحلیل
        has_impact = analysis_result.get("has_impact", False)
        impact_level = analysis_result.get("impact_level", "low")
        sentiment = analysis_result.get("sentiment", {}).get("sentiment", "neutral")

        if not has_impact:
            return "این خبر تأثیر قابل توجهی بر بازار ارزهای دیجیتال ندارد."

        # ساخت متن تأثیر
        impact_text = {
            "very_high": "بسیار زیادی",
            "high": "زیادی",
            "moderate": "متوسطی",
            "low": "کمی",
            "very_low": "بسیار کمی"
        }.get(impact_level, "نامشخصی")

        # متن احساسات
        sentiment_text = {
            "positive": "مثبت",
            "negative": "منفی",
            "neutral": "خنثی"
        }.get(sentiment, "نامشخص")

        # ارزهای متأثر
        affected_coins = analysis_result.get("affected_coins", [])
        coins_text = ""

        if affected_coins:
            coins_list = [coin["symbol"] for coin in affected_coins[:3]]
            coins_text = f"ارزهای {', '.join(coins_list)}"

            if len(affected_coins) > 3:
                coins_text += f" و {len(affected_coins) - 3} ارز دیگر"
        else:
            coins_text = "بازار ارزهای دیجیتال"

        # پیش‌بینی روند
        prediction = analysis_result.get("prediction", {})
        trend_text = ""

        if prediction:
            trend = prediction.get("trend_fa", "")
            estimated_change = prediction.get("estimated_change", 0)
            timeframe = prediction.get("timeframe_fa", "")

            if trend and estimated_change and timeframe:
                direction = "افزایش" if estimated_change > 0 else "کاهش"
                trend_text = f" پیش‌بینی می‌شود که در {timeframe} روند بازار {trend} باشد و احتمال {direction} قیمت حدود {abs(estimated_change)}٪ وجود دارد."

        # ساخت متن نهایی
        text = f"این خبر تأثیر {impact_text} بر {coins_text} دارد. جهت تأثیر {sentiment_text} ارزیابی می‌شود.{trend_text}"

        # افزودن توضیحات بیشتر اگر موجود باشد
        if "analysis" in prediction:
            text += f"\n\nتحلیل بیشتر: {prediction['analysis']}"

        return text

    def _generate_investment_recommendation_text(self, analysis_result: Dict[str, Any]) -> str:
        """
        تولید متن تحلیلی برای توصیه سرمایه‌گذاری

        Args:
            analysis_result: نتایج تحلیل توصیه سرمایه‌گذاری

        Returns:
            متن تحلیلی
        """
        symbol = analysis_result.get("symbol", "")
        recommendation = analysis_result.get("recommendation", "")
        recommendation_fa = {"buy": "خرید", "sell": "فروش", "hold": "نگهداری"}.get(recommendation, "")
        reasoning = analysis_result.get("reasoning", "")
        risk_level = analysis_result.get("risk_level", "")
        risk_fa = {"high": "بالا", "medium": "متوسط", "low": "پایین"}.get(risk_level, "")

        price_targets = analysis_result.get("price_targets", {})
        timing = analysis_result.get("timing", {})

        # ساخت متن توصیه
        text = f"توصیه برای {symbol}: {recommendation_fa}\n\n"
        text += f"دلیل: {reasoning}\n"
        text += f"سطح ریسک: {risk_fa}\n"

        # افزودن اطلاعات زمان‌بندی
        if timing:
            text += f"زمان‌بندی پیشنهادی: {timing.get('time_description', '')}\n"

        # افزودن اطلاعات قیمت‌های هدف
        if price_targets:
            current_price = price_targets.get("current_price", 0)
            text += f"\nقیمت فعلی: {current_price}\n"

            if recommendation == "buy":
                buy_info = price_targets.get("buy", {})
                text += f"قیمت ورود پیشنهادی: {buy_info.get('entry', 0)}\n"
                text += f"قیمت هدف اول: {buy_info.get('target1', 0)}\n"
                text += f"قیمت هدف دوم: {buy_info.get('target2', 0)}\n"
                text += f"حد ضرر: {buy_info.get('stop_loss', 0)}\n"

                risk_reward = buy_info.get('risk_reward1')
                if risk_reward:
                    text += f"نسبت ریسک به ریوارد: {risk_reward}\n"

            elif recommendation == "sell":
                sell_info = price_targets.get("sell", {})
                text += f"قیمت ورود پیشنهادی: {sell_info.get('entry', 0)}\n"
                text += f"قیمت هدف اول: {sell_info.get('target1', 0)}\n"
                text += f"قیمت هدف دوم: {sell_info.get('target2', 0)}\n"
                text += f"حد ضرر: {sell_info.get('stop_loss', 0)}\n"

                risk_reward = sell_info.get('risk_reward1')
                if risk_reward:
                    text += f"نسبت ریسک به ریوارد: {risk_reward}\n"

        # هشدار
        text += f"\nتوجه: {analysis_result.get('warning', 'این تحلیل صرفاً جنبه آموزشی دارد و توصیه سرمایه‌گذاری محسوب نمی‌شود.')}"

        return text

