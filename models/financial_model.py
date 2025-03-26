"""
ماژول تحلیل مالی و ارزهای دیجیتال برای CryptoNewsBot

این ماژول مسئول تحلیل مالی متون خبری مرتبط با ارزهای دیجیتال است.
قابلیت‌های اصلی آن شامل: تشخیص ارزها، تحلیل احساسات مالی، استخراج اشاره‌های قیمتی،
تحلیل تأثیر خبر بر بازار و محاسبه امتیاز اهمیت خبر است.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import Counter

from .base_model import BaseModel, retry_on_error


class FinancialModel(BaseModel):
    """
    مدل تحلیل مالی برای تحلیل اخبار ارزهای دیجیتال

    این کلاس قابلیت‌های پیشرفته تحلیل متون مالی از جمله تشخیص ارزهای دیجیتال،
    تحلیل احساسات مالی، استخراج اطلاعات قیمتی، تحلیل اهمیت خبر و پیش‌بینی 
    تأثیر خبر بر بازار را فراهم می‌کند.
    """

    # انواع احساسات مالی
    SENTIMENT_TYPES = {
        'positive': 'صعودی',
        'negative': 'نزولی',
        'neutral': 'خنثی'
    }

    # انواع حرکت بازار
    MARKET_MOVEMENT = {
        'bullish': 'صعودی',
        'bearish': 'نزولی',
        'sideways': 'رنج',
        'volatile': 'پرنوسان'
    }

    # سطوح تأثیر خبر
    IMPACT_LEVELS = {
        'high': 'زیاد',
        'medium': 'متوسط',
        'low': 'کم'
    }

    def __init__(self) -> None:
        """راه‌اندازی اولیه مدل تحلیل مالی"""
        super().__init__(model_name="FinancialModel", priority=8)  # اولویت بالاتر از مدل ترجمه

        # پیکربندی‌های خاص مدل
        self.model_config.update({
            'financial_model': 'ProsusAI/finbert',  # مدل تحلیل مالی
            'price_threshold': 0.6,  # آستانه تشخیص قیمت
            'market_impact_threshold': 0.7,  # آستانه تأثیر بر بازار
            'batch_size': 16,
            'importance_score_weights': {
                'crypto_importance': 0.25,  # اهمیت ارز دیجیتال ذکر شده
                'sentiment_strength': 0.20,  # قدرت احساسات مثبت/منفی
                'price_mentions': 0.15,  # اشاره به قیمت
                'source_credibility': 0.20,  # اعتبار منبع
                'event_importance': 0.20  # اهمیت رویداد ذکر شده
            }
        })

        # وضعیت مدل‌های مختلف
        self.financial_model_loaded = False

        # دریافت داده‌های ارزهای دیجیتال
        self._crypto_data = None
        self._financial_terms = None
        self._important_events = None

        self.logger.debug("مدل تحلیل مالی راه‌اندازی شد")

    def _setup_model(self) -> None:
        """تنظیمات اولیه مدل - هیچ کاری انجام نمی‌دهد (بارگیری تنبل)"""
        pass

    def load_model(self) -> None:
        """
        بارگیری اولیه مدل‌های مورد نیاز

        این متد برای بارگیری پایه و اطمینان از عملکرد صحیح است.
        هر مدل به صورت جداگانه با بارگیری تنبل مدیریت می‌شود.
        """
        if self.model_loaded:
            return

        try:
            # بارگیری داده‌های پایه
            self._load_crypto_data()
            self._load_financial_terms()
            self._load_important_events()

            self.model_loaded = True
            self.logger.info("مدل تحلیل مالی با موفقیت بارگیری شد")

        except Exception as e:
            self.logger.error(f"خطا در بارگیری مدل تحلیل مالی: {str(e)}")
            raise

    def _load_financial_model(self) -> None:
        """بارگیری مدل تحلیل مالی"""
        if self.financial_model_loaded:
            return

        self.logger.info("در حال بارگیری مدل تحلیل احساسات مالی...")

        # بارگیری مدل از HuggingFace
        model_name = self.model_config['financial_model']
        self.financial_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.financial_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # انتقال به دستگاه مناسب
        self.financial_model.to(self.device)

        # تنظیم حالت ارزیابی
        self.financial_model.eval()

        self.financial_model_loaded = True
        self.logger.info(f"مدل تحلیل احساسات مالی {model_name} با موفقیت بارگیری شد")

    def _load_crypto_data(self) -> None:
        """بارگیری داده‌های ارزهای دیجیتال از مخزن داده"""
        if self._crypto_data is not None:
            return

        self.logger.debug("در حال بارگیری داده‌های ارزهای دیجیتال...")

        # دریافت از مخزن داده
        self._crypto_data = self.data_repo.get_data('crypto')

        if not self._crypto_data:
            self.logger.warning("داده‌های ارزهای دیجیتال یافت نشد")
            self._crypto_data = {}

        self.logger.debug(f"داده‌های {len(self._crypto_data)} ارز دیجیتال با موفقیت بارگیری شد")

    def _load_financial_terms(self) -> None:
        """بارگیری اصطلاحات مالی از مخزن داده"""
        if self._financial_terms is not None:
            return

        self.logger.debug("در حال بارگیری اصطلاحات مالی...")

        # دریافت از مخزن داده
        self._financial_terms = self.data_repo.get_data('terms')

        if not self._financial_terms:
            self.logger.warning("اصطلاحات مالی یافت نشد")
            self._financial_terms = {}

        self.logger.debug(
            f"اصطلاحات مالی با موفقیت بارگیری شد ({sum(len(terms) for terms in self._financial_terms.values())} اصطلاح)")

    def _load_important_events(self) -> None:
        """بارگیری رویدادهای مهم از مخزن داده"""
        if self._important_events is not None:
            return

        self.logger.debug("در حال بارگیری رویدادهای مهم...")

        # دریافت از مخزن داده
        self._important_events = self.data_repo.get_data('events')

        if not self._important_events:
            self.logger.warning("رویدادهای مهم یافت نشد")
            self._important_events = {}

        self.logger.debug("رویدادهای مهم با موفقیت بارگیری شد")

    @retry_on_error()
    def extract_cryptocurrencies(self, text: str) -> Dict[str, Any]:
        """
        استخراج ارزهای دیجیتال ذکر شده در متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'found': bool, 'cryptocurrencies': [...], 'primary': str}
        """
        return self.cached_call(
            "extract_cryptocurrencies",
            text,
            self._extract_cryptocurrencies_impl
        )

    def _extract_cryptocurrencies_impl(self, text: str) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی استخراج ارزهای دیجیتال

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'found': bool, 'cryptocurrencies': [...], 'primary': str}
        """
        # بررسی متن خالی
        if not text:
            return {
                'found': False,
                'cryptocurrencies': [],
                'primary': None
            }

        # بارگیری داده‌های ارزها
        self._load_crypto_data()

        # تبدیل متن به حروف کوچک برای تطبیق بهتر
        text = text.lower()

        # یافتن ارزهای مذکور در متن
        found_cryptos = {}

        for symbol, data in self._crypto_data.items():
            for name in data['names']:
                name_lower = name.lower()
                # جستجوی دقیق نام ارز (با در نظر گرفتن مرز کلمات)
                if re.search(r'\b' + re.escape(name_lower) + r'\b', text):
                    if symbol not in found_cryptos:
                        found_cryptos[symbol] = {
                            'symbol': symbol,
                            'mentions': [],
                            'importance': data['importance'],
                            'count': 0
                        }
                    found_cryptos[symbol]['mentions'].append(name)
                    found_cryptos[symbol]['count'] += 1

        # تبدیل به لیست و مرتب‌سازی بر اساس تعداد تکرار و اهمیت
        crypto_list = list(found_cryptos.values())
        crypto_list.sort(key=lambda x: (x['count'], x['importance']), reverse=True)

        result = {
            'found': len(crypto_list) > 0,
            'cryptocurrencies': crypto_list,
            'primary': crypto_list[0]['symbol'] if crypto_list else None
        }

        return result

    def batch_extract_cryptocurrencies(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        استخراج ارزهای دیجیتال از چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست دیکشنری‌های نتایج
        """
        return self.batch_cached_call(
            "extract_cryptocurrencies",
            texts,
            lambda x: [self._extract_cryptocurrencies_impl(text) for text in x]
        )

    @retry_on_error()
    def analyze_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات مالی متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'sentiment': str, 'score': float, 'financial_terms': []}
        """
        return self.cached_call(
            "analyze_financial_sentiment",
            text,
            self._analyze_financial_sentiment_impl
        )

    def _analyze_financial_sentiment_impl(self, text: str) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی تحلیل احساسات مالی

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'sentiment': str, 'score': float, 'financial_terms': []}
        """
        # بررسی متن خالی
        if not text:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'market_sentiment': 'sideways',
                'financial_terms': []
            }

        # بارگیری اصطلاحات مالی
        self._load_financial_terms()

        # 1. تحلیل بر اساس اصطلاحات مالی
        term_based_result = self._analyze_sentiment_by_terms(text)

        # 2. تحلیل با مدل تحلیل احساسات مالی
        model_based_result = self._analyze_sentiment_by_model(text)

        # 3. ترکیب نتایج
        final_result = self._combine_sentiment_results(term_based_result, model_based_result)

        return final_result

    def _analyze_sentiment_by_terms(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات بر اساس اصطلاحات مالی

        Args:
            text: متن ورودی

        Returns:
            دیکشنری نتایج
        """
        # تبدیل به حروف کوچک
        text_lower = text.lower()

        # شمارش اصطلاحات مختلف
        term_counts = {
            'bullish': 0,
            'bearish': 0,
            'stable': 0,
            'volatility': 0,
            'accumulation': 0,
            'distribution': 0,
            'fomo': 0,
            'fud': 0
        }

        found_terms = {
            'bullish': [],
            'bearish': [],
            'stable': [],
            'volatility': [],
            'accumulation': [],
            'distribution': [],
            'fomo': [],
            'fud': []
        }

        # بررسی هر دسته اصطلاح
        for term_type, terms in self._financial_terms.items():
            if term_type in term_counts:
                for term in terms:
                    if re.search(r'\b' + re.escape(term.lower()) + r'\b', text_lower):
                        term_counts[term_type] += 1
                        found_terms[term_type].append(term)

        # محاسبه نتیجه احساسات بر اساس اصطلاحات
        bullish_score = term_counts['bullish'] + term_counts['accumulation'] + term_counts['fomo'] * 0.5
        bearish_score = term_counts['bearish'] + term_counts['distribution'] + term_counts['fud'] * 0.5
        stable_score = term_counts['stable']
        volatility_score = term_counts['volatility']

        # تعیین احساس نهایی
        if bullish_score > bearish_score:
            sentiment = 'positive'
            score = min(1.0, bullish_score / (bullish_score + bearish_score + stable_score + 0.001))
        elif bearish_score > bullish_score:
            sentiment = 'negative'
            score = min(1.0, bearish_score / (bullish_score + bearish_score + stable_score + 0.001))
        else:
            sentiment = 'neutral'
            score = min(1.0, stable_score / (bullish_score + bearish_score + stable_score + 0.001))

        # تعیین حالت بازار
        if volatility_score > max(bullish_score, bearish_score, stable_score):
            market_sentiment = 'volatile'
        elif stable_score > max(bullish_score, bearish_score):
            market_sentiment = 'sideways'
        elif bullish_score > bearish_score:
            market_sentiment = 'bullish'
        else:
            market_sentiment = 'bearish'

        # ایجاد فهرست مسطح اصطلاحات یافت شده
        all_found_terms = []
        for term_type, terms in found_terms.items():
            all_found_terms.extend(terms)

        return {
            'sentiment': sentiment,
            'score': score,
            'market_sentiment': market_sentiment,
            'financial_terms': all_found_terms,
            'term_counts': term_counts
        }

    def _analyze_sentiment_by_model(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات با استفاده از مدل تحلیل احساسات مالی

        Args:
            text: متن ورودی

        Returns:
            دیکشنری نتایج
        """
        try:
            # بارگیری مدل
            self._load_financial_model()

            # ترکیب چند متن طولانی می‌تواند باعث خطای حافظه شود
            # متن را کوتاه می‌کنیم
            text = text[:2048]

            # پردازش متن با توکنایزر
            inputs = self.financial_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # انتقال به دستگاه مناسب
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # تحلیل احساسات
            with torch.no_grad():
                outputs = self.financial_model(**inputs)

            # پیش‌بینی نتایج با softmax
            predictions = F.softmax(outputs.logits, dim=1)

            # تبدیل به numpy
            predictions = predictions[0].cpu().numpy()

            # بررسی برچسب‌های مدل FinBERT
            # 0: نزولی (negative)، 1: خنثی (neutral)، 2: صعودی (positive)
            negative_score = predictions[0]
            neutral_score = predictions[1]
            positive_score = predictions[2]

            # تعیین احساس نهایی
            if positive_score > negative_score and positive_score > neutral_score:
                sentiment = 'positive'
                score = positive_score
                market_sentiment = 'bullish'
            elif negative_score > positive_score and negative_score > neutral_score:
                sentiment = 'negative'
                score = negative_score
                market_sentiment = 'bearish'
            else:
                sentiment = 'neutral'
                score = neutral_score
                market_sentiment = 'sideways'

            return {
                'sentiment': sentiment,
                'score': float(score),
                'market_sentiment': market_sentiment,
                'model_predictions': {
                    'positive': float(positive_score),
                    'negative': float(negative_score),
                    'neutral': float(neutral_score)
                }
            }

        except Exception as e:
            self.logger.warning(f"خطا در تحلیل احساسات با مدل: {str(e)}")
            # بازگشت مقدار پیش‌فرض
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'market_sentiment': 'sideways',
                'model_predictions': None
            }

    def _combine_sentiment_results(self, term_result: Dict[str, Any], model_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        ترکیب نتایج تحلیل احساسات از روش‌های مختلف

        Args:
            term_result: نتیجه تحلیل بر اساس اصطلاحات
            model_result: نتیجه تحلیل با مدل

        Returns:
            دیکشنری نتیجه نهایی
        """
        # وزن هر روش
        term_weight = 0.6
        model_weight = 0.4

        # اگر تعداد اصطلاحات زیاد باشد، وزن آن بیشتر می‌شود
        total_terms = sum(term_result['term_counts'].values())
        if total_terms > 5:
            term_weight = 0.7
            model_weight = 0.3

        # ترکیب امتیاز احساسات
        if term_result['sentiment'] == model_result['sentiment']:
            # اگر هر دو روش به یک نتیجه رسیدند
            sentiment = term_result['sentiment']
            score = term_result['score'] * term_weight + model_result['score'] * model_weight
        else:
            # اگر نتایج متفاوت است، روش با امتیاز بالاتر را انتخاب می‌کنیم
            if term_result['score'] * term_weight > model_result['score'] * model_weight:
                sentiment = term_result['sentiment']
                score = term_result['score'] * term_weight
            else:
                sentiment = model_result['sentiment']
                score = model_result['score'] * model_weight

        # ترکیب حالت بازار
        if term_result['market_sentiment'] == model_result['market_sentiment']:
            market_sentiment = term_result['market_sentiment']
        else:
            # اگر نتایج متفاوت است، روش با وزن بیشتر را انتخاب می‌کنیم
            if term_weight > model_weight:
                market_sentiment = term_result['market_sentiment']
            else:
                market_sentiment = model_result['market_sentiment']

        # ایجاد نتیجه نهایی
        result = {
            'sentiment': sentiment,
            'score': float(min(1.0, score)),
            'market_sentiment': market_sentiment,
            'financial_terms': term_result['financial_terms'],
            'term_analysis': {
                'sentiment': term_result['sentiment'],
                'score': term_result['score'],
                'term_counts': term_result['term_counts']
            }
        }

        # افزودن نتایج مدل اگر در دسترس باشد
        if model_result['model_predictions'] is not None:
            result['model_analysis'] = {
                'sentiment': model_result['sentiment'],
                'score': model_result['score'],
                'predictions': model_result['model_predictions']
            }

        return result

    @retry_on_error()
    def extract_price_mentions(self, text: str) -> Dict[str, Any]:
        """
        استخراج اشاره‌های قیمتی و درصدی از متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'has_price_info': bool, 'prices': [...], 'percentages': [...]}
        """
        return self.cached_call(
            "extract_price_mentions",
            text,
            self._extract_price_mentions_impl
        )

    def _extract_price_mentions_impl(self, text: str) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی استخراج اشاره‌های قیمتی

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'has_price_info': bool, 'prices': [...], 'percentages': [...]}
        """
        # بررسی متن خالی
        if not text:
            return {
                'has_price_info': False,
                'prices': [],
                'percentages': []
            }

        # الگوهای مختلف قیمت
        price_patterns = [
            # دلار
            r'(\$[\d,.]+)',  # $10,000
            r'([\d,.]+\s*دلار)',  # 10,000 دلار
            r'([\d,.]+\s*\$)',  # 10,000$
            r'([\d,.]+\s*USD)',  # 10,000 USD
            # تومان
            r'([\d,.]+\s*تومان)',  # 10,000 تومان
            r'([\d,.]+\s*تومن)',  # 10,000 تومن
            r'([\d,.]+\s*IRT)',  # 10,000 IRT
            # واحدهای بزرگ
            r'([\d,.]+\s*[Kk])',  # 10K
            r'([\d,.]+\s*هزار)',  # 10 هزار
            r'([\d,.]+\s*میلیون)',  # 10 میلیون
            r'([\d,.]+\s*میلیارد)',  # 10 میلیارد
            r'([\d,.]+\s*[Mm]illion)',  # 10 Million
            r'([\d,.]+\s*[Bb]illion)',  # 10 Billion
            # سایر ارزها
            r'([\d,.]+\s*یورو)',  # 10,000 یورو
            r'([\d,.]+\s*EUR)',  # 10,000 EUR
            r'([\d,.]+\s*€)',  # 10,000€
        ]

        # الگوهای درصد
        percentage_patterns = [
            r'([\d,.]+\s*درصد)',  # 10 درصد
            r'([\d,.]+\s*%)',  # 10%
            r'([\d,.]+\s*٪)'  # 10٪
        ]

        # استخراج قیمت‌ها
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            prices.extend(matches)

        # استخراج درصدها
        percentages = []
        for pattern in percentage_patterns:
            matches = re.findall(pattern, text)
            percentages.extend(matches)

        # پاکسازی و یکتاسازی
        prices = [price.strip() for price in prices]
        prices = list(set(prices))  # حذف تکرارها

        percentages = [percentage.strip() for percentage in percentages]
        percentages = list(set(percentages))  # حذف تکرارها

        return {
            'has_price_info': len(prices) > 0 or len(percentages) > 0,
            'prices': prices,
            'percentages': percentages
        }

    @retry_on_error()
    def extract_important_events(self, text: str) -> Dict[str, Any]:
        """
        استخراج رویدادهای مهم مذکور در متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'has_events': bool, 'events': [...]}
        """
        return self.cached_call(
            "extract_important_events",
            text,
            self._extract_important_events_impl
        )

    def _extract_important_events_impl(self, text: str) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی استخراج رویدادهای مهم

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {'has_events': bool, 'events': [...]}
        """
        # بررسی متن خالی
        if not text:
            return {
                'has_events': False,
                'events': []
            }

        # بارگیری رویدادهای مهم
        self._load_important_events()

        # تبدیل به حروف کوچک
        text_lower = text.lower()

        # یافتن رویدادهای مذکور در متن
        found_events = []

        # بررسی رویدادهای فارسی
        if 'fa' in self._important_events:
            for event in self._important_events['fa']:
                event_name = event['name'].lower()
                if re.search(r'\b' + re.escape(event_name) + r'\b', text_lower):
                    found_events.append({
                        'name': event['name'],
                        'importance': event['importance'],
                        'language': 'fa'
                    })

                    # افزایش شمارنده تشخیص رویداد
                    try:
                        self.data_repo.increment_event_detection(event['name'], 'fa')
                    except Exception:
                        pass

        # بررسی رویدادهای انگلیسی
        if 'en' in self._important_events:
            for event in self._important_events['en']:
                event_name = event['name'].lower()
                if re.search(r'\b' + re.escape(event_name) + r'\b', text_lower):
                    found_events.append({
                        'name': event['name'],
                        'importance': event['importance'],
                        'language': 'en'
                    })

                    # افزایش شمارنده تشخیص رویداد
                    try:
                        self.data_repo.increment_event_detection(event['name'], 'en')
                    except Exception:
                        pass

        # مرتب‌سازی بر اساس اهمیت
        found_events.sort(key=lambda x: x['importance'], reverse=True)

        return {
            'has_events': len(found_events) > 0,
            'events': found_events
        }

    @retry_on_error()
    def analyze_market_impact(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        تحلیل تأثیر خبر بر بازار

        Args:
            text: متن خبر
            source_info: اطلاعات منبع خبر (اختیاری)

        Returns:
            دیکشنری {'impact_level': str, 'impact_score': float, ...}
        """
        return self.cached_call(
            "analyze_market_impact",
            text,
            lambda x: self._analyze_market_impact_impl(x, source_info)
        )

    def _analyze_market_impact_impl(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی تحلیل تأثیر خبر بر بازار

        Args:
            text: متن خبر
            source_info: اطلاعات منبع خبر (اختیاری)

        Returns:
            دیکشنری {'impact_level': str, 'impact_score': float, ...}
        """
        # بررسی متن خالی
        if not text:
            return {
                'impact_level': 'low',
                'impact_score': 0.0,
                'market_sentiment': 'neutral',
                'primary_crypto': None,
                'analysis': "اطلاعات کافی برای تحلیل وجود ندارد."
            }

        # تحلیل‌های مختلف
        crypto_result = self.extract_cryptocurrencies(text)
        sentiment_result = self.analyze_financial_sentiment(text)
        price_result = self.extract_price_mentions(text)
        events_result = self.extract_important_events(text)

        # محاسبه امتیاز تأثیر
        impact_score = self._calculate_impact_score(
            crypto_result,
            sentiment_result,
            price_result,
            events_result,
            source_info
        )

        # تعیین سطح تأثیر
        if impact_score >= self.model_config['market_impact_threshold']:
            impact_level = 'high'
        elif impact_score >= self.model_config['market_impact_threshold'] * 0.6:
            impact_level = 'medium'
        else:
            impact_level = 'low'

        # تولید تحلیل متنی
        analysis_text = self._generate_market_analysis(
            text,
            crypto_result,
            sentiment_result,
            impact_level,
            price_result,
            events_result
        )

        # ایجاد نتیجه نهایی
        result = {
            'impact_level': impact_level,
            'impact_score': round(float(impact_score), 2),
            'market_sentiment': sentiment_result['market_sentiment'],
            'sentiment_score': round(float(sentiment_result['score']), 2),
            'primary_crypto': crypto_result['primary'],
            'cryptocurrencies': [c['symbol'] for c in crypto_result['cryptocurrencies']],
            'has_price_info': price_result['has_price_info'],
            'has_events': events_result['has_events'],
            'events': [e['name'] for e in events_result['events']],
            'analysis': analysis_text
        }

        return result

    def _calculate_impact_score(self, crypto_result: Dict[str, Any], sentiment_result: Dict[str, Any],
                                price_result: Dict[str, Any], events_result: Dict[str, Any],
                                source_info: Optional[Dict[str, Any]] = None) -> float:
        """
        محاسبه امتیاز تأثیر خبر بر بازار

        Args:
            crypto_result: نتیجه استخراج ارزها
            sentiment_result: نتیجه تحلیل احساسات
            price_result: نتیجه استخراج قیمت‌ها
            events_result: نتیجه استخراج رویدادها
            source_info: اطلاعات منبع خبر

        Returns:
            امتیاز تأثیر (0 تا 1)
        """
        # وزن‌های هر فاکتور
        weights = self.model_config['importance_score_weights']

        # 1. امتیاز بر اساس ارزهای مذکور
        crypto_score = 0.0
        if crypto_result['found']:
            primary_crypto = crypto_result['primary']
            crypto_data = next((c for c in crypto_result['cryptocurrencies'] if c['symbol'] == primary_crypto), None)

            if crypto_data:
                crypto_score = crypto_data['importance']
                # بونوس برای تعدد ارزها
                crypto_score += min(0.2, len(crypto_result['cryptocurrencies']) * 0.05)

        # 2. امتیاز بر اساس قدرت احساسات
        sentiment_score = 0.0
        if sentiment_result['sentiment'] != 'neutral':
            sentiment_score = sentiment_result['score']

        # 3. امتیاز بر اساس اشاره‌های قیمتی
        price_score = 0.0
        if price_result['has_price_info']:
            price_count = len(price_result['prices']) + len(price_result['percentages'])
            price_score = min(1.0, price_count * 0.2)

        # 4. امتیاز بر اساس اعتبار منبع
        source_score = 0.5  # پیش‌فرض
        if source_info and 'credibility' in source_info:
            source_score = source_info['credibility']

        # 5. امتیاز بر اساس رویدادهای مهم
        event_score = 0.0
        if events_result['has_events']:
            # استفاده از مهم‌ترین رویداد
            if events_result['events']:
                event_score = events_result['events'][0]['importance']

        # محاسبه امتیاز نهایی با وزن‌دهی
        final_score = (
                crypto_score * weights['crypto_importance'] +
                sentiment_score * weights['sentiment_strength'] +
                price_score * weights['price_mentions'] +
                source_score * weights['source_credibility'] +
                event_score * weights['event_importance']
        )

        # محدود کردن به بازه [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        return final_score

    def _generate_market_analysis(self, text: str, crypto_result: Dict[str, Any], sentiment_result: Dict[str, Any],
                                  impact_level: str, price_result: Dict[str, Any],
                                  events_result: Dict[str, Any]) -> str:
        """
        تولید تحلیل متنی بازار بر اساس نتایج تحلیل‌ها

        Args:
            text: متن خبر
            crypto_result: نتیجه استخراج ارزها
            sentiment_result: نتیجه تحلیل احساسات
            impact_level: سطح تأثیر
            price_result: نتیجه استخراج قیمت‌ها
            events_result: نتیجه استخراج رویدادها

        Returns:
            متن تحلیل
        """
        # عبارات پایه برای هر سطح تأثیر
        impact_phrases = {
            'high': [
                "این خبر می‌تواند تأثیر قابل توجهی بر بازار داشته باشد.",
                "این خبر احتمالاً تأثیر معناداری بر قیمت‌ها خواهد داشت.",
                "با توجه به اهمیت این خبر، انتظار تغییرات قیمتی قابل توجهی می‌رود."
            ],
            'medium': [
                "این خبر ممکن است تأثیر متوسطی بر بازار داشته باشد.",
                "با توجه به محتوای خبر، احتمال تغییرات معتدل در قیمت‌ها وجود دارد.",
                "بازار احتمالاً به این خبر واکنش نشان خواهد داد، اما تأثیر آن محدود خواهد بود."
            ],
            'low': [
                "تأثیر این خبر بر بازار احتمالاً محدود خواهد بود.",
                "انتظار نمی‌رود این خبر تأثیر قابل توجهی بر قیمت‌ها داشته باشد.",
                "این خبر احتمالاً تأثیر کمی بر روند فعلی بازار خواهد داشت."
            ]
        }

        # عبارات احساسات بازار
        sentiment_phrases = {
            'bullish': [
                "روند {crypto} ممکن است صعودی شود.",
                "این خبر می‌تواند باعث افزایش قیمت {crypto} شود.",
                "سرمایه‌گذاران ممکن است با اشتیاق بیشتری به خرید {crypto} بپردازند."
            ],
            'bearish': [
                "روند {crypto} ممکن است نزولی شود.",
                "این خبر می‌تواند باعث کاهش قیمت {crypto} شود.",
                "احتمال فشار فروش بر روی {crypto} وجود دارد."
            ],
            'sideways': [
                "انتظار می‌رود قیمت {crypto} در محدوده فعلی حرکت کند.",
                "تغییر قیمت قابل توجهی برای {crypto} پیش‌بینی نمی‌شود.",
                "بازار {crypto} احتمالاً در کوتاه‌مدت ثبات خواهد داشت."
            ],
            'volatile': [
                "نوسانات قیمتی شدیدی برای {crypto} قابل پیش‌بینی است.",
                "افزایش تلاطم بازار {crypto} محتمل است.",
                "سرمایه‌گذاران باید آماده نوسانات بیشتر در قیمت {crypto} باشند."
            ]
        }

        # ساخت تحلیل
        analysis = []

        # افزودن عبارت سطح تأثیر
        import random
        analysis.append(random.choice(impact_phrases[impact_level]))

        # افزودن اطلاعات ارز اصلی
        if crypto_result['found']:
            primary_crypto = crypto_result['primary']

            # افزودن عبارت احساسات بازار
            market_sentiment = sentiment_result['market_sentiment']
            selected_phrase = random.choice(sentiment_phrases[market_sentiment])
            analysis.append(selected_phrase.format(crypto=primary_crypto))

            # افزودن اطلاعات قیمتی
            if price_result['has_price_info'] and impact_level != 'low':
                analysis.append(f"در این خبر به قیمت‌ها و تغییرات قیمتی اشاره شده است.")

            # افزودن اطلاعات رویدادها
            if events_result['has_events'] and events_result['events']:
                top_event = events_result['events'][0]
                analysis.append(
                    f"این خبر به رویداد مهم \"{top_event['name']}\" اشاره دارد که می‌تواند عامل تأثیرگذاری باشد.")

        else:
            # اگر ارز خاصی ذکر نشده باشد
            analysis.append(
                "در این خبر به ارز دیجیتال خاصی اشاره نشده، بنابراین تأثیر آن بر بازار ارزهای دیجیتال عمومی خواهد بود.")

        # توصیه نهایی بر اساس سطح تأثیر
        if impact_level == 'high':
            analysis.append("سرمایه‌گذاران و فعالان بازار باید این خبر را با دقت دنبال کنند.")
        elif impact_level == 'medium':
            analysis.append("پیشنهاد می‌شود فعالان بازار توجه ویژه‌ای به تحولات بعدی مرتبط با این خبر داشته باشند.")

        # ترکیب تحلیل نهایی
        return " ".join(analysis)

    @retry_on_error()
    def calculate_news_importance(self, text: str, source_info: Optional[Dict[str, Any]] = None,
                                  previous_news: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        محاسبه امتیاز کلی اهمیت خبر

        این متد تمام تحلیل‌های قبلی را ترکیب می‌کند تا یک امتیاز کلی
        برای اهمیت خبر محاسبه کند.

        Args:
            text: متن خبر
            source_info: اطلاعات منبع خبر (اختیاری)
            previous_news: لیست اخبار قبلی برای تشخیص تکرار (اختیاری)

        Returns:
            دیکشنری {'importance_score': float, 'impact_score': float, ...}
        """
        return self.cached_call(
            "calculate_news_importance",
            text,
            lambda x: self._calculate_news_importance_impl(x, source_info, previous_news)
        )

    def _calculate_news_importance_impl(self, text: str, source_info: Optional[Dict[str, Any]] = None,
                                        previous_news: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی محاسبه امتیاز کلی اهمیت خبر

        Args:
            text: متن خبر
            source_info: اطلاعات منبع خبر (اختیاری)
            previous_news: لیست اخبار قبلی برای تشخیص تکرار (اختیاری)

        Returns:
            دیکشنری {'importance_score': float, 'impact_score': float, ...}
        """
        # بررسی متن خالی
        if not text:
            return {
                'importance_score': 0.0,
                'impact_score': 0.0,
                'is_important': False,
                'market_impact': {
                    'impact_level': 'low',
                    'market_sentiment': 'neutral'
                },
                'duplicated': False,
                'primary_crypto': None
            }

        # تحلیل تأثیر بر بازار
        market_impact = self.analyze_market_impact(text, source_info)

        # فاکتورهای دیگر مؤثر بر اهمیت
        importance_factors = {}

        # 1. تازگی خبر (قابل اعمال اگر اطلاعات زمانی ارائه شود)
        recency_score = 1.0  # پیش‌فرض
        if source_info and 'published_at' in source_info:
            import datetime
            now = datetime.datetime.now()
            published_at = source_info['published_at']

            # محاسبه تفاوت زمانی به ساعت
            time_diff_hours = (now - published_at).total_seconds() / 3600

            # خبر تازه‌تر، مهم‌تر است
            if time_diff_hours <= 2:  # 2 ساعت اخیر
                recency_score = 1.0
            elif time_diff_hours <= 6:  # 6 ساعت اخیر
                recency_score = 0.9
            elif time_diff_hours <= 12:  # 12 ساعت اخیر
                recency_score = 0.8
            elif time_diff_hours <= 24:  # 24 ساعت اخیر
                recency_score = 0.7
            elif time_diff_hours <= 48:  # 48 ساعت اخیر
                recency_score = 0.5
            else:
                recency_score = 0.3

        importance_factors['recency'] = recency_score

        # 2. تکراری بودن خبر
        duplicated_score = 1.0  # پیش‌فرض (غیرتکراری)
        is_duplicated = False

        if previous_news:
            # بررسی تکراری بودن با اخبار قبلی
            for prev_news in previous_news:
                # محاسبه شباهت متنی ساده (می‌توان از روش‌های پیچیده‌تر استفاده کرد)
                similarity = self._calculate_text_similarity(text, prev_news)

                if similarity > 0.7:  # آستانه تشخیص تکرار
                    duplicated_score = 0.2
                    is_duplicated = True
                    break
                elif similarity > 0.5:  # شباهت نسبی
                    duplicated_score = 0.5
                    is_duplicated = True
                    break

        importance_factors['uniqueness'] = duplicated_score

        # 3. طول خبر (خبرهای طولانی‌تر معمولاً مهم‌ترند)
        length_score = min(1.0, len(text) / 2000)  # نرمال‌سازی طول
        importance_factors['length'] = length_score

        # 4. داشتن اطلاعات دقیق (مثل قیمت، درصد و...)
        precision_score = 0.5  # پیش‌فرض
        if market_impact['has_price_info']:
            precision_score = 0.8

        importance_factors['precision'] = precision_score

        # محاسبه امتیاز نهایی اهمیت
        # تأثیر بازار، مهم‌ترین فاکتور است
        impact_weight = 0.5
        recency_weight = 0.15
        uniqueness_weight = 0.15
        length_weight = 0.1
        precision_weight = 0.1

        importance_score = (
                market_impact['impact_score'] * impact_weight +
                recency_score * recency_weight +
                duplicated_score * uniqueness_weight +
                length_score * length_weight +
                precision_score * precision_weight
        )

        # محدود کردن به بازه [0, 1]
        importance_score = max(0.0, min(1.0, importance_score))

        # تعیین اهمیت کلی خبر
        is_important = importance_score >= 0.6 and not is_duplicated

        # ایجاد نتیجه نهایی
        result = {
            'importance_score': round(float(importance_score), 2),
            'impact_score': market_impact['impact_score'],
            'is_important': is_important,
            'market_impact': {
                'impact_level': market_impact['impact_level'],
                'market_sentiment': market_impact['market_sentiment'],
                'analysis': market_impact['analysis']
            },
            'importance_factors': importance_factors,
            'duplicated': is_duplicated,
            'primary_crypto': market_impact['primary_crypto'],
            'events': market_impact.get('events', [])
        }

        return result

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه شباهت ساده بین دو متن

        برای محاسبه شباهت از روش اشتراک کلمات استفاده می‌کند.

        Args:
            text1: متن اول
            text2: متن دوم

        Returns:
            شباهت (0 تا 1)
        """
        # تبدیل به حروف کوچک و تقسیم به کلمات
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # محاسبه اشتراک و اجتماع کلمات
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # محاسبه ضریب جاکارد
        if not union:
            return 0.0

        return len(intersection) / len(union)

    def analyze_complete_news(self, text: str, title: Optional[str] = None,
                              source_info: Optional[Dict[str, Any]] = None,
                              previous_news: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        تحلیل جامع و کامل یک خبر

        این متد تمام تحلیل‌های مختلف را در یک فراخوانی انجام می‌دهد
        و یک نتیجه جامع برمی‌گرداند.

        Args:
            text: متن خبر
            title: عنوان خبر (اختیاری)
            source_info: اطلاعات منبع خبر (اختیاری)
            previous_news: لیست اخبار قبلی برای تشخیص تکرار (اختیاری)

        Returns:
            دیکشنری با تمام نتایج تحلیل
        """
        # ترکیب عنوان و متن
        full_text = text
        if title:
            full_text = f"{title}\n{text}"

        # تحلیل‌های مختلف
        cryptocurrencies = self.extract_cryptocurrencies(full_text)
        sentiment = self.analyze_financial_sentiment(full_text)
        price_mentions = self.extract_price_mentions(full_text)
        events = self.extract_important_events(full_text)
        market_impact = self.analyze_market_impact(full_text, source_info)
        importance = self.calculate_news_importance(full_text, source_info, previous_news)

        # ایجاد نتیجه نهایی
        result = {
            'title': title,
            'cryptocurrencies': cryptocurrencies,
            'sentiment': sentiment,
            'price_mentions': price_mentions,
            'events': events,
            'market_impact': market_impact,
            'importance': importance,
            'summary': {
                'is_important': importance['is_important'],
                'importance_score': importance['importance_score'],
                'impact_level': market_impact['impact_level'],
                'market_sentiment': sentiment['market_sentiment'],
                'primary_crypto': cryptocurrencies['primary'],
                'duplicated': importance['duplicated']
            }
        }

        return result
