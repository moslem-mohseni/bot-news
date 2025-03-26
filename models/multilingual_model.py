"""
ماژول پردازش چندزبانه برای CryptoNewsBot

این ماژول مسئول تشخیص زبان و ترجمه متون به فارسی است. از مدل‌های پیشرفته
ترجمه ماشینی چندزبانه استفاده می‌کند و قابلیت پردازش متون بلند را با 
تقسیم به بخش‌های کوچکتر فراهم می‌سازد.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import fasttext
import fasttext.util
import langid
from sacremoses import MosesDetokenizer
import threading
import os

from .base_model import BaseModel, retry_on_error


class MultilingualModel(BaseModel):
    """
    مدل چندزبانه برای تشخیص زبان و ترجمه متون

    این کلاس از مدل‌های پیشرفته ترجمه ماشینی مانند M2M100 یا mBART
    برای ترجمه متون از زبان‌های مختلف به فارسی استفاده می‌کند.
    همچنین قابلیت تشخیص زبان متن را با استفاده از چندین روش
    برای دقت بیشتر فراهم می‌سازد.
    """

    # زبان‌های پشتیبانی شده برای ترجمه
    SUPPORTED_LANGUAGES = {
        'en': 'انگلیسی',
        'ar': 'عربی',
        'tr': 'ترکی',
        'fr': 'فرانسوی',
        'de': 'آلمانی',
        'es': 'اسپانیایی',
        'ru': 'روسی',
        'zh': 'چینی',
        'ja': 'ژاپنی',
        'ko': 'کره‌ای',
        'it': 'ایتالیایی',
        'nl': 'هلندی',
        'pt': 'پرتغالی',
        'fa': 'فارسی'
    }

    # نگاشت کد زبان‌ها برای مدل‌های مختلف
    LANGUAGE_CODE_MAP = {
        # کد ISO به کد مدل M2M100
        'M2M100': {
            'en': 'en',
            'ar': 'ar',
            'tr': 'tr',
            'fr': 'fr',
            'de': 'de',
            'es': 'es',
            'ru': 'ru',
            'zh': 'zh',
            'ja': 'ja',
            'ko': 'ko',
            'it': 'it',
            'nl': 'nl',
            'pt': 'pt',
            'fa': 'fa'
        },
        # کد ISO به کد مدل mBART
        'MBART': {
            'en': 'en_XX',
            'ar': 'ar_AR',
            'fr': 'fr_XX',
            'de': 'de_DE',
            'es': 'es_XX',
            'ru': 'ru_RU',
            'zh': 'zh_CN',
            'ja': 'ja_XX',
            'ko': 'ko_KR',
            'it': 'it_IT',
            'nl': 'nl_XX',
            'pt': 'pt_XX',
            'fa': 'fa_IR',
            'tr': 'tr_TR'
        }
    }

    def __init__(self) -> None:
        """راه‌اندازی اولیه مدل چندزبانه"""
        super().__init__(model_name="MultilingualModel", priority=10)  # اولویت متوسط

        # پیکربندی‌های خاص مدل
        self.model_config.update({
            'translation_model_type': 'M2M100',  # یا 'MBART'
            'translation_model': 'facebook/m2m100_418M',  # نسخه سبک‌تر برای CPU
            'mbart_model': 'facebook/mbart-large-50-many-to-many-mmt',  # جایگزین M2M100
            'fasttext_model': 'lid.176.bin',
            'max_length': 512,
            'batch_size': 8,
            'quality_threshold': 0.7  # حداقل کیفیت قابل قبول برای ترجمه
        })

        # وضعیت مدل‌های مختلف
        self.language_detector_loaded = False
        self.fasttext_loaded = False
        self.translation_model_loaded = False
        self.detokenizer_loaded = False

        # پیش‌پردازش‌گرهای خاص هر زبان
        self.preprocessors = {}
        self.postprocessors = {}

        # مدیریت حافظه
        self.max_chunk_size = 200  # حداکثر تعداد توکن در هر بخش

        self.logger.debug("مدل چندزبانه راه‌اندازی شد")

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
            # بارگیری حداقل مدل‌ها برای اطمینان از کارکرد صحیح
            self._load_language_detector()

            self.model_loaded = True
            self.logger.info("مدل چندزبانه با موفقیت بارگیری شد")

        except Exception as e:
            self.logger.error(f"خطا در بارگیری مدل چندزبانه: {str(e)}")
            raise

    def _load_language_detector(self) -> None:
        """بارگیری تشخیص‌دهنده زبان"""
        if self.language_detector_loaded:
            return

        self.logger.debug("در حال بارگیری مدل تشخیص زبان...")

        # آماده‌سازی langid برای تشخیص سریع
        langid.set_languages(list(self.SUPPORTED_LANGUAGES.keys()))

        self.language_detector_loaded = True
        self.logger.debug("مدل تشخیص زبان با موفقیت بارگیری شد")

    def _load_fasttext_model(self) -> None:
        """بارگیری مدل FastText برای تشخیص دقیق‌تر زبان"""
        if self.fasttext_loaded:
            return

        self.logger.info("در حال بارگیری مدل FastText برای تشخیص زبان...")

        # مسیر فایل مدل
        model_path = os.path.join(self.model_cache_dir, self.model_config['fasttext_model'])

        # دانلود مدل اگر وجود نداشته باشد
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.logger.info("دانلود مدل FastText...")
            fasttext.util.download_model('lid.176.bin', if_exists='ignore')
            # انتقال فایل به مسیر مدنظر
            if os.path.exists('lid.176.bin'):
                import shutil
                shutil.move('lid.176.bin', model_path)

        # بارگیری مدل
        self.fasttext_model = fasttext.load_model(model_path)

        self.fasttext_loaded = True
        self.logger.info("مدل FastText با موفقیت بارگیری شد")

    def _load_translation_model(self) -> None:
        """بارگیری مدل ترجمه"""
        if self.translation_model_loaded:
            return

        self.logger.info("در حال بارگیری مدل ترجمه...")

        # انتخاب نوع مدل
        if self.model_config['translation_model_type'] == 'M2M100':
            # استفاده از M2M100
            model_name = self.model_config['translation_model']
            self.translation_tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            self.model_language_map = self.LANGUAGE_CODE_MAP['M2M100']
        else:
            # استفاده از mBART
            model_name = self.model_config['mbart_model']
            self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.translation_model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.model_language_map = self.LANGUAGE_CODE_MAP['MBART']

        # انتقال به دستگاه مناسب
        self.translation_model.to(self.device)

        # تنظیم حالت ارزیابی
        self.translation_model.eval()

        # بارگیری توکن‌زدای معکوس
        self._load_detokenizers()

        self.translation_model_loaded = True
        self.logger.info(f"مدل ترجمه {model_name} با موفقیت بارگیری شد")

    def _load_detokenizers(self) -> None:
        """بارگیری توکن‌زداهای مختلف برای زبان‌ها"""
        if self.detokenizer_loaded:
            return

        self.logger.debug("در حال بارگیری توکن‌زداها...")

        self.detokenizers = {}

        # ایجاد توکن‌زدای معکوس برای هر زبان
        for lang in self.SUPPORTED_LANGUAGES:
            if lang != 'fa':  # فارسی نیاز به توکن‌زدای خاصی ندارد
                self.detokenizers[lang] = MosesDetokenizer(lang=lang)

        # پیش‌پردازش‌گرها و پس‌پردازش‌گرهای خاص هر زبان
        self._setup_language_processors()

        self.detokenizer_loaded = True
        self.logger.debug("توکن‌زداها با موفقیت بارگیری شدند")

    def _setup_language_processors(self) -> None:
        """تنظیم پیش‌پردازش‌گرها و پس‌پردازش‌گرهای خاص هر زبان"""
        # پیش‌پردازش‌گرهای خاص هر زبان
        self.preprocessors = {
            'ar': self._preprocess_arabic,
            'zh': self._preprocess_chinese,
            'ja': self._preprocess_japanese,
            # سایر زبان‌ها با پیش‌پردازش پیش‌فرض
        }

        # پس‌پردازش‌گرهای خاص هر زبان
        self.postprocessors = {
            'fa': self._postprocess_persian,
            # سایر زبان‌ها با پس‌پردازش پیش‌فرض
        }

    def _preprocess_arabic(self, text: str) -> str:
        """پیش‌پردازش خاص متن عربی"""
        # نرمال‌سازی کاراکترهای عربی
        # تبدیل برخی کاراکترهای عربی به معادل‌های آنها
        replacements = {
            'ي': 'ی',
            'ك': 'ک',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }

        for arabic, persian in replacements.items():
            text = text.replace(arabic, persian)

        return text

    def _preprocess_chinese(self, text: str) -> str:
        """پیش‌پردازش خاص متن چینی"""
        # اضافه کردن فاصله بین کاراکترهای چینی برای پردازش بهتر
        return ' '.join(text)

    def _preprocess_japanese(self, text: str) -> str:
        """پیش‌پردازش خاص متن ژاپنی"""
        # مشابه چینی، اضافه کردن فاصله بین کاراکترها
        return ' '.join(text)

    def _postprocess_persian(self, text: str) -> str:
        """پس‌پردازش متن فارسی"""
        # اصلاح برخی مشکلات رایج در ترجمه‌های فارسی

        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()

        # اصلاح نیم‌فاصله‌ها
        text = text.replace(' می ', ' می‌')
        text = text.replace(' ای ', '‌ای ')
        text = text.replace(' های ', '‌های ')
        text = text.replace(' ها ', '‌ها ')

        return text

    @retry_on_error()
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        تشخیص زبان متن

        از ترکیب چند روش برای افزایش دقت استفاده می‌کند.

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"language": str, "confidence": float, "language_name": str}
        """
        return self.cached_call(
            "detect_language",
            text,
            self._detect_language_impl
        )

    def _detect_language_impl(self, text: str) -> Dict[str, Any]:
        """
        پیاده‌سازی اصلی تشخیص زبان

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"language": str, "confidence": float, "language_name": str}
        """
        # بررسی متن خالی
        if not text or len(text.strip()) < 2:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "language_name": "نامشخص"
            }

        # تشخیص اولیه با langid (سریع)
        lang, confidence = langid.classify(text)

        # اگر متن کوتاه است یا اطمینان بالاست، همین نتیجه را برمی‌گردانیم
        if len(text) < 100 or confidence > 0.9:
            return {
                "language": lang,
                "confidence": float(confidence),
                "language_name": self.SUPPORTED_LANGUAGES.get(lang, "نامشخص")
            }

        # برای متون بلندتر یا اطمینان کمتر، از FastText استفاده می‌کنیم
        try:
            # بارگیری مدل FastText
            self._load_fasttext_model()

            # تشخیص با FastText
            ft_predictions = self.fasttext_model.predict(text.replace('\n', ' '))
            ft_lang = ft_predictions[0][0].replace('__label__', '')
            ft_confidence = float(ft_predictions[1][0])

            # ترکیب نتایج
            # اگر هر دو روش به یک نتیجه رسیدند، اطمینان را افزایش می‌دهیم
            if lang == ft_lang:
                confidence = max(confidence, ft_confidence) + 0.1 * min(confidence, ft_confidence)
                confidence = min(confidence, 1.0)  # اطمینان نباید از 1 بیشتر شود
            else:
                # اگر نتایج متفاوت است، نتیجه با اطمینان بیشتر را انتخاب می‌کنیم
                if ft_confidence > confidence:
                    lang = ft_lang
                    confidence = ft_confidence

        except Exception as e:
            # اگر خطایی رخ داد، از نتیجه langid استفاده می‌کنیم
            self.logger.warning(f"خطا در تشخیص زبان با FastText: {str(e)}")

        # بررسی کاراکترهای فارسی
        if any('\u0600' <= c <= '\u06FF' for c in text):
            # اگر کاراکترهای فارسی/عربی وجود دارد، زبان را فارسی تشخیص می‌دهیم
            # مگر اینکه با اطمینان بالا زبان عربی تشخیص داده شده باشد
            if lang != 'ar' or confidence < 0.8:
                lang = 'fa'
                confidence = max(confidence, 0.8)

        return {
            "language": lang,
            "confidence": float(confidence),
            "language_name": self.SUPPORTED_LANGUAGES.get(lang, "نامشخص")
        }

    def batch_detect_language(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        تشخیص زبان چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست دیکشنری‌های نتایج
        """
        return self.batch_cached_call(
            "detect_language",
            texts,
            self._batch_detect_language_impl
        )

    def _batch_detect_language_impl(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        پیاده‌سازی اصلی تشخیص زبان دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست دیکشنری‌های نتایج
        """
        # بررسی ورودی خالی
        if not texts:
            return []

        results = []

        # تقسیم کار برای متون طولانی و کوتاه
        short_texts = []
        short_indices = []
        long_texts = []
        long_indices = []

        for i, text in enumerate(texts):
            if not text or len(text.strip()) < 2:
                # متن خالی
                results.append({
                    "language": "unknown",
                    "confidence": 0.0,
                    "language_name": "نامشخص"
                })
            elif len(text) < 100:
                # متن کوتاه
                short_texts.append(text)
                short_indices.append(i)
            else:
                # متن بلند
                long_texts.append(text)
                long_indices.append(i)

        # پر کردن نتایج با None برای جای‌گذاری بعدی
        for _ in range(len(short_texts) + len(long_texts)):
            results.append(None)

        # تشخیص زبان متون کوتاه با langid
        if short_texts:
            langid_results = langid.classify_many(short_texts)

            for idx, (i, (lang, confidence)) in enumerate(zip(short_indices, langid_results)):
                results[i] = {
                    "language": lang,
                    "confidence": float(confidence),
                    "language_name": self.SUPPORTED_LANGUAGES.get(lang, "نامشخص")
                }

        # تشخیص زبان متون بلند با FastText
        if long_texts:
            try:
                # بارگیری مدل FastText
                self._load_fasttext_model()

                for idx, (i, text) in enumerate(zip(long_indices, long_texts)):
                    # تشخیص با langid
                    lang, confidence = langid.classify(text)

                    # تشخیص با FastText
                    ft_predictions = self.fasttext_model.predict(text.replace('\n', ' '))
                    ft_lang = ft_predictions[0][0].replace('__label__', '')
                    ft_confidence = float(ft_predictions[1][0])

                    # ترکیب نتایج
                    if lang == ft_lang:
                        confidence = max(confidence, ft_confidence) + 0.1 * min(confidence, ft_confidence)
                        confidence = min(confidence, 1.0)
                    else:
                        if ft_confidence > confidence:
                            lang = ft_lang
                            confidence = ft_confidence

                    # بررسی کاراکترهای فارسی
                    if any('\u0600' <= c <= '\u06FF' for c in text):
                        if lang != 'ar' or confidence < 0.8:
                            lang = 'fa'
                            confidence = max(confidence, 0.8)

                    results[i] = {
                        "language": lang,
                        "confidence": float(confidence),
                        "language_name": self.SUPPORTED_LANGUAGES.get(lang, "نامشخص")
                    }

            except Exception as e:
                # در صورت خطا، از langid استفاده می‌کنیم
                self.logger.warning(f"خطا در تشخیص دسته‌ای زبان با FastText: {str(e)}")

                langid_results = langid.classify_many(long_texts)

                for idx, (i, (lang, confidence)) in enumerate(zip(long_indices, langid_results)):
                    text = long_texts[idx]

                    # بررسی کاراکترهای فارسی
                    if any('\u0600' <= c <= '\u06FF' for c in text):
                        if lang != 'ar' or confidence < 0.8:
                            lang = 'fa'
                            confidence = max(confidence, 0.8)

                    results[i] = {
                        "language": lang,
                        "confidence": float(confidence),
                        "language_name": self.SUPPORTED_LANGUAGES.get(lang, "نامشخص")
                    }

        return results

    @retry_on_error()
    def translate_to_persian(self, text: str, source_lang: Optional[str] = None) -> str:
        """
        ترجمه متن به فارسی

        Args:
            text: متن ورودی
            source_lang: زبان مبدا (اختیاری، در صورت عدم تعیین، خودکار تشخیص داده می‌شود)

        Returns:
            متن ترجمه شده به فارسی
        """
        return self.cached_call(
            "translate_to_persian",
            f"{source_lang or 'auto'}|{text[:100]}",  # کلید کش
            lambda x: self._translate_to_persian_impl(text, source_lang)
        )

    def _translate_to_persian_impl(self, text: str, source_lang: Optional[str] = None) -> str:
        """
        پیاده‌سازی اصلی ترجمه به فارسی

        Args:
            text: متن ورودی
            source_lang: زبان مبدا (اختیاری)

        Returns:
            متن ترجمه شده به فارسی
        """
        # پیش‌پردازش متن
        text = text.strip()
        if not text:
            return ""

        # تشخیص زبان اگر مشخص نشده باشد
        if not source_lang:
            lang_info = self.detect_language(text)
            source_lang = lang_info["language"]

        # اگر متن فارسی است، نیازی به ترجمه نیست
        if source_lang == "fa":
            return text

        # اگر زبان پشتیبانی نمی‌شود، خطا می‌دهیم
        if source_lang not in self.SUPPORTED_LANGUAGES:
            self.logger.warning(f"زبان '{source_lang}' برای ترجمه پشتیبانی نمی‌شود")
            return text

        # بارگیری مدل ترجمه
        self._load_translation_model()

        # پیش‌پردازش خاص زبان
        if source_lang in self.preprocessors:
            text = self.preprocessors[source_lang](text)

        # تبدیل کد زبان به فرمت مناسب مدل
        src_lang_code = self.model_language_map.get(source_lang, source_lang)
        tgt_lang_code = self.model_language_map.get('fa', 'fa')

        # تنظیم مدل برای زبان مبدا
        if self.model_config['translation_model_type'] == 'M2M100':
            self.translation_tokenizer.src_lang = src_lang_code
        else:  # mBART
            self.translation_tokenizer.src_lang = src_lang_code

        # تقسیم متن به بخش‌های قابل مدیریت
        chunks = self._split_text_into_chunks(text, source_lang)
        translated_chunks = []

        # ترجمه هر بخش
        for chunk in chunks:
            # توکنایز کردن
            inputs = self.translation_tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_length']
            )

            # انتقال به دستگاه مناسب
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # ترجمه
            if self.model_config['translation_model_type'] == 'M2M100':
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.get_lang_id('fa'),
                    max_length=self.model_config['max_length'],
                    num_beams=5,  # استفاده از beam search برای نتایج بهتر
                    length_penalty=1.0  # تشویق به تولید متن‌های بلندتر
                )
            else:  # mBART
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang_code],
                    max_length=self.model_config['max_length'],
                    num_beams=5,
                    length_penalty=1.0
                )

            # تبدیل توکن‌ها به متن
            translated_chunk = self.translation_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            translated_chunks.append(translated_chunk)

        # ترکیب بخش‌ها
        translated_text = " ".join(translated_chunks)

        # پس‌پردازش فارسی
        if 'fa' in self.postprocessors:
            translated_text = self.postprocessors['fa'](translated_text)

        return translated_text

    def batch_translate_to_persian(self, texts: List[str], source_langs: Optional[List[str]] = None) -> List[str]:
        """
        ترجمه چندین متن به فارسی به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی
            source_langs: لیست زبان‌های مبدا (اختیاری)

        Returns:
            لیست متون ترجمه شده
        """
        # بررسی ورودی خالی
        if not texts:
            return []

        # ایجاد لیست زبان‌های مبدا اگر ارائه نشده است
        if not source_langs:
            # تشخیص زبان همه متون
            lang_results = self.batch_detect_language(texts)
            source_langs = [result["language"] for result in lang_results]

        # ساخت کلیدهای کش برای تابع batch_cached_call
        cache_keys = []
        for i, text in enumerate(texts):
            source_lang = source_langs[i] if i < len(source_langs) else 'auto'
            cache_keys.append(f"{source_lang}|{text[:100]}")

        # استفاده از batch_cached_call با کلیدهای ساخته شده
        return self.batch_cached_call(
            "translate_to_persian",
            cache_keys,
            lambda k: self._batch_translate_to_persian_impl(texts, source_langs)
        )

    def _batch_translate_to_persian_impl(self, texts: List[str], source_langs: List[str]) -> List[str]:
        """
        پیاده‌سازی اصلی ترجمه دسته‌ای به فارسی

        Args:
            texts: لیست متون ورودی
            source_langs: لیست زبان‌های مبدا

        Returns:
            لیست متون ترجمه شده
        """
        # بررسی ورودی خالی
        if not texts:
            return []

        # بارگیری مدل ترجمه
        self._load_translation_model()

        results = []
        batch_size = self.model_config['batch_size']

        # گروه‌بندی متون بر اساس زبان برای پردازش بهینه‌تر
        lang_groups = {}

        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                results.append("")
                continue

            source_lang = source_langs[i] if i < len(source_langs) else 'auto'

            # اگر زبان خودکار است، تشخیص زبان
            if source_lang == 'auto':
                lang_info = self.detect_language(text)
                source_lang = lang_info["language"]

            # اگر متن فارسی است، نیازی به ترجمه نیست
            if source_lang == "fa":
                results.append(text)
                continue

            # اگر زبان پشتیبانی نمی‌شود، متن اصلی را برمی‌گردانیم
            if source_lang not in self.SUPPORTED_LANGUAGES:
                self.logger.warning(f"زبان '{source_lang}' برای ترجمه پشتیبانی نمی‌شود")
                results.append(text)
                continue

            # گروه‌بندی بر اساس زبان
            if source_lang not in lang_groups:
                lang_groups[source_lang] = []

            # پیش‌پردازش خاص زبان
            if source_lang in self.preprocessors:
                text = self.preprocessors[source_lang](text)

            # تقسیم به بخش‌ها اگر متن بلند است
            chunks = self._split_text_into_chunks(text, source_lang)
            lang_groups[source_lang].extend(chunks)

        # ترجمه گروه‌های زبانی
        translated_chunks = {}

        for source_lang, chunks in lang_groups.items():
            # تبدیل کد زبان به فرمت مناسب مدل
            src_lang_code = self.model_language_map.get(source_lang, source_lang)
            tgt_lang_code = self.model_language_map.get('fa', 'fa')

            # تنظیم مدل برای زبان مبدا
            if self.model_config['translation_model_type'] == 'M2M100':
                self.translation_tokenizer.src_lang = src_lang_code
            else:  # mBART
                self.translation_tokenizer.src_lang = src_lang_code

            translated_chunks[source_lang] = []

            # پردازش بخش‌ها در دسته‌های کوچک
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # توکنایز کردن
                inputs = self.translation_tokenizer(
                    batch_chunks,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_config['max_length']
                )

                # انتقال به دستگاه مناسب
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # ترجمه
                if self.model_config['translation_model_type'] == 'M2M100':
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        forced_bos_token_id=self.translation_tokenizer.get_lang_id('fa'),
                        max_length=self.model_config['max_length'],
                        num_beams=3,  # کاهش beam search برای سرعت بیشتر در پردازش دسته‌ای
                        length_penalty=1.0
                    )
                else:  # mBART
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang_code],
                        max_length=self.model_config['max_length'],
                        num_beams=3,
                        length_penalty=1.0
                    )

                # تبدیل توکن‌ها به متن
                batch_translations = self.translation_tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                # افزودن به نتایج
                translated_chunks[source_lang].extend(batch_translations)

        # بازسازی نتایج در ترتیب اصلی
        chunk_index = {lang: 0 for lang in lang_groups}

        for i, text in enumerate(texts):
            if i >= len(results):  # یعنی قبلاً افزوده نشده است
                text = text.strip()
                if not text:
                    results.append("")
                    continue

                source_lang = source_langs[i] if i < len(source_langs) else 'auto'

                # اگر زبان خودکار است، احتمالاً قبلاً تشخیص داده شده
                if source_lang == 'auto':
                    lang_info = self.detect_language(text)
                    source_lang = lang_info["language"]

                # اگر فارسی است یا پشتیبانی نمی‌شود، قبلاً افزوده شده است
                if source_lang == "fa" or source_lang not in self.SUPPORTED_LANGUAGES:
                    continue

                # محاسبه تعداد بخش‌های متن
                chunks = self._split_text_into_chunks(text, source_lang)
                chunk_count = len(chunks)

                # ایجاد ترجمه کامل با ترکیب بخش‌ها
                if chunk_count == 1:
                    # اگر فقط یک بخش داشت
                    translation = translated_chunks[source_lang][chunk_index[source_lang]]
                    chunk_index[source_lang] += 1
                else:
                    # اگر چند بخش داشت
                    translations = []
                    for _ in range(chunk_count):
                        if chunk_index[source_lang] < len(translated_chunks[source_lang]):
                            translations.append(translated_chunks[source_lang][chunk_index[source_lang]])
                            chunk_index[source_lang] += 1

                    translation = " ".join(translations)

                # پس‌پردازش فارسی
                if 'fa' in self.postprocessors:
                    translation = self.postprocessors['fa'](translation)

                results.append(translation)

        return results

    def translate_between_languages(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        ترجمه متن بین دو زبان

        Args:
            text: متن ورودی
            source_lang: زبان مبدا
            target_lang: زبان مقصد

        Returns:
            متن ترجمه شده
        """
        return self.cached_call(
            "translate_between_languages",
            f"{source_lang}|{target_lang}|{text[:100]}",  # کلید کش
            lambda x: self._translate_between_languages_impl(text, source_lang, target_lang)
        )

    def _translate_between_languages_impl(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        پیاده‌سازی اصلی ترجمه بین دو زبان

        Args:
            text: متن ورودی
            source_lang: زبان مبدا
            target_lang: زبان مقصد

        Returns:
            متن ترجمه شده
        """
        # پیش‌پردازش متن
        text = text.strip()
        if not text:
            return ""

        # بررسی زبان‌های پشتیبانی شده
        if source_lang not in self.SUPPORTED_LANGUAGES:
            self.logger.warning(f"زبان مبدا '{source_lang}' پشتیبانی نمی‌شود")
            return text

        if target_lang not in self.SUPPORTED_LANGUAGES:
            self.logger.warning(f"زبان مقصد '{target_lang}' پشتیبانی نمی‌شود")
            return text

        # اگر زبان‌ها یکسان هستند، نیازی به ترجمه نیست
        if source_lang == target_lang:
            return text

        # بارگیری مدل ترجمه
        self._load_translation_model()

        # پیش‌پردازش خاص زبان
        if source_lang in self.preprocessors:
            text = self.preprocessors[source_lang](text)

        # تبدیل کد زبان به فرمت مناسب مدل
        src_lang_code = self.model_language_map.get(source_lang, source_lang)
        tgt_lang_code = self.model_language_map.get(target_lang, target_lang)

        # تنظیم مدل برای زبان مبدا
        if self.model_config['translation_model_type'] == 'M2M100':
            self.translation_tokenizer.src_lang = src_lang_code
        else:  # mBART
            self.translation_tokenizer.src_lang = src_lang_code

        # تقسیم متن به بخش‌های قابل مدیریت
        chunks = self._split_text_into_chunks(text, source_lang)
        translated_chunks = []

        # ترجمه هر بخش
        for chunk in chunks:
            # توکنایز کردن
            inputs = self.translation_tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_length']
            )

            # انتقال به دستگاه مناسب
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # ترجمه
            if self.model_config['translation_model_type'] == 'M2M100':
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.get_lang_id(target_lang),
                    max_length=self.model_config['max_length'],
                    num_beams=5,
                    length_penalty=1.0
                )
            else:  # mBART
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[tgt_lang_code],
                    max_length=self.model_config['max_length'],
                    num_beams=5,
                    length_penalty=1.0
                )

            # تبدیل توکن‌ها به متن
            translated_chunk = self.translation_tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            translated_chunks.append(translated_chunk)

        # ترکیب بخش‌ها
        translated_text = " ".join(translated_chunks)

        # پس‌پردازش خاص زبان
        if target_lang in self.postprocessors:
            translated_text = self.postprocessors[target_lang](translated_text)

        return translated_text

    def _split_text_into_chunks(self, text: str, language: str) -> List[str]:
        """
        تقسیم متن به بخش‌های کوچکتر برای مدیریت بهتر حافظه

        تقسیم هوشمند بر اساس مرزهای جمله انجام می‌شود.

        Args:
            text: متن ورودی
            language: زبان متن

        Returns:
            لیست بخش‌ها
        """
        # اگر متن کوتاه است، تقسیم نمی‌کنیم
        if len(text) <= self.model_config['max_length'] * 0.8:
            return [text]

        chunks = []

        # تشخیص الگوی پایان جمله بر اساس زبان
        if language in ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl']:
            # زبان‌های لاتین
            sentence_end_pattern = r'(?<=[.!?])\s+'
        elif language in ['zh', 'ja', 'ko']:
            # زبان‌های شرق آسیا
            sentence_end_pattern = r'(?<=[。！？])'
        elif language in ['ar', 'fa']:
            # زبان‌های عربی و فارسی
            sentence_end_pattern = r'(?<=[.!?؟])\s+'
        elif language == 'ru':
            # روسی
            sentence_end_pattern = r'(?<=[.!?])\s+'
        else:
            # پیش‌فرض
            sentence_end_pattern = r'(?<=[.!?])\s+'

        # تقسیم متن به جملات
        sentences = re.split(sentence_end_pattern, text)

        # حذف جملات خالی
        sentences = [s.strip() for s in sentences if s.strip()]

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # تخمین طول توکن‌های جمله (تخمین ساده)
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.max_chunk_size:
                # اگر افزودن جمله جدید باعث افزایش بیش از حد طول بخش شود
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # اگر جمله خیلی طولانی است، آن را اضافه می‌کنیم و به بخش بعدی می‌رویم
                    chunks.append(sentence)
            else:
                # افزودن جمله به بخش فعلی
                current_chunk.append(sentence)
                current_length += sentence_length

        # افزودن آخرین بخش
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_supported_languages(self) -> Dict[str, str]:
        """
        دریافت لیست زبان‌های پشتیبانی شده

        Returns:
            دیکشنری {کد زبان: نام زبان}
        """
        return self.SUPPORTED_LANGUAGES.copy()

    def analyze_translation_quality(self, original_text: str, translated_text: str) -> Dict[str, Any]:
        """
        تحلیل کیفیت ترجمه

        Args:
            original_text: متن اصلی
            translated_text: متن ترجمه شده

        Returns:
            دیکشنری با معیارهای مختلف کیفیت
        """
        # بررسی متون خالی
        if not original_text or not translated_text:
            return {
                "quality_score": 0.0,
                "length_ratio": 0.0,
                "is_acceptable": False,
                "message": "متن خالی"
            }

        # تشخیص زبان متن اصلی
        lang_info = self.detect_language(original_text)
        source_lang = lang_info["language"]

        # نسبت طول متن ترجمه شده به متن اصلی
        original_length = len(original_text.split())
        translated_length = len(translated_text.split())

        # نسبت مناسب طول بر اساس زبان
        if source_lang in ['zh', 'ja', 'ko']:
            # زبان‌های شرق آسیا معمولاً در ترجمه طولانی‌تر می‌شوند
            min_ratio = 1.5
            max_ratio = 3.5
        elif source_lang in ['ar', 'fa']:
            # زبان‌های عربی و فارسی
            min_ratio = 0.7
            max_ratio = 1.3
        elif source_lang in ['de', 'ru']:
            # زبان‌های با کلمات طولانی
            min_ratio = 0.6
            max_ratio = 1.5
        else:
            # سایر زبان‌ها
            min_ratio = 0.5
            max_ratio = 2.0

        # محاسبه نسبت طول
        if original_length > 0:
            length_ratio = translated_length / original_length
        else:
            length_ratio = 0

        # ارزیابی کیفیت بر اساس نسبت طول
        if min_ratio <= length_ratio <= max_ratio:
            ratio_score = 1.0
        else:
            # محاسبه فاصله از بازه مطلوب
            distance = min(abs(length_ratio - min_ratio), abs(length_ratio - max_ratio))
            ratio_score = max(0.0, 1.0 - distance / min_ratio)

        # بررسی وجود کاراکترهای خاص در متن ترجمه شده
        special_char_ratio = len(re.findall(r'[«»؛،{}()\[\]"\']', translated_text)) / max(1, len(translated_text))
        special_char_score = 1.0 if special_char_ratio < 0.1 else max(0.0, 1.0 - (special_char_ratio - 0.1) * 5)

        # بررسی کلمات تکراری متوالی (نشانه خطا)
        repeated_words = re.findall(r'\b(\w+)(\s+\1){2,}\b', translated_text.lower())
        repetition_score = 1.0 if len(repeated_words) == 0 else max(0.0, 1.0 - len(repeated_words) * 0.2)

        # محاسبه امتیاز کلی کیفیت
        quality_score = (ratio_score * 0.5) + (special_char_score * 0.3) + (repetition_score * 0.2)

        # تعیین قابل قبول بودن
        is_acceptable = quality_score >= self.model_config['quality_threshold']

        # تولید پیام
        if not is_acceptable:
            if ratio_score < 0.5:
                message = "نسبت طول متن ترجمه شده مناسب نیست"
            elif special_char_score < 0.5:
                message = "استفاده بیش از حد از کاراکترهای خاص"
            elif repetition_score < 0.5:
                message = "تکرار کلمات متوالی در متن ترجمه شده"
            else:
                message = "کیفیت ترجمه پایین است"
        else:
            message = "کیفیت ترجمه قابل قبول است"

        return {
            "quality_score": round(quality_score, 2),
            "length_ratio": round(length_ratio, 2),
            "is_acceptable": is_acceptable,
            "message": message
        }

    def translate_with_fallback(self, text: str, source_lang: Optional[str] = None,
                                target_lang: str = 'fa') -> Dict[str, Any]:
        """
        ترجمه با سیستم پشتیبان در صورت شکست

        این متد از سیستم‌های مختلف ترجمه استفاده می‌کند و در صورت
        شکست یکی، از دیگری استفاده می‌کند.

        Args:
            text: متن ورودی
            source_lang: زبان مبدا (اختیاری)
            target_lang: زبان مقصد (پیش‌فرض: فارسی)

        Returns:
            دیکشنری {"translation": str, "quality": dict, "method": str}
        """
        # تشخیص زبان اگر مشخص نشده باشد
        if not source_lang:
            lang_info = self.detect_language(text)
            source_lang = lang_info["language"]
            source_confidence = lang_info["confidence"]
        else:
            source_confidence = 1.0

        # اگر زبان مبدا و مقصد یکسان هستند
        if source_lang == target_lang:
            return {
                "translation": text,
                "quality": {
                    "quality_score": 1.0,
                    "is_acceptable": True,
                    "message": "زبان مبدا و مقصد یکسان هستند"
                },
                "method": "direct"
            }

        # روش 1: ترجمه مستقیم با مدل اصلی
        try:
            if target_lang == 'fa':
                translation = self.translate_to_persian(text, source_lang)
                method = "direct_m2m"
            else:
                translation = self.translate_between_languages(text, source_lang, target_lang)
                method = "direct_m2m"

            # بررسی کیفیت
            quality = self.analyze_translation_quality(text, translation)

            # اگر کیفیت قابل قبول است، همین نتیجه را برمی‌گردانیم
            if quality["is_acceptable"]:
                return {
                    "translation": translation,
                    "quality": quality,
                    "method": method
                }

        except Exception as e:
            self.logger.warning(f"خطا در ترجمه مستقیم: {str(e)}")
            translation = ""
            quality = {"quality_score": 0.0, "is_acceptable": False, "message": f"خطا: {str(e)}"}
            method = "failed"

        # روش 2: تغییر نوع مدل ترجمه
        if not quality["is_acceptable"]:
            try:
                # تغییر نوع مدل
                original_model_type = self.model_config['translation_model_type']

                if original_model_type == 'M2M100':
                    self.model_config['translation_model_type'] = 'MBART'
                else:
                    self.model_config['translation_model_type'] = 'M2M100'

                # تخلیه مدل فعلی
                self.unload_model()
                self.translation_model_loaded = False

                # ترجمه با مدل جدید
                if target_lang == 'fa':
                    translation_alt = self.translate_to_persian(text, source_lang)
                    method = "fallback_model_switch"
                else:
                    translation_alt = self.translate_between_languages(text, source_lang, target_lang)
                    method = "fallback_model_switch"

                # بررسی کیفیت
                quality_alt = self.analyze_translation_quality(text, translation_alt)

                # مقایسه کیفیت
                if quality_alt["quality_score"] > quality["quality_score"]:
                    translation = translation_alt
                    quality = quality_alt

                # بازگرداندن تنظیمات به حالت اول
                self.model_config['translation_model_type'] = original_model_type
                self.unload_model()
                self.translation_model_loaded = False

            except Exception as e:
                self.logger.warning(f"خطا در ترجمه جایگزین: {str(e)}")
                # بازگرداندن تنظیمات به حالت اول
                self.model_config['translation_model_type'] = original_model_type
                self.unload_model()
                self.translation_model_loaded = False

        # روش 3: ترجمه از طریق زبان واسط (پل زدن)
        if not quality["is_acceptable"] and source_confidence > 0.7:
            try:
                # انتخاب زبان واسط
                bridge_lang = 'en'  # انگلیسی به عنوان زبان واسط

                # ترجمه به زبان واسط
                bridge_text = self.translate_between_languages(text, source_lang, bridge_lang)

                # ترجمه از زبان واسط به مقصد
                translation_bridge = self.translate_between_languages(bridge_text, bridge_lang, target_lang)

                # بررسی کیفیت
                quality_bridge = self.analyze_translation_quality(text, translation_bridge)

                # مقایسه کیفیت
                if quality_bridge["quality_score"] > quality["quality_score"]:
                    translation = translation_bridge
                    quality = quality_bridge
                    method = "bridge_translation"

            except Exception as e:
                self.logger.warning(f"خطا در ترجمه پل: {str(e)}")

        return {
            "translation": translation,
            "quality": quality,
            "method": method
        }
