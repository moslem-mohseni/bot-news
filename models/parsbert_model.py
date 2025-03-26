"""
ماژول ParsBERT برای CryptoNewsBot

این ماژول مسئول پردازش متون فارسی با استفاده از مدل ParsBERT و ابزارهای پردازش زبان فارسی
است. قابلیت‌های اصلی آن شامل تحلیل احساسات، استخراج کلمات کلیدی، خلاصه‌سازی،
دسته‌بندی متن و تولید امبدینگ است.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel, pipeline, BertForSequenceClassification
import hazm
from sklearn.metrics.pairwise import cosine_similarity

from .base_model import BaseModel, retry_on_error


class ParsBertModel(BaseModel):
    """
    مدل ParsBERT برای پردازش متون فارسی

    این کلاس از کتابخانه Transformers و Hazm برای پردازش متون فارسی استفاده می‌کند
    و قابلیت‌های پیشرفته‌ای مانند تحلیل احساسات، استخراج کلمات کلیدی و خلاصه‌سازی
    را فراهم می‌کند.
    """

    def __init__(self) -> None:
        """راه‌اندازی اولیه مدل ParsBERT"""
        super().__init__(model_name="ParsBERT", priority=5)  # اولویت بالا (کمتر = مهمتر)

        # پیکربندی‌های خاص مدل
        self.model_config.update({
            'embedding_model': 'HooshvareLab/bert-fa-zwnj-base',
            'sentiment_model': 'm3hrdadfi/albert-fa-base-v2-sentiment-multi',
            'classification_model': 'HooshvareLab/bert-fa-zwnj-base-uncased',
            'max_length': 512,
            'batch_size': 16
        })

        # وضعیت مدل‌های مختلف
        self.hazm_loaded = False
        self.embedding_model_loaded = False
        self.sentiment_model_loaded = False
        self.classification_model_loaded = False

        # دسته‌بندی‌های خبری ارز دیجیتال
        self.news_categories = {
            'PRICE': 'قیمت و بازار',
            'REGULATION': 'قوانین و مقررات',
            'ADOPTION': 'پذیرش و استفاده',
            'TECHNOLOGY': 'تکنولوژی و توسعه',
            'SECURITY': 'امنیت و هک',
            'EXCHANGE': 'صرافی‌ها',
            'GENERAL': 'عمومی',
            'OTHER': 'سایر'
        }

        self.logger.debug("مدل ParsBERT راه‌اندازی شد")

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
            self._load_hazm()
            self._load_embedding_model()

            self.model_loaded = True
            self.logger.info("مدل ParsBERT با موفقیت بارگیری شد")

        except Exception as e:
            self.logger.error(f"خطا در بارگیری مدل ParsBERT: {str(e)}")
            raise

    def _load_hazm(self) -> None:
        """بارگیری و پیکربندی ابزارهای Hazm برای پردازش متن فارسی"""
        if self.hazm_loaded:
            return

        self.logger.debug("در حال بارگیری ابزارهای Hazm...")

        # ایجاد نمونه‌های ابزارهای Hazm
        self.normalizer = hazm.Normalizer()
        self.stemmer = hazm.Stemmer()
        self.lemmatizer = hazm.Lemmatizer()
        self.sent_tokenizer = hazm.SentenceTokenizer()
        self.word_tokenizer = hazm.WordTokenizer()

        # بارگیری POS Tagger (برچسب‌گذاری اجزای سخن) فقط در صورت نیاز
        self.tagger = None  # بارگیری تنبل

        # ایجاد لیست کلمات ایست (stop words)
        self._generate_stopwords()

        self.hazm_loaded = True
        self.logger.debug("ابزارهای Hazm با موفقیت بارگیری شدند")

    def _load_pos_tagger(self) -> None:
        """بارگیری برچسب‌گذار اجزای سخن (POS Tagger)"""
        if self.tagger is not None:
            return

        self.logger.debug("در حال بارگیری برچسب‌گذار اجزای سخن Hazm...")
        self.tagger = hazm.POSTagger(model=f"{self.model_cache_dir}/resources/postagger.model")
        self.logger.debug("برچسب‌گذار اجزای سخن Hazm با موفقیت بارگیری شد")

    def _generate_stopwords(self) -> None:
        """ایجاد لیست جامع کلمات ایست فارسی"""
        # لیست پایه کلمات ایست
        basic_stopwords = hazm.stopwords_list()

        # کلمات ایست اضافی مرتبط با اخبار و ارزهای دیجیتال
        additional_stopwords = [
            "گفت", "اعلام", "کرد", "شد", "می‌شود", "است", "خواهد", "داد", "دارد", "کرده",
            "گزارش", "اظهار", "بیان", "اخبار", "گزارش", "روز", "امروز", "دیروز", "فردا", "هفته",
            "ماه", "سال", "قیمت", "ارز", "دیجیتال", "رمز", "رمزارز", "بازار", "قیمت‌ها",
            "افزایش", "کاهش", "رشد", "سقوط", "معامله", "معاملات", "خرید", "فروش", "کوین",
            "ارزش", "بیت", "اتر", "بیت‌کوین", "اتریوم", "تتر", "دلار", "تومان", "ریال"
        ]

        # ترکیب و یکتاسازی
        self.stopwords = set(basic_stopwords + additional_stopwords)

    def _load_embedding_model(self) -> None:
        """بارگیری مدل امبدینگ متن"""
        if self.embedding_model_loaded:
            return

        self.logger.info("در حال بارگیری مدل امبدینگ ParsBERT...")

        # بارگیری مدل و توکنایزر از HuggingFace
        model_name = self.model_config['embedding_model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # انتقال به دستگاه مناسب (CPU/GPU)
        self.model.to(self.device)

        # تنظیم حالت ارزیابی
        self.model.eval()

        self.embedding_model_loaded = True
        self.logger.info(f"مدل امبدینگ {model_name} با موفقیت بارگیری شد")

    def _load_sentiment_model(self) -> None:
        """بارگیری مدل تحلیل احساسات"""
        if self.sentiment_model_loaded:
            return

        self.logger.info("در حال بارگیری مدل تحلیل احساسات...")

        # بارگیری مدل احساسات از HuggingFace با pipeline
        model_name = self.model_config['sentiment_model']
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device.type == 'cuda' else -1  # استفاده از GPU در صورت وجود
        )

        self.sentiment_model_loaded = True
        self.logger.info(f"مدل تحلیل احساسات {model_name} با موفقیت بارگیری شد")

    def _load_classification_model(self) -> None:
        """بارگیری مدل دسته‌بندی متن"""
        if self.classification_model_loaded:
            return

        self.logger.info("در حال بارگیری مدل دسته‌بندی متن...")

        # بارگیری مدل دسته‌بندی (می‌توان همان مدل پایه را تنظیم کرد)
        model_name = self.model_config['classification_model']
        self.classification_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classification_model = AutoModel.from_pretrained(model_name)

        # انتقال به دستگاه مناسب
        self.classification_model.to(self.device)

        # تنظیم حالت ارزیابی
        self.classification_model.eval()

        self.classification_model_loaded = True
        self.logger.info(f"مدل دسته‌بندی {model_name} با موفقیت بارگیری شد")

    def preprocess_text(self, text: str) -> str:
        """
        پیش‌پردازش متن فارسی

        Args:
            text: متن ورودی

        Returns:
            متن پیش‌پردازش شده
        """
        # بارگیری ابزارهای Hazm اگر بارگیری نشده باشند
        self._load_hazm()

        # پیش‌پردازش اولیه
        if not text:
            return ""

        # نرمال‌سازی
        normalized_text = self.normalizer.normalize(text)

        # حذف کاراکترهای خاص و اعداد
        normalized_text = re.sub(r'[^\w\s\u0600-\u06FF.]', ' ', normalized_text)
        normalized_text = re.sub(r'\d+', ' ', normalized_text)

        # حذف فاصله‌های اضافی
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

        return normalized_text

    @retry_on_error()
    def get_text_embedding(self, text: str) -> List[float]:
        """
        دریافت امبدینگ متن

        متن را به یک بردار عددی تبدیل می‌کند که برای مقایسه معنایی
        و مدل‌سازی دیگر قابل استفاده است.

        Args:
            text: متن ورودی

        Returns:
            لیست اعداد اعشاری (امبدینگ)
        """
        return self.cached_call(
            "get_text_embedding",
            text,
            self._compute_text_embedding
        )

    def _compute_text_embedding(self, text: str) -> List[float]:
        """
        محاسبه امبدینگ متن

        Args:
            text: متن ورودی

        Returns:
            لیست اعداد اعشاری (امبدینگ)
        """
        # پیش‌پردازش متن
        text = self.preprocess_text(text)
        if not text:
            return [0.0] * 768  # امبدینگ صفر برای متن خالی

        # بارگیری مدل
        self._load_embedding_model()

        # توکنایز کردن متن
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config['max_length']
        )

        # انتقال به دستگاه مناسب
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # محاسبه امبدینگ
        with torch.no_grad():
            outputs = self.model(**inputs)

        # استخراج امبدینگ از لایه آخر (CLS token)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # تبدیل به لیست
        return embeddings[0].tolist()

    def batch_get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        دریافت امبدینگ‌های چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست امبدینگ‌ها
        """
        return self.batch_cached_call(
            "get_text_embedding",
            texts,
            self._compute_batch_text_embeddings
        )

    def _compute_batch_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        محاسبه امبدینگ‌های چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست امبدینگ‌ها
        """
        # بررسی ورودی خالی
        if not texts:
            return []

        # پیش‌پردازش متون
        processed_texts = [self.preprocess_text(text) for text in texts]

        # جایگزینی متون خالی با فضای خالی
        processed_texts = [text if text else " " for text in processed_texts]

        # بارگیری مدل
        self._load_embedding_model()

        # تقسیم به دسته‌های کوچکتر برای مدیریت حافظه
        batch_size = self.model_config['batch_size']
        all_embeddings = []

        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]

            # توکنایز کردن متون
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config['max_length']
            )

            # انتقال به دستگاه مناسب
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # محاسبه امبدینگ‌ها
            with torch.no_grad():
                outputs = self.model(**inputs)

            # استخراج امبدینگ‌ها از لایه آخر (CLS token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # افزودن به نتایج
            all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings

    @retry_on_error()
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        تحلیل احساسات متن

        تعیین می‌کند که متن مثبت، منفی یا خنثی است و میزان
        اطمینان برای هر دسته را مشخص می‌کند.

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"sentiment": str, "positive": float, "negative": float, "neutral": float}
        """
        return self.cached_call(
            "analyze_sentiment",
            text,
            self._compute_sentiment
        )

    def _compute_sentiment(self, text: str) -> Dict[str, Any]:
        """
        محاسبه احساسات متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"sentiment": str, "positive": float, "negative": float, "neutral": float}
        """
        # پیش‌پردازش متن
        text = self.preprocess_text(text)
        if not text:
            return {
                "sentiment": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "score": 0.0
            }

        # بارگیری مدل احساسات
        self._load_sentiment_model()

        # تحلیل احساسات
        result = self.sentiment_analyzer(text)[0]
        label = result["label"]
        score = result["score"]

        # نگاشت برچسب‌ها به مقادیر مثبت/منفی/خنثی
        sentiment_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral",
            "POSITIVE": "positive",
            # افزودن سایر برچسب‌های احتمالی
        }

        sentiment = sentiment_mapping.get(label, "neutral")

        # ایجاد دیکشنری نتیجه
        result_dict = {
            "sentiment": sentiment,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "score": float(score)
        }

        # تنظیم مقدار احساس تشخیص داده شده
        result_dict[sentiment] = float(score)

        # تنظیم سایر مقادیر به گونه‌ای که مجموع 1 شود
        if sentiment == "positive":
            result_dict["neutral"] = (1.0 - score) * 0.7
            result_dict["negative"] = (1.0 - score) * 0.3
        elif sentiment == "negative":
            result_dict["neutral"] = (1.0 - score) * 0.7
            result_dict["positive"] = (1.0 - score) * 0.3
        else:  # neutral
            result_dict["positive"] = (1.0 - score) * 0.5
            result_dict["negative"] = (1.0 - score) * 0.5

        return result_dict

    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        تحلیل احساسات چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست دیکشنری‌های نتایج
        """
        return self.batch_cached_call(
            "analyze_sentiment",
            texts,
            self._compute_batch_sentiment
        )

    def _compute_batch_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        محاسبه احساسات چندین متن به صورت دسته‌ای

        Args:
            texts: لیست متون ورودی

        Returns:
            لیست دیکشنری‌های نتایج
        """
        # بررسی ورودی خالی
        if not texts:
            return []

        # پیش‌پردازش متون
        processed_texts = [self.preprocess_text(text) if text else " " for text in texts]

        # بارگیری مدل احساسات
        self._load_sentiment_model()

        # تحلیل احساسات
        results = self.sentiment_analyzer(processed_texts)

        # نگاشت برچسب‌ها و فرمت‌بندی نتایج
        sentiment_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral",
            "POSITIVE": "positive",
        }

        formatted_results = []
        for result in results:
            label = result["label"]
            score = result["score"]

            sentiment = sentiment_mapping.get(label, "neutral")

            result_dict = {
                "sentiment": sentiment,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "score": float(score)
            }

            # تنظیم مقدار احساس تشخیص داده شده
            result_dict[sentiment] = float(score)

            # تنظیم سایر مقادیر به گونه‌ای که مجموع 1 شود
            if sentiment == "positive":
                result_dict["neutral"] = (1.0 - score) * 0.7
                result_dict["negative"] = (1.0 - score) * 0.3
            elif sentiment == "negative":
                result_dict["neutral"] = (1.0 - score) * 0.7
                result_dict["positive"] = (1.0 - score) * 0.3
            else:  # neutral
                result_dict["positive"] = (1.0 - score) * 0.5
                result_dict["negative"] = (1.0 - score) * 0.5

            formatted_results.append(result_dict)

        return formatted_results

    @retry_on_error()
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        استخراج کلمات کلیدی از متن

        Args:
            text: متن ورودی
            top_n: تعداد کلمات کلیدی مورد نیاز

        Returns:
            لیست دیکشنری‌های {"keyword": str, "score": float}
        """
        return self.cached_call(
            f"extract_keywords_{top_n}",
            text,
            lambda x: self._extract_keywords_impl(x, top_n)
        )

    def _extract_keywords_impl(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        استخراج کلمات کلیدی از متن (پیاده‌سازی اصلی)

        Args:
            text: متن ورودی
            top_n: تعداد کلمات کلیدی مورد نیاز

        Returns:
            لیست دیکشنری‌های {"keyword": str, "score": float}
        """
        # پیش‌پردازش متن
        text = self.preprocess_text(text)
        if not text:
            return []

        # بارگیری ابزارهای مورد نیاز
        self._load_hazm()
        self._load_pos_tagger()

        # تقسیم به جملات
        sentences = self.sent_tokenizer.tokenize(text)

        # جمع‌آوری تمام کلمات
        all_words = []
        for sentence in sentences:
            words = self.word_tokenizer.tokenize(sentence)
            all_words.extend(words)

        # برچسب‌گذاری اجزای سخن (POS tagging)
        tagged_words = self.tagger.tag(all_words)

        # فیلتر کردن کلمات کلیدی (اسم‌ها، صفت‌ها و ...)
        potential_keywords = []
        for word, tag in tagged_words:
            # فیلتر کردن stop words و کلمات کوتاه
            if word.lower() in self.stopwords or len(word) < 2:
                continue

            # انتخاب اسم‌ها، صفت‌ها و برخی دیگر از اجزای سخن
            if tag.startswith(('N', 'ADJ')) or tag in ('FW',):  # اسم‌ها، صفت‌ها و کلمات خارجی
                potential_keywords.append(word)

        # شمارش فراوانی کلمات
        from collections import Counter
        keyword_counts = Counter(potential_keywords)

        # تعیین امتیاز بر اساس فراوانی و موقعیت
        keyword_scores = {}
        max_count = max(keyword_counts.values()) if keyword_counts else 1

        for word, count in keyword_counts.items():
            # محاسبه امتیاز نرمال‌شده (0 تا 1)
            score = count / max_count
            keyword_scores[word] = score

        # انتخاب top_n کلمات با بالاترین امتیاز
        top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # ایجاد لیست نتیجه
        result = [{"keyword": word, "score": score} for word, score in top_keywords]

        # یادگیری اصطلاحات جدید
        self._learn_keywords(text, [item["keyword"] for item in result])

        return result

    def _learn_keywords(self, text: str, keywords: List[str]) -> None:
        """
        یادگیری اصطلاحات جدید از کلمات کلیدی

        Args:
            text: متن اصلی
            keywords: لیست کلمات کلیدی
        """
        try:
            # از هر 10 متن، 1 مورد را بررسی کن (برای کاهش بار سیستم)
            import random
            if random.random() > 0.1 or not keywords:
                return

            # ارسال به مخزن داده برای یادگیری
            for keyword in keywords[:3]:  # استفاده از 3 کلمه کلیدی برتر
                self.data_repo.add_learned_term(keyword, 1, None, text[:300])

        except Exception as e:
            self.logger.warning(f"خطا در یادگیری اصطلاحات جدید: {str(e)}")

    @retry_on_error()
    def summarize_text(self, text: str, ratio: float = 0.3) -> str:
        """
        خلاصه‌سازی متن فارسی

        Args:
            text: متن ورودی
            ratio: نسبت خلاصه به متن اصلی (0 تا 1)

        Returns:
            متن خلاصه شده
        """
        return self.cached_call(
            f"summarize_text_{ratio:.1f}",
            text,
            lambda x: self._summarize_text_impl(x, ratio)
        )

    def _summarize_text_impl(self, text: str, ratio: float = 0.3) -> str:
        """
        خلاصه‌سازی متن فارسی (پیاده‌سازی اصلی)

        از روش استخراجی برای خلاصه‌سازی استفاده می‌کند که جملات مهم را
        از متن اصلی انتخاب می‌کند.

        Args:
            text: متن ورودی
            ratio: نسبت خلاصه به متن اصلی (0 تا 1)

        Returns:
            متن خلاصه شده
        """
        # محدود کردن نسبت
        ratio = max(0.1, min(0.9, ratio))

        # پیش‌پردازش متن
        text = self.preprocess_text(text)
        if not text or len(text) < 100:  # متن کوتاه را خلاصه نمی‌کنیم
            return text

        # بارگیری ابزارهای مورد نیاز
        self._load_hazm()

        # تقسیم به جملات
        sentences = self.sent_tokenizer.tokenize(text)

        # اگر تعداد جملات کم است، نیازی به خلاصه‌سازی نیست
        if len(sentences) <= 3:
            return text

        # محاسبه تعداد جملات برای خلاصه
        num_sentences = max(1, int(len(sentences) * ratio))

        # استخراج کلمات کلیدی
        keywords_result = self.extract_keywords(text, top_n=min(20, len(sentences)))
        keywords = [item["keyword"] for item in keywords_result]

        # محاسبه امتیاز جملات
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            # امتیاز بر اساس موقعیت (جملات اول و آخر مهم‌تر هستند)
            position_score = 0
            if i == 0:  # اولین جمله
                position_score = 0.3
            elif i == len(sentences) - 1:  # آخرین جمله
                position_score = 0.2
            elif i <= len(sentences) * 0.2:  # 20٪ اول
                position_score = 0.1

            # امتیاز بر اساس طول (جملات خیلی کوتاه یا خیلی بلند کمتر مهم هستند)
            len_score = 0
            if 10 <= len(sentence) <= 100:
                len_score = 0.1

            # امتیاز بر اساس کلمات کلیدی
            keyword_score = 0
            for keyword in keywords:
                if keyword in sentence:
                    keyword_score += 0.1
            keyword_score = min(0.6, keyword_score)  # حداکثر 0.6

            # امتیاز کل
            sentence_scores[i] = position_score + len_score + keyword_score

        # انتخاب جملات با بالاترین امتیاز
        top_indices = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]

        # مرتب‌سازی بر اساس ترتیب اصلی
        top_indices.sort()

        # ایجاد خلاصه
        summary = " ".join([sentences[i] for i in top_indices])

        return summary

    @retry_on_error()
    def classify_content(self, text: str) -> Dict[str, Any]:
        """
        دسته‌بندی محتوای متن

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"category": str, "confidence": float, "all_categories": []}
        """
        return self.cached_call(
            "classify_content",
            text,
            self._classify_content_impl
        )

    def _classify_content_impl(self, text: str) -> Dict[str, Any]:
        """
        دسته‌بندی محتوای متن (پیاده‌سازی اصلی)

        از امبدینگ و مقایسه با دسته‌بندی‌های پیش‌تعیین شده استفاده می‌کند.

        Args:
            text: متن ورودی

        Returns:
            دیکشنری {"category": str, "confidence": float, "all_categories": []}
        """
        # پیش‌پردازش متن
        text = self.preprocess_text(text)
        if not text:
            return {
                "category": "OTHER",
                "category_name": self.news_categories["OTHER"],
                "confidence": 0.0,
                "all_categories": []
            }

        # استخراج کلمات کلیدی
        keywords_result = self.extract_keywords(text, top_n=15)
        keywords = [item["keyword"] for item in keywords_result]

        # بررسی کلمات کلیدی برای تشخیص دسته‌بندی
        category_scores = {
            "PRICE": 0.0,
            "REGULATION": 0.0,
            "ADOPTION": 0.0,
            "TECHNOLOGY": 0.0,
            "SECURITY": 0.0,
            "EXCHANGE": 0.0,
            "GENERAL": 0.0,
            "OTHER": 0.0
        }

        # کلمات کلیدی مرتبط با هر دسته‌بندی
        category_keywords = {
            "PRICE": ["قیمت", "ارزش", "دلار", "تومان", "صعودی", "نزولی", "افزایش", "کاهش",
                      "رشد", "سقوط", "بازار", "معامله", "خرید", "فروش", "سرمایه‌گذاری",
                      "معاملات", "نمودار", "چارت", "روند", "تحلیل", "پیش‌بینی", "آینده"],
            "REGULATION": ["قانون", "مقررات", "قانونی", "دولت", "مجلس", "مقام", "ناظر",
                           "رگولاتوری", "ممنوع", "مجاز", "جرم", "مالیات", "مصادره", "دارایی",
                           "نهاد", "سازمان", "وزارت", "مرکزی", "بخشنامه", "اعلامیه"],
            "ADOPTION": ["پذیرش", "استفاده", "کاربرد", "پرداخت", "فروشگاه", "خدمات",
                         "به‌کارگیری", "پیاده‌سازی", "راه‌اندازی", "کیف پول", "تراکنش",
                         "کسب‌وکار", "شرکت", "همکاری", "قرارداد", "مشارکت"],
            "TECHNOLOGY": ["فناوری", "تکنولوژی", "توسعه", "بلاکچین", "به‌روزرسانی", "شبکه",
                           "پروتکل", "قرارداد هوشمند", "کد", "توسعه‌دهنده", "لایه", "کانال",
                           "زنجیره", "کیف", "استخراج", "ماینینگ", "سخت‌افزار", "نرم‌افزار"],
            "SECURITY": ["امنیت", "هک", "سرقت", "حمله", "آسیب‌پذیری", "باگ", "خطا", "نفوذ",
                         "فیشینگ", "کلاهبرداری", "اسکم", "رمز", "کلید", "محرمانه", "حفاظت",
                         "دو عاملی", "تأیید", "احراز هویت"],
            "EXCHANGE": ["صرافی", "پلتفرم", "معاملاتی", "کارمزد", "واریز", "برداشت", "اکانت",
                         "حساب", "ثبت‌نام", "احراز هویت", "کاربری", "بررسی", "لیست", "جفت ارز"]
        }

        # بررسی متن برای کلمات کلیدی هر دسته‌بندی
        for category, words in category_keywords.items():
            score = 0.0
            for word in words:
                if word in text.lower():
                    score += 0.1

            for keyword in keywords:
                if keyword in words:
                    score += 0.2

            category_scores[category] = min(1.0, score)

        # حداقل امتیاز برای GENERAL
        category_scores["GENERAL"] = 0.2

        # انتخاب دسته‌بندی با بالاترین امتیاز
        best_category = max(category_scores.items(), key=lambda x: x[1])

        # اگر امتیاز کم باشد، به OTHER تغییر می‌دهیم
        if best_category[1] < 0.3:
            best_category = ("OTHER", 0.5)

        # ایجاد لیست تمام دسته‌بندی‌ها با امتیاز
        all_categories = []
        for category, score in category_scores.items():
            if score > 0:
                all_categories.append({
                    "category": category,
                    "category_name": self.news_categories[category],
                    "confidence": score
                })

        # مرتب‌سازی بر اساس امتیاز
        all_categories.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "category": best_category[0],
            "category_name": self.news_categories[best_category[0]],
            "confidence": best_category[1],
            "all_categories": all_categories
        }

    @retry_on_error()
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه شباهت معنایی بین دو متن

        Args:
            text1: متن اول
            text2: متن دوم

        Returns:
            عدد اعشاری بین 0 تا 1 (شباهت)
        """
        return self.cached_call(
            "calculate_text_similarity",
            f"{text1[:100]}|{text2[:100]}",  # کلید کش با ترکیب دو متن
            lambda x: self._calculate_text_similarity_impl(text1, text2)
        )

    def _calculate_text_similarity_impl(self, text1: str, text2: str) -> float:
        """
        محاسبه شباهت معنایی بین دو متن (پیاده‌سازی اصلی)

        Args:
            text1: متن اول
            text2: متن دوم

        Returns:
            عدد اعشاری بین 0 تا 1 (شباهت)
        """
        # پیش‌پردازش متون
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)

        # بررسی متون خالی
        if not text1 or not text2:
            return 0.0

        # متون یکسان
        if text1 == text2:
            return 1.0

        # دریافت امبدینگ‌ها
        embedding1 = self.get_text_embedding(text1)
        embedding2 = self.get_text_embedding(text2)

        # محاسبه شباهت کسینوسی
        embedding1 = np.array(embedding1).reshape(1, -1)
        embedding2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]

        # محدود کردن به بازه [0, 1]
        similarity = max(0.0, min(1.0, similarity))

        return float(similarity)

    def analyze_news(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        تحلیل جامع خبر

        تحلیل کامل یک خبر شامل احساسات، دسته‌بندی، کلمات کلیدی و خلاصه

        Args:
            text: متن خبر
            title: عنوان خبر (اختیاری)

        Returns:
            دیکشنری نتایج تحلیل
        """
        # ترکیب عنوان و متن
        full_text = text
        if title:
            full_text = f"{title}\n{text}"

        # تحلیل همه جنبه‌ها
        sentiment_result = self.analyze_sentiment(full_text)
        category_result = self.classify_content(full_text)
        keywords = self.extract_keywords(full_text, top_n=10)

        # خلاصه‌سازی اگر متن طولانی باشد
        summary = ""
        if len(text) > 500:
            summary = self.summarize_text(text, ratio=0.3)

        # ایجاد نتیجه نهایی
        result = {
            "sentiment": sentiment_result,
            "category": category_result,
            "keywords": keywords,
            "summary": summary if summary else None
        }

        return result

    def find_related_news(self, target_text: str, all_texts: List[str],
                          min_similarity: float = 0.6, limit: int = 5) -> List[Dict[str, Any]]:
        """
        یافتن اخبار مرتبط با یک خبر

        Args:
            target_text: متن خبر هدف
            all_texts: لیست تمام متون خبری
            min_similarity: حداقل شباهت (0 تا 1)
            limit: حداکثر تعداد نتایج

        Returns:
            لیست دیکشنری‌ها با اطلاعات اخبار مرتبط
        """
        # پیش‌پردازش متن هدف
        target_text = self.preprocess_text(target_text)
        if not target_text or not all_texts:
            return []

        # محاسبه امبدینگ متن هدف
        target_embedding = np.array(self.get_text_embedding(target_text))

        # محاسبه امبدینگ‌های تمام متون
        all_embeddings = self.batch_get_text_embeddings([text for text in all_texts])
        all_embeddings = np.array(all_embeddings)

        # محاسبه شباهت با هر متن
        similarities = []
        for i, embedding in enumerate(all_embeddings):
            # جلوگیری از مقایسه متن با خودش
            if all_texts[i] == target_text:
                continue

            # محاسبه شباهت
            similarity = cosine_similarity(target_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]

            # افزودن به نتایج اگر شباهت کافی باشد
            if similarity >= min_similarity:
                similarities.append({
                    "index": i,
                    "similarity": float(similarity)
                })

        # مرتب‌سازی بر اساس شباهت
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # محدود کردن تعداد نتایج
        similarities = similarities[:limit]

        # ایجاد نتیجه نهایی
        results = []
        for item in similarities:
            index = item["index"]
            results.append({
                "text": all_texts[index],
                "similarity": item["similarity"]
            })

        return results
