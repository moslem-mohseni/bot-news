"""
ماژول مدیریت دیتابیس برای CryptoNewsBot

این ماژول مسئول مدیریت ارتباط با دیتابیس و انجام عملیات پایه است.
"""

import contextlib
from typing import Any, Dict, List, Optional, TypeVar, Type, Generic

from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.sql import text

from ..utils.config import Config
from ..utils.logger import Logger
from .models import Base

# تعریف نوع برای استفاده در Generic
T = TypeVar('T', bound=Base)


class DatabaseManager:
    """
    کلاس مدیریت دیتابیس

    این کلاس مسئول ارتباط با دیتابیس، ایجاد جداول و مدیریت جلسه‌هاست.
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
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        راه‌اندازی اولیه کلاس و اتصال به دیتابیس
        """
        self.config = Config()
        self.logger = Logger(__name__).get_logger()

        # ساخت URL اتصال
        self.db_url = self.config.get_db_url()

        # راه‌اندازی موتور دیتابیس
        self.engine = create_engine(
            self.db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,
            echo=self.config.debug
        )

        # ساخت کارخانه جلسه
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # ایجاد جلسه با محدوده نخ
        self.Session = scoped_session(self.session_factory)

        self.logger.info(f"اتصال به دیتابیس برقرار شد - {self.db_url}")

    def create_tables(self) -> None:
        """
        ایجاد جداول دیتابیس (اگر وجود نداشته باشند)
        """
        try:
            self.logger.info("در حال ایجاد جداول دیتابیس...")
            Base.metadata.create_all(self.engine)
            self.logger.info("جداول دیتابیس با موفقیت ایجاد شدند")
        except Exception as e:
            self.logger.error(f"خطا در ایجاد جداول دیتابیس: {str(e)}")
            raise

    def drop_tables(self) -> None:
        """
        حذف تمام جداول دیتابیس (خطرناک!)
        """
        try:
            self.logger.warning("در حال حذف تمام جداول دیتابیس...")
            Base.metadata.drop_all(self.engine)
            self.logger.warning("تمام جداول دیتابیس با موفقیت حذف شدند")
        except Exception as e:
            self.logger.error(f"خطا در حذف جداول دیتابیس: {str(e)}")
            raise

    def get_session(self) -> Session:
        """
        دریافت یک جلسه دیتابیس

        Returns:
            جلسه دیتابیس
        """
        return self.Session()

    @contextlib.contextmanager
    def session_scope(self):
        """
        مدیریت جلسه دیتابیس با استفاده از context manager

        این متد یک جلسه را ایجاد می‌کند و پس از اتمام کار، تغییرات را ثبت
        یا در صورت خطا، rollback می‌کند.

        Yields:
            جلسه دیتابیس
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"خطا در تراکنش دیتابیس: {str(e)}")
            raise
        finally:
            session.close()

    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        اجرای یک دستور SQL خام

        Args:
            sql: دستور SQL
            params: پارامترهای دستور SQL

        Returns:
            نتایج دستور SQL به صورت لیستی از دیکشنری‌ها
        """
        params = params or {}
        with self.engine.connect() as connection:
            result = connection.execute(text(sql), params)
            return [dict(row) for row in result]

    # === روش‌های عمومی دسترسی به داده ===

    def add(self, obj: T) -> T:
        """
        افزودن یک شیء به دیتابیس

        Args:
            obj: شیء مورد نظر

        Returns:
            شیء افزوده شده
        """
        with self.session_scope() as session:
            session.add(obj)
            session.flush()
            session.refresh(obj)
            return obj

    def add_all(self, objects: List[T]) -> List[T]:
        """
        افزودن چندین شیء به دیتابیس

        Args:
            objects: لیست اشیاء

        Returns:
            لیست اشیاء افزوده شده
        """
        with self.session_scope() as session:
            session.add_all(objects)
            session.flush()
            for obj in objects:
                session.refresh(obj)
            return objects

    def get_by_id(self, model_class: Type[T], id: int) -> Optional[T]:
        """
        دریافت یک شیء با شناسه آن

        Args:
            model_class: کلاس مدل
            id: شناسه

        Returns:
            شیء مورد نظر یا None
        """
        with self.session_scope() as session:
            return session.query(model_class).filter(model_class.id == id).first()

    def get_all(self, model_class: Type[T], limit: Optional[int] = None, offset: Optional[int] = 0) -> List[T]:
        """
        دریافت تمام اشیاء یک مدل

        Args:
            model_class: کلاس مدل
            limit: محدودیت تعداد نتایج
            offset: تعداد نتایج برای پرش از ابتدا

        Returns:
            لیست اشیاء
        """
        with self.session_scope() as session:
            query = session.query(model_class)
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            return query.all()

    def update(self, obj: T) -> T:
        """
        به‌روزرسانی یک شیء در دیتابیس

        Args:
            obj: شیء مورد نظر

        Returns:
            شیء به‌روزرسانی شده
        """
        with self.session_scope() as session:
            session.merge(obj)
            session.flush()
            session.refresh(obj)
            return obj

    def delete(self, obj: T) -> bool:
        """
        حذف یک شیء از دیتابیس

        Args:
            obj: شیء مورد نظر

        Returns:
            وضعیت موفقیت عملیات
        """
        with self.session_scope() as session:
            session.delete(obj)
            return True

    def delete_by_id(self, model_class: Type[T], id: int) -> bool:
        """
        حذف یک شیء با استفاده از شناسه آن

        Args:
            model_class: کلاس مدل
            id: شناسه

        Returns:
            وضعیت موفقیت عملیات
        """
        with self.session_scope() as session:
            obj = session.query(model_class).filter(model_class.id == id).first()
            if obj:
                session.delete(obj)
                return True
            return False

    def count(self, model_class: Type[T]) -> int:
        """
        شمارش تعداد اشیاء یک مدل

        Args:
            model_class: کلاس مدل

        Returns:
            تعداد اشیاء
        """
        with self.session_scope() as session:
            return session.query(model_class).count()

    def truncate(self, model_class: Type[T]) -> None:
        """
        حذف تمام اشیاء یک مدل

        Args:
            model_class: کلاس مدل
        """
        table_name = model_class.__tablename__
        with self.session_scope() as session:
            session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))

    def get_table_names(self) -> List[str]:
        """
        دریافت نام تمام جداول دیتابیس

        Returns:
            لیست نام جداول
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_db_size(self) -> Dict[str, Any]:
        """
        دریافت اندازه دیتابیس و جداول

        Returns:
            دیکشنری اطلاعات اندازه
        """
        sql = """
        SELECT
            pg_size_pretty(pg_database_size(current_database())) as db_size,
            pg_database_size(current_database()) as db_size_bytes
        """
        db_size = self.execute_raw_sql(sql)[0]

        sql = """
        SELECT
            table_name,
            pg_size_pretty(pg_total_relation_size('"' || table_name || '"')) as table_size,
            pg_total_relation_size('"' || table_name || '"') as table_size_bytes
        FROM
            information_schema.tables
        WHERE
            table_schema = 'public'
        ORDER BY
            table_size_bytes DESC
        """
        tables_size = self.execute_raw_sql(sql)

        return {
            'database': db_size,
            'tables': tables_size
        }
