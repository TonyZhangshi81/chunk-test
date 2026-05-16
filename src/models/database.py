"""数据库引擎与会话管理辅助工具。"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import config


logger = logging.getLogger(__name__)


Base = declarative_base()
# 该项目以短生命周期 CLI 命令运行，因此引擎对象按进程复用即可。
engine = create_engine(config.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session():
    """为当前 CLI 调用创建一个新的 SQLAlchemy 会话。"""
    logger.debug("Creating database session")
    return SessionLocal()


def init_db() -> None:
    """初始化数据库扩展，并创建 ORM 管理的全部表。"""
    from models import chunk, document, experiment  # noqa: F401

    logger.info("Initializing database schema")
    with engine.begin() as connection:
        # 必须先确保 pgvector 扩展存在，t_chunk 中的向量列才能成功创建。
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)


def rebuild_chunk_table() -> None:
    """只重建 chunk 表，同时保持其余表结构不变。"""
    from models.chunk import Chunk

    logger.warning("Rebuilding chunk table")
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 向量列结构变化都集中在 chunk 表，因此只需重建这一张表。
    Chunk.__table__.drop(bind=engine, checkfirst=True)
    init_db()
