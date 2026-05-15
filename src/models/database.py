from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from config import config


Base = declarative_base()
engine = create_engine(config.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session():
    return SessionLocal()


def init_db() -> None:
    from models import chunk, document, experiment  # noqa: F401

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)


def rebuild_chunk_table() -> None:
    from models.chunk import Chunk

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Chunk.__table__.drop(bind=engine, checkfirst=True)
    init_db()
