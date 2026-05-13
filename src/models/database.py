from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import config


Base = declarative_base()
engine = create_engine(config.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session():
    return SessionLocal()


def init_db() -> None:
    from models import chunk, document, experiment  # noqa: F401

    Base.metadata.create_all(bind=engine)
