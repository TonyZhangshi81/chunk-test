from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from config import config
from models.database import Base


class Chunk(Base):
    __tablename__ = "t_chunk"

    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("t_document.id"), nullable=False)
    chunk_type: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_position: Mapped[int | None] = mapped_column(Integer)
    end_position: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None]
    embedding_vector: Mapped[list[float] | None] = mapped_column(Vector(config.EMBEDDING_DIMENSION))
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
