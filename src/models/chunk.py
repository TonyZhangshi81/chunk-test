"""Chunk ORM 模型及多维向量列辅助函数。"""

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from models.database import Base


EMBEDDING_COLUMN_BY_DIMENSION = {
    # 每种支持的向量维度都对应独立列，这样不同模型的结果可以并存。
    2048: "embedding_2048",
    1536: "embedding_1536",
    1024: "embedding_1024",
    768: "embedding_768",
    512: "embedding_512",
    256: "embedding_256",
    3072: "embedding_3072",
}


def get_embedding_column_name(dimension: int) -> str:
    """根据向量维度返回对应的数据库列名。"""
    try:
        return EMBEDDING_COLUMN_BY_DIMENSION[dimension]
    except KeyError as exc:
        supported = ", ".join(str(item) for item in sorted(EMBEDDING_COLUMN_BY_DIMENSION))
        raise ValueError(f"Unsupported embedding dimension: {dimension}. Supported: {supported}") from exc


def build_embedding_column_values(embedding: list[float]) -> dict[str, list[float] | None]:
    """只填充匹配维度的向量列，其余列保持为空。"""
    # 每条记录只写一个向量列，能让后续相似度检索条件保持明确。
    values = {column_name: None for column_name in EMBEDDING_COLUMN_BY_DIMENSION.values()}
    values[get_embedding_column_name(len(embedding))] = embedding
    return values


class Chunk(Base):
    """持久化后的文本块，包含元数据以及唯一生效的向量列。"""

    __tablename__ = "t_chunk"

    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("t_document.id"), nullable=False)
    chunk_type: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_position: Mapped[int | None] = mapped_column(Integer)
    end_position: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    embedding_2048: Mapped[list[float] | None] = mapped_column(Vector(2048))
    embedding_1536: Mapped[list[float] | None] = mapped_column(Vector(1536))
    embedding_1024: Mapped[list[float] | None] = mapped_column(Vector(1024))
    embedding_768: Mapped[list[float] | None] = mapped_column(Vector(768))
    embedding_512: Mapped[list[float] | None] = mapped_column(Vector(512))
    embedding_256: Mapped[list[float] | None] = mapped_column(Vector(256))
    embedding_3072: Mapped[list[float] | None] = mapped_column(Vector(3072))
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
