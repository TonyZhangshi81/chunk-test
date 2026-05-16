"""保存检索结果与评分的实验 ORM 模型。"""

from sqlalchemy import DateTime, Float, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.database import Base


class Experiment(Base):
    """持久化一次查询运行结果，包含答案、上下文与相似度得分。"""

    __tablename__ = "t_experiment"

    id: Mapped[str] = mapped_column(primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("t_document.id"), nullable=False)
    chunk_type: Mapped[str] = mapped_column(nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    contexts: Mapped[str | None] = mapped_column(Text)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
