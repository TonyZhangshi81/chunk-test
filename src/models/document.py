from sqlalchemy import BigInteger, DateTime, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.database import Base


class Document(Base):
    __tablename__ = "t_document"

    id: Mapped[str] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(nullable=False)
    file_size: Mapped[int | None] = mapped_column(BigInteger)
    file_etag: Mapped[str | None]
    file_type: Mapped[str | None]
    mime_type: Mapped[str | None]
    content: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
