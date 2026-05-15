from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from models.chunk import Chunk, get_embedding_column_name


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def replace_for_document(self, document_id: str, chunk_type: str, chunks: list[Chunk]) -> None:
        self.session.execute(
            delete(Chunk).where(Chunk.document_id == document_id, Chunk.chunk_type == chunk_type)
        )
        self.session.add_all(chunks)
        self.session.commit()

    def search_similar(
        self,
        document_id: str,
        chunk_type: str,
        embedding_model: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[Chunk]:
        vector_column = getattr(Chunk, get_embedding_column_name(len(query_embedding)))
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.chunk_type == chunk_type,
                Chunk.embedding_model == embedding_model,
                vector_column.is_not(None),
            )
            .order_by(vector_column.cosine_distance(query_embedding))
            .limit(top_k)
        )
        return list(self.session.execute(stmt).scalars().all())
