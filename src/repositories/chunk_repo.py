from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from models.chunk import Chunk


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
        query_embedding: list[float],
        top_k: int,
    ) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.chunk_type == chunk_type)
            .order_by(Chunk.embedding_vector.cosine_distance(query_embedding))
            .limit(top_k)
        )
        return list(self.session.execute(stmt).scalars().all())
