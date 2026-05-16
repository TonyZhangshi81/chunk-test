"""Chunk 记录的持久化与检索辅助封装。"""

import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from models.chunk import Chunk, get_embedding_column_name


logger = logging.getLogger(__name__)


class ChunkRepository:
    """封装 chunk 的写入与向量检索操作。"""

    def __init__(self, session: Session):
        self.session = session

    def replace_for_document(self, document_id: str, chunk_type: str, chunks: list[Chunk]) -> None:
        """在单次事务中替换某文档某策略下的全部 chunk。"""
        logger.info(
            "Replacing chunks for document_id=%s chunk_type=%s chunk_count=%s",
            document_id,
            chunk_type,
            len(chunks),
        )
        # 采用整批替换而不是局部修补，能保证每次策略运行只有一套权威结果。
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
        """根据查询向量返回最相似的已保存 chunk。"""
        # 不同模型会写入不同维度列，因此这里需要动态选择向量列。
        vector_column = getattr(Chunk, get_embedding_column_name(len(query_embedding)))
        logger.info(
            "Searching similar chunks for document_id=%s chunk_type=%s model=%s top_k=%s",
            document_id,
            chunk_type,
            embedding_model,
            top_k,
        )
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == document_id,
                Chunk.chunk_type == chunk_type,
                Chunk.embedding_model == embedding_model,
                # 只查询当前维度列非空的数据，避免扫到其它模型维度产生的记录。
                vector_column.is_not(None),
            )
            .order_by(vector_column.cosine_distance(query_embedding))
            .limit(top_k)
        )
        results = list(self.session.execute(stmt).scalars().all())
        logger.info("Retrieved %s similar chunks", len(results))
        return results
