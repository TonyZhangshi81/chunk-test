"""基于相似度的回答质量评估器。"""

import logging
from typing import List

import numpy as np

from services.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class QualityEvaluator:
    """根据答案与问题、上下文的相似度估算回答质量。"""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def evaluate(self, query: str, answer: str, contexts: List[str]) -> float:
        """为一次问答结果返回加权后的相似度评分。"""
        if not answer or not contexts:
            logger.warning("Skipping quality evaluation because answer or contexts are empty")
            return 0.0

        logger.info("Evaluating answer quality with %s contexts", len(contexts))
        # 评分同时考虑答案对问题的贴合度，以及答案和检索证据的一致性。
        answer_emb = self.embedding_service.embed([answer])[0]
        query_emb = self.embedding_service.embed([query])[0]
        context_embs = self.embedding_service.embed(contexts)
        context_sims = [self._cosine_similarity(answer_emb, ctx_emb) for ctx_emb in context_embs]
        ctx_score = max(context_sims) if context_sims else 0.5
        query_score = self._cosine_similarity(answer_emb, query_emb)
        return 0.7 * ctx_score + 0.3 * query_score

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个稠密向量之间的余弦相似度。"""
        vector_a = np.array(a)
        vector_b = np.array(b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)
