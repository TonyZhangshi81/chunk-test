from typing import List

import numpy as np

from services.embedding_service import EmbeddingService


class QualityEvaluator:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    def evaluate(self, query: str, answer: str, contexts: List[str]) -> float:
        if not answer or not contexts:
            return 0.0

        answer_emb = self.embedding_service.embed([answer])[0]
        query_emb = self.embedding_service.embed([query])[0]
        context_embs = self.embedding_service.embed(contexts)
        context_sims = [self._cosine_similarity(answer_emb, ctx_emb) for ctx_emb in context_embs]
        ctx_score = max(context_sims) if context_sims else 0.5
        query_score = self._cosine_similarity(answer_emb, query_emb)
        return 0.7 * ctx_score + 0.3 * query_score

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vector_a = np.array(a)
        vector_b = np.array(b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)
