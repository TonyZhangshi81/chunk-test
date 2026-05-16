from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.payloads import build_chunk_payloads


logger = logging.getLogger(__name__)


class SCStrategy(BaseChunkStrategy):
    """语义切块策略，优先使用 LangChain，不可用时退回本地实现。"""

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    @property
    def strategy_type(self) -> str:
        return "SC"

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        """根据 embedding 相似度推断语义边界并切分文本。"""
        min_size = kwargs.get("min_chunk_size", 100)
        breakpoint_type = kwargs.get("breakpoint_type", "percentile")
        split_regex = kwargs.get("split_regex", r"(?<=[.。．?!？！、])|\n")
        if SemanticChunker is not None:
            # 若库实现可用则优先使用，它已经封装了较成熟的语义边界处理逻辑。
            logger.info("Running SemanticChunker min_size=%s breakpoint_type=%s", min_size, breakpoint_type)
            splitter = SemanticChunker(
                embeddings=self.embedding_service,
                breakpoint_threshold_type=breakpoint_type,
                sentence_split_regex=split_regex,
                min_chunk_size=min_size,
            )
            chunks = splitter.split_text(text)
            logger.info("SemanticChunker produced %s chunks", len(chunks))
            return build_chunk_payloads(chunks, text)

        sentences = [part.strip() for part in re.split(split_regex, text) if part.strip()]
        if len(sentences) <= 1:
            logger.info("SC fallback returned original text because sentence_count=%s", len(sentences))
            return build_chunk_payloads([text], text)

        logger.info("Running local SC fallback sentence_count=%s", len(sentences))
        embeddings = self.embedding_service.embed_documents(sentences)
        similarities = [
            # 回退路径里用相邻句子相似度作为低成本的边界信号。
            self._cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        # 相似度越低越可能发生主题切换，因此用分位数阈值做断点判断较稳妥。
        threshold = float(np.percentile(similarities, 30)) if similarities else 0.0

        chunks = []
        current_parts = []
        current_length = 0
        for index, sentence in enumerate(sentences):
            current_parts.append(sentence)
            current_length += len(sentence)
            should_break = index == len(sentences) - 1
            if not should_break and similarities[index] < threshold and current_length >= min_size:
                should_break = True
            if should_break:
                # 当前片段达到最小长度或已经到末尾时，才真正固化为一个 chunk。
                chunks.append("".join(current_parts))
                current_parts = []
                current_length = 0

        logger.info("SC fallback produced %s chunks", len(chunks))
        return build_chunk_payloads(chunks, text)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度，并处理零向量情况。"""
        vector_a = np.array(a)
        vector_b = np.array(b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)
