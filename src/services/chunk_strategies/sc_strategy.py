from __future__ import annotations

import re
from typing import Any

import numpy as np

try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    SemanticChunker = None

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.rcts_strategy import _build_chunk_payloads


class SCStrategy(BaseChunkStrategy):
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service

    @property
    def strategy_type(self) -> str:
        return "SC"

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        min_size = kwargs.get("min_chunk_size", 100)
        breakpoint_type = kwargs.get("breakpoint_type", "percentile")
        split_regex = kwargs.get("split_regex", r"(?<=[.。．?!？！、])|\n")
        if SemanticChunker is not None:
            splitter = SemanticChunker(
                embeddings=self.embedding_service,
                breakpoint_threshold_type=breakpoint_type,
                sentence_split_regex=split_regex,
                min_chunk_size=min_size,
            )
            chunks = splitter.split_text(text)
            return _build_chunk_payloads(chunks, text)

        sentences = [part.strip() for part in re.split(split_regex, text) if part.strip()]
        if len(sentences) <= 1:
            return _build_chunk_payloads([text], text)

        embeddings = self.embedding_service.embed_documents(sentences)
        similarities = [
            self._cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
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
                chunks.append("".join(current_parts))
                current_parts = []
                current_length = 0

        return _build_chunk_payloads(chunks, text)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vector_a = np.array(a)
        vector_b = np.array(b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)
