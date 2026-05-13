from __future__ import annotations

import re
from typing import Any

import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.rcts_strategy import _build_chunk_payloads


class JEStrategy(BaseChunkStrategy):
    def __init__(self, cfg):
        self.api_key = cfg.JINA_API_KEY
        self.api_base = cfg.JINA_API_BASE.rstrip("/")
        self.model = cfg.JINA_MODEL
        self.pooling = cfg.JINA_POOLING_STRATEGY

    @property
    def strategy_type(self) -> str:
        return "JE"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "pooling_strategy": self.pooling,
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vector_a = np.array(a)
        vector_b = np.array(b)
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)

    def _split_sentences(self, text: str, regex: str) -> list[str]:
        return [segment.strip() for segment in re.split(regex, text) if segment.strip()]

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        chunk_size = kwargs.get("chunk_size", 500)
        overlap = kwargs.get("overlap", 50)
        min_size = kwargs.get("min_chunk_size", 100)
        split_regex = kwargs.get("split_regex", r"(?<=[.。．?!？！、])|\n")

        sentences = self._split_sentences(text, split_regex)
        if len(sentences) <= 1:
            return _build_chunk_payloads([text], text)

        embeddings = self._get_embeddings(sentences)
        similarities = [
            self._cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(embeddings) - 1)
        ]
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        std_similarity = float(np.std(similarities)) if similarities else 0.0
        threshold = mean_similarity - 0.5 * std_similarity

        breakpoints = []
        current_size = 0
        for index, similarity in enumerate(similarities):
            current_size += len(sentences[index])
            if similarity < threshold and current_size >= min_size:
                breakpoints.append(index + 1)
                current_size = 0

        chunks = []
        start = 0
        for end in sorted(set(breakpoints + [len(sentences)])):
            if end <= start:
                continue
            chunks.append("。".join(sentences[start:end]))
            start = end

        final_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        for chunk in chunks:
            if len(chunk) > chunk_size:
                final_chunks.extend(splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)

        return _build_chunk_payloads(final_chunks, text)
