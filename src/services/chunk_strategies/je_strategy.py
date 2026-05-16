from __future__ import annotations

import re
from typing import Any

import requests

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.payloads import build_chunk_payloads


class JEStrategy(BaseChunkStrategy):
    def __init__(self, cfg):
        self.api_key = cfg.JINA_API_KEY
        self.api_base = cfg.JINA_API_BASE.rstrip("/")
        self.model = cfg.JINA_MODEL
        self.pooling = cfg.JINA_POOLING_STRATEGY
        self.chunk_type = cfg.JINA_CHUNK_TYPE
        self.max_chunk_length = cfg.JINA_MAX_CHUNK_LENGTH
        self.task = cfg.JINA_TASK
        self.dimensions = cfg.JINA_EMBEDDING_DIMENSION
        

    @property
    def strategy_type(self) -> str:
        return "JE"

    @property
    def embedding_model(self) -> str:
        return self.model

    def _get_embeddings(self, texts: list[str]) -> list[dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
            "late_chunking": True,
            "task": self.task,
            "return_chunks": True,
            "chunk_type": self.chunk_type,
            "pooling_strategy": self.pooling,
            "max_chunk_length": self.max_chunk_length,
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        response_items = result.get("data", [])
        print(f"Jina API returned {len(response_items)} chunks for the document.")
        chunks = []
        for index, item in enumerate(response_items):
            if index >= len(texts):
                continue
            chunks.append(
                {
                    "text": texts[index],
                    "embedding": item.get("embedding", []),
                    "index": item.get("index", 0),
                }
            )
        return chunks

    def embed_query(self, query: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": query,
            "pooling_strategy": self.pooling,
        }
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        data = result.get("data", [])
        if not data:
            raise ValueError("Jina API returned empty embedding data for query")
        return data[0].get("embedding", [])
    
    def _split_sentences(self, text: str, regex: str) -> list[str]:
        return [segment.strip() for segment in re.split(regex, text) if segment.strip()]

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        split_regex = r"\n"
        sentences = self._split_sentences(text, split_regex)

        chunks = sorted(self._get_embeddings(sentences), key=lambda item: item["index"])
        chunk_texts = [item["text"] for item in chunks]
        print(f"JE Strategy split text into {len(chunk_texts)} chunks.")
        payloads = build_chunk_payloads(chunk_texts or [text], text)
        if not chunks:
            return payloads

        for payload, chunk in zip(payloads, chunks):
            payload["embedding"] = chunk["embedding"]
        return payloads
