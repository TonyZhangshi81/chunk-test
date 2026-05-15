from __future__ import annotations

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

    @property
    def strategy_type(self) -> str:
        return "JE"

    @property
    def embedding_model(self) -> str:
        return self.model

    def _get_embeddings(self, document: str) -> list[dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": document,
            "return_chunks": True,
            "chunk_type": self.chunk_type,
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
        response_items = result.get("data", [])
        chunks = []
        for item in response_items:
            chunk_text = (item.get("text") or item.get("chunk") or "").strip()
            if not chunk_text and len(response_items) == 1:
                chunk_text = document.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "text": chunk_text,
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

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        chunks = sorted(self._get_embeddings(text), key=lambda item: item["index"])
        chunk_texts = [item["text"] for item in chunks]
        payloads = build_chunk_payloads(chunk_texts or [text], text)
        if not chunks:
            return payloads

        for payload, chunk in zip(payloads, chunks):
            payload["embedding"] = chunk["embedding"]
        return payloads
