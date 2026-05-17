from __future__ import annotations

import logging
import re
from typing import Any

import requests

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.payloads import build_chunk_payloads


logger = logging.getLogger(__name__)


class JEStrategy(BaseChunkStrategy):
    """基于 Jina 的切块策略，切块时同步返回每个块的向量。"""

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
        """为输入文本片段请求 Jina 返回切块感知的向量结果。"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            # Jina 在远端完成带切块感知的向量生成，本地只负责归一化返回结构。
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
        logger.info("Jina embedding API returned %s items", len(response_items))
        chunks = []
        for index, item in enumerate(response_items):
            if index >= len(texts):
                logger.warning("Skipping Jina response index=%s because input count=%s", index, len(texts))
                continue
            chunks.append(
                {
                    # 文本内容取自本地预切分输入，这样后续仍能映射回原文位置。
                    "text": texts[index],
                    "embedding": item.get("embedding", []),
                    "index": item.get("index", 0),
                }
            )
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """向 Jina 请求检索问题的向量表示。"""
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
        """在发送给 Jina 前，先把原文切成较粗粒度片段。"""
        return [segment.strip() for segment in re.split(regex, text) if segment.strip()]

    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        """构造 chunk payload，并挂载 Jina 返回的向量。"""
        # split_regex = r"\n"
        split_regex = kwargs.get("split_regex", r"(?<=[.。．?!？！、])|\n")
        # 先按换行切分，能让送入 Jina 的本地片段边界保持稳定。
        sentences = self._split_sentences(text, split_regex)

        chunks = sorted(self._get_embeddings(sentences), key=lambda item: item["index"])
        chunk_texts = [item["text"] for item in chunks]
        logger.info("JE strategy produced %s chunks", len(chunk_texts))
        payloads = build_chunk_payloads(chunk_texts or [text], text)
        if not chunks:
            return payloads

        for payload, chunk in zip(payloads, chunks):
            # 把向量直接挂到 payload 上，主流程后续就无需再次调用 embedding 服务。
            payload["embedding"] = chunk["embedding"]
        return payloads
