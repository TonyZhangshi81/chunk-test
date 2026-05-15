from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from config import Config


def _extract_response_data(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("data", []))
    return list(getattr(response, "data", []))


def _extract_embedding(item: Any) -> list[float]:
    if isinstance(item, dict):
        return list(item["embedding"])
    return list(item.embedding)


class BaseEmbeddingProvider(ABC):
    def __init__(self, cfg: Config):
        self.config = cfg

    @property
    def model_name(self) -> str:
        return self.config.EMBEDDING_MODEL

    @property
    def requested_dimension(self) -> int | None:
        return self.config.EMBEDDING_DIMENSION

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.config.EMBEDDING_API_KEY, base_url=self.config.EMBEDDING_API_BASE)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [_extract_embedding(item) for item in _extract_response_data(response)]


class ZhipuEmbeddingProvider(BaseEmbeddingProvider):
    _EMBEDDING_3_DIMENSIONS = {256, 512, 1024, 2048}

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from zai import ZhipuAiClient
            except ImportError as exc:
                raise ImportError("zai-sdk is required for the zhipu embedding provider") from exc
            self._client = ZhipuAiClient(api_key=self.config.EMBEDDING_API_KEY)
        return self._client

    @property
    def effective_dimension(self) -> int | None:
        if self.model_name == "embedding-3":
            dimension = self.requested_dimension or 1024
            if dimension not in self._EMBEDDING_3_DIMENSIONS:
                supported = ", ".join(str(item) for item in sorted(self._EMBEDDING_3_DIMENSIONS))
                raise ValueError(f"embedding-3 only supports dimensions: {supported}")
            return dimension
        if self.model_name == "embedding-2":
            if self.requested_dimension not in {None, 1024}:
                raise ValueError("embedding-2 only supports dimension 1024")
            return 1024
        return self.requested_dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        request: dict[str, Any] = {"model": self.model_name, "input": texts}
        dimension = self.effective_dimension
        if self.model_name == "embedding-3" and dimension is not None:
            request["dimensions"] = dimension
        response = self.client.embeddings.create(**request)
        return [_extract_embedding(item) for item in _extract_response_data(response)]


class EmbeddingService:
    def __init__(self, cfg: Config):
        self.config = cfg
        self._provider: BaseEmbeddingProvider | None = None

    @property
    def provider(self) -> BaseEmbeddingProvider:
        if self._provider is None:
            provider_type = self.config.EMBEDDING_PROVIDER.lower()
            provider_map: dict[str, type[BaseEmbeddingProvider]] = {
                "openai": OpenAIEmbeddingProvider,
                "openai-compatible": OpenAIEmbeddingProvider,
                "zhipu": ZhipuEmbeddingProvider,
            }
            try:
                provider_class = provider_map[provider_type]
            except KeyError as exc:
                supported = ", ".join(sorted(provider_map))
                raise ValueError(f"Unsupported embedding provider: {provider_type}. Supported: {supported}") from exc
            self._provider = provider_class(self.config)
        return self._provider

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.provider.embed(texts)

    def get_vector_dimension(self, embedding: list[float]) -> int:
        return len(embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
