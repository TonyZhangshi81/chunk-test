from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from openai import OpenAI

from config import Config


logger = logging.getLogger(__name__)


def _extract_response_data(response: Any) -> list[Any]:
    """把不同 SDK 的响应结构统一转换为列表。"""
    if isinstance(response, dict):
        return list(response.get("data", []))
    return list(getattr(response, "data", []))


def _extract_embedding(item: Any) -> list[float]:
    """把单个 embedding 项统一转换成 Python 列表。"""
    if isinstance(item, dict):
        return list(item["embedding"])
    return list(item.embedding)


class BaseEmbeddingProvider(ABC):
    """所有 embedding 提供方都必须实现的公共协议。"""

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
        """为每条输入文本返回一个向量。"""
        raise NotImplementedError


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """面向 OpenAI 兼容接口的 embedding 提供方。"""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            logger.info("Initializing OpenAI-compatible embedding client model=%s", self.model_name)
            self._client = OpenAI(api_key=self.config.EMBEDDING_API_KEY, base_url=self.config.EMBEDDING_API_BASE)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        logger.info("Requesting embeddings from OpenAI-compatible provider text_count=%s", len(texts))
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [_extract_embedding(item) for item in _extract_response_data(response)]


class ZhipuEmbeddingProvider(BaseEmbeddingProvider):
    """面向智谱 embedding 接口的提供方。"""

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
            logger.info("Initializing Zhipu embedding client model=%s", self.model_name)
            self._client = ZhipuAiClient(api_key=self.config.EMBEDDING_API_KEY)
        return self._client

    @property
    def effective_dimension(self) -> int | None:
        # 智谱不同模型家族对维度有限制，因此统一在这里集中校验。
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
        logger.info("Requesting embeddings from Zhipu provider text_count=%s", len(texts))
        request: dict[str, Any] = {"model": self.model_name, "input": texts}
        dimension = self.effective_dimension
        if self.model_name == "embedding-3" and dimension is not None:
            # 当前项目里只有 embedding-3 支持由调用方指定维度。
            request["dimensions"] = dimension
        response = self.client.embeddings.create(**request)
        return [_extract_embedding(item) for item in _extract_response_data(response)]


class EmbeddingService:
    """对外隐藏 provider 选择细节的统一入口。"""

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
            # 延迟到第一次真实请求时再创建 provider，避免 CLI 启动时做无用初始化。
            logger.info("Selected embedding provider provider=%s", provider_type)
            self._provider = provider_class(self.config)
        return self._provider

    def embed(self, texts: list[str]) -> list[list[float]]:
        """对一组文本执行 embedding，空输入时直接短路返回。"""
        if not texts:
            logger.info("Skipping embedding request because the input list is empty")
            return []
        return self.provider.embed(texts)

    def get_vector_dimension(self, embedding: list[float]) -> int:
        """返回单个向量的维度。"""
        return len(embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """供语义切块路径调用的兼容包装方法。"""
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """为单条检索查询生成向量。"""
        return self.embed([text])[0]
