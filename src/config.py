import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "rag_experiment")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")

    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "rag-chunks")
    MINIO_SECURE: bool = _to_bool(os.getenv("MINIO_SECURE"), False)
    MINIO_PATH_PATTERN: str = os.getenv("MINIO_PATH_PATTERN", "test/{uuid}/{filename}")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    EMBEDDING_API_TYPE: str = os.getenv("EMBEDDING_API_TYPE", "openai")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_API_BASE: str = os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    JINA_API_BASE: str = os.getenv("JINA_API_BASE", "https://api.jina.ai/v1")
    JINA_MODEL: str = os.getenv("JINA_MODEL", "jina-embeddings-v2-base")
    JINA_EMBEDDING_DIMENSION: int = int(os.getenv("JINA_EMBEDDING_DIMENSION", "768"))
    JINA_POOLING_STRATEGY: str = os.getenv("JINA_POOLING_STRATEGY", "mean")

    LLM_API_TYPE: str = os.getenv("LLM_API_TYPE", "zhipu")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "glm-4")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNK_SC_MIN_SIZE: int = int(os.getenv("CHUNK_SC_MIN_SIZE", "100"))
    CHUNK_SC_BREAKPOINT_TYPE: str = os.getenv("CHUNK_SC_BREAKPOINT_TYPE", "percentile")
    CHUNK_SC_SPLIT_REGEX: str = os.getenv("CHUNK_SC_SPLIT_REGEX", r"(?<=[.。．?!？！、])|\n")
    CHUNK_JE_MIN_SIZE: int = int(os.getenv("CHUNK_JE_MIN_SIZE", "100"))

    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "4"))

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


config = Config()
