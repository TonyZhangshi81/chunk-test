from openai import OpenAI

from config import Config


class EmbeddingService:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.client = OpenAI(api_key=cfg.EMBEDDING_API_KEY, base_url=cfg.EMBEDDING_API_BASE)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.config.EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    def get_vector_dimension(self, embedding: list[float]) -> int:
        return len(embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
