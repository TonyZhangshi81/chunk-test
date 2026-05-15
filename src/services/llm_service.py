from openai import OpenAI

from config import Config


class LLMService:
    def __init__(self, cfg: Config):
        self.config = cfg
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.config.LLM_API_KEY, base_url=self.config.LLM_API_BASE)
        return self._client

    def answer(self, query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"片段{i + 1}:\n{context}" for i, context in enumerate(contexts))
        prompt = (
            "你是RAG问答助手。请仅基于给定上下文回答问题；"
            "若上下文不足，请明确说明。\n\n"
            f"上下文:\n{context_block}\n\n"
            f"问题: {query}"
        )
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
