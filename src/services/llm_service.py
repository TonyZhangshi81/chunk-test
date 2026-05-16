"""负责生成回答的 LLM 客户端封装。"""

import logging

from openai import OpenAI

from config import Config


logger = logging.getLogger(__name__)


class LLMService:
    """使用配置好的对话模型，根据检索上下文生成答案。"""

    def __init__(self, cfg: Config):
        self.config = cfg
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """延迟创建 OpenAI 兼容客户端。"""
        if self._client is None:
            logger.info("Initializing LLM client model=%s", self.config.LLM_MODEL)
            self._client = OpenAI(api_key=self.config.LLM_API_KEY, base_url=self.config.LLM_API_BASE)
        return self._client

    def answer(self, query: str, contexts: list[str]) -> str:
        """根据给定问题和检索上下文生成单轮回答。"""
        # 当前 CLI 只处理单轮问答，因此把所有上下文拼成一个提示词块更直接。
        context_block = "\n\n".join(f"片段{i + 1}:\n{context}" for i, context in enumerate(contexts))
        prompt = (
            "你是RAG问答助手。请仅基于给定上下文回答问题；"
            "若上下文不足，请明确说明。\n\n"
            f"上下文:\n{context_block}\n\n"
            f"问题: {query}"
        )
        logger.info("Requesting LLM answer with %s contexts", len(contexts))
        response = self.client.chat.completions.create(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
