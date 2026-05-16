"""递归字符切块策略。"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.payloads import build_chunk_payloads


logger = logging.getLogger(__name__)


class RCTSStrategy(BaseChunkStrategy):
    """把文本切成固定大小且带重叠的窗口。"""

    @property
    def strategy_type(self) -> str:
        return "RCTS"

    def split(self, text: str, **kwargs) -> list[dict[str, int | str | None]]:
        """按配置的窗口大小与重叠量执行递归字符切分。"""
        chunk_size = kwargs.get("chunk_size", 500)
        overlap = kwargs.get("overlap", 50)
        logger.info("Running RCTS split chunk_size=%s overlap=%s", chunk_size, overlap)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        logger.info("RCTS produced %s chunks", len(chunks))
        return build_chunk_payloads(chunks, text)
