from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.chunk_strategies.base import BaseChunkStrategy
from services.chunk_strategies.payloads import build_chunk_payloads


class RCTSStrategy(BaseChunkStrategy):
    @property
    def strategy_type(self) -> str:
        return "RCTS"

    def split(self, text: str, **kwargs) -> list[dict[str, int | str | None]]:
        chunk_size = kwargs.get("chunk_size", 500)
        overlap = kwargs.get("overlap", 50)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        return build_chunk_payloads(chunks, text)
