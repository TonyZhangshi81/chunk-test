"""把 chunk 文本转换为可持久化 payload 的辅助函数。"""

def build_chunk_payloads(chunks: list[str], source_text: str) -> list[dict[str, int | str | None]]:
    """为每个 chunk 补充顺序索引和尽力匹配得到的原文偏移量。"""
    payloads = []
    search_from = 0
    for index, chunk in enumerate(chunks):
        # 优先从上一次命中的位置继续向前找，减少重复片段错配到更早位置的概率。
        start_pos = source_text.find(chunk, search_from)
        if start_pos < 0:
            # 若前向查找失败，再做全局回退搜索，尽量保留这个 chunk 的位置信息。
            start_pos = source_text.find(chunk)
        end_pos = start_pos + len(chunk) if start_pos >= 0 else None
        if start_pos >= 0:
            search_from = start_pos + len(chunk)
        payloads.append(
            {
                "content": chunk,
                "chunk_index": index,
                "start_pos": start_pos if start_pos >= 0 else None,
                "end_pos": end_pos,
            }
        )
    return payloads