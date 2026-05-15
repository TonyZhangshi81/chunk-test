def build_chunk_payloads(chunks: list[str], source_text: str) -> list[dict[str, int | str | None]]:
    payloads = []
    search_from = 0
    for index, chunk in enumerate(chunks):
        start_pos = source_text.find(chunk, search_from)
        if start_pos < 0:
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