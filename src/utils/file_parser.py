"""负责解析受支持文档格式的辅助工具。"""

import logging
from pathlib import Path

from docx import Document as DocxDocument


logger = logging.getLogger(__name__)


def parse_file(file_path: str) -> tuple[str, bytes]:
    """解析 txt 或 docx 文件，并返回规范化文本与原始字节流。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    # 同时返回原始字节给对象存储使用，避免为上传再次读取文件。
    raw_bytes = path.read_bytes()
    logger.info("Parsing file path=%s suffix=%s size=%s", file_path, suffix, len(raw_bytes))
    if suffix == ".txt":
        return raw_bytes.decode("utf-8"), raw_bytes
    if suffix == ".docx":
        document = DocxDocument(path)
        # 跳过空段落，减少后续切块阶段的噪声。
        content = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        return content, raw_bytes
    logger.error("Unsupported file type for path=%s suffix=%s", file_path, suffix)
    raise ValueError("Only .txt and .docx files are supported")
