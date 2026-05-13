from pathlib import Path

from docx import Document as DocxDocument


def parse_file(file_path: str) -> tuple[str, bytes]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    raw_bytes = path.read_bytes()
    if suffix == ".txt":
        return raw_bytes.decode("utf-8"), raw_bytes
    if suffix == ".docx":
        document = DocxDocument(path)
        content = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        return content, raw_bytes
    raise ValueError("Only .txt and .docx files are supported")
