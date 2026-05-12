import uuid
from pathlib import Path

import pandas as pd

from src.knowledge.vector_store import VectorStore


class DocumentIngester:
    def __init__(self, vector_store: VectorStore, chunk_size: int = 500):
        self._store = vector_store
        self._chunk_size = chunk_size

    def ingest_file(self, client_id: str, file_path: str) -> int:
        path = Path(file_path)
        suffix = path.suffix.lower()

        parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".xlsx": self._parse_xlsx,
            ".xls": self._parse_xlsx,
            ".csv": self._parse_csv,
            ".txt": lambda p: Path(p).read_text(encoding="utf-8", errors="ignore"),
        }

        if suffix not in parsers:
            raise ValueError(f"Unsupported file type: {suffix}")

        text = parsers[suffix](file_path)
        chunks = self._chunk(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": path.name, "client_id": client_id} for _ in chunks]
        self._store.add(client_id, chunks, ids, metadatas)
        return len(chunks)

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self._chunk_size):
            chunk = " ".join(words[i : i + self._chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks or [text[:200]]

    def _parse_pdf(self, path: str) -> str:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def _parse_docx(self, path: str) -> str:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    def _parse_xlsx(self, path: str) -> str:
        sheets = pd.read_excel(path, sheet_name=None)
        parts = [f"Sheet: {name}\n{df.to_string()}" for name, df in sheets.items()]
        return "\n\n".join(parts)

    def _parse_csv(self, path: str) -> str:
        return pd.read_csv(path).to_string()
