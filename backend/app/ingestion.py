"""Loads source documents (resume PDF, pasted JD text) into plain text."""
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: str | Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")
