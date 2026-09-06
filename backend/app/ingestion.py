import hashlib
import io
import re
from pathlib import Path
from bs4 import BeautifulSoup
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from .config import get_settings


ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("仅支持 PDF、DOCX、TXT、Markdown 和 HTML")
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages)
    elif suffix == ".docx":
        text = "\n".join(p.text for p in Document(io.BytesIO(content)).paragraphs)
    else:
        text = content.decode("utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            text = BeautifulSoup(text, "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str) -> list[str]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n", "。", "；", "，", ""],
    )
    return [part.strip() for part in splitter.split_text(text) if part.strip()]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

