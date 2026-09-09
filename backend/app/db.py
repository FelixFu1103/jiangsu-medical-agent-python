from functools import lru_cache
from pathlib import Path
import chromadb
from .config import get_settings


@lru_cache
def client():
    return chromadb.PersistentClient(path=str(Path(get_settings().chroma_path).resolve()))


def documents():
    return client().get_or_create_collection("knowledge_documents")


def chunks():
    return client().get_or_create_collection("knowledge_chunks", metadata={"hnsw:space": "cosine"})


def open_db() -> None:
    documents()
    chunks()


def close_db() -> None:
    return None
