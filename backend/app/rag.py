from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
from .config import get_settings
from .db import connection


@lru_cache
def embeddings():
    settings = get_settings()
    cache = str(Path(settings.hf_cache_dir).resolve())
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_XET_CACHE", f"{cache}/xet")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.hf_cache_dir,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def reranker():
    settings = get_settings()
    cache = str(Path(settings.hf_cache_dir).resolve())
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_XET_CACHE", f"{cache}/xet")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.rerank_model, device=settings.embedding_device, cache_folder=settings.hf_cache_dir) if settings.rerank_enabled else None


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embeddings().embed_documents(texts)


def as_pgvector(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def rrf(keyword_rows: list[dict], vector_rows: list[dict], k: int = 60) -> list[dict]:
    merged: dict[int, dict] = {}
    for channel, rows in (("keyword", keyword_rows), ("vector", vector_rows)):
        for rank, row in enumerate(rows, 1):
            item = merged.setdefault(row["chunk_id"], {**row, "score": 0.0, "channels": [], "keyword_score": 0.0, "vector_score": 0.0})
            item["score"] += 1 / (k + rank)
            item["channels"].append(channel)
            item[f"{channel}_score"] = float(row.get("raw_score", 0))
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def retrieve(query: str, limit: int = 3) -> list[dict]:
    vector = as_pgvector(embeddings().embed_query(query))
    sql_fields = """SELECT c.id AS chunk_id, d.id AS document_id, d.title, d.department,
      d.source_url, c.content, c.chunk_index"""
    with connection() as conn:
        keyword = conn.execute(f"""{sql_fields}, similarity(c.content, %s) AS raw_score
          FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id
          WHERE d.status='published' AND c.content %% %s
          ORDER BY raw_score DESC LIMIT 12""", (query, query)).fetchall()
        vector_rows = conn.execute(f"""{sql_fields}, 1-(c.embedding <=> %s) AS raw_score
          FROM knowledge_chunks c JOIN knowledge_documents d ON d.id=c.document_id
          WHERE d.status='published' ORDER BY c.embedding <=> %s LIMIT 12""", (vector, vector)).fetchall()
        columns = ["chunk_id", "document_id", "title", "department", "source_url", "content", "chunk_index", "raw_score"]
        candidates = rrf([dict(zip(columns, row)) for row in keyword], [dict(zip(columns, row)) for row in vector_rows])
    model = reranker()
    if model and len(candidates) > 1:
        scores = model.predict([(query, item["content"]) for item in candidates])
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates[:limit]
