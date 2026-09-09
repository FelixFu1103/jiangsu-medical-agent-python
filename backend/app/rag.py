from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
import re
from .config import get_settings
from .db import chunks


@lru_cache
def embeddings():
    settings = get_settings()
    cache = str(Path(settings.hf_cache_dir).resolve())
    os.environ.setdefault("HF_HOME", cache)
    os.environ.setdefault("HF_XET_CACHE", f"{cache}/xet")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model, cache_folder=settings.hf_cache_dir,
        model_kwargs={"device": settings.embedding_device}, encode_kwargs={"normalize_embeddings": True})


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


def lexical_score(query: str, text: str) -> float:
    tokens = set(re.findall(r"[\u4e00-\u9fff]{2}|[a-zA-Z0-9]+", query.lower()))
    if not tokens:
        return 0.0
    haystack = text.lower()
    return sum(token in haystack for token in tokens) / len(tokens)


def rrf(keyword_rows: list[dict], vector_rows: list[dict], k: int = 60) -> list[dict]:
    merged: dict[str, dict] = {}
    for channel, rows in (("keyword", keyword_rows), ("vector", vector_rows)):
        for rank, row in enumerate(rows, 1):
            item = merged.setdefault(row["chunk_id"], {**row, "score": 0.0, "channels": [], "keyword_score": 0.0, "vector_score": 0.0})
            item["score"] += 1 / (k + rank)
            item["channels"].append(channel)
            item[f"{channel}_score"] = float(row.get("raw_score", 0))
    return sorted(merged.values(), key=lambda item: (item["score"], item["vector_score"] + item["keyword_score"]), reverse=True)


def _row(chunk_id: str, content: str, metadata: dict, raw_score: float) -> dict:
    return {"chunk_id": chunk_id, "document_id": metadata["document_id"], "title": metadata["title"],
            "department": metadata["department"], "source_url": metadata["source_url"],
            "content": content, "chunk_index": metadata["chunk_index"], "raw_score": raw_score}


def retrieve(query: str, limit: int = 3) -> list[dict]:
    store = chunks()
    # ponytail: scans published text for lexical recall; add a dedicated BM25 service only when corpus size proves this too slow.
    all_rows = store.get(where={"status": "published"}, include=["documents", "metadatas"])
    keyword = sorted((_row(i, text, meta, lexical_score(query, text))
                      for i, text, meta in zip(all_rows["ids"], all_rows["documents"], all_rows["metadatas"])),
                     key=lambda item: item["raw_score"], reverse=True)
    keyword = [item for item in keyword if item["raw_score"] > 0][:12]
    if not all_rows["ids"]:
        return []
    vector = store.query(query_embeddings=[embeddings().embed_query(query)], n_results=min(12, len(all_rows["ids"])),
                         where={"status": "published"}, include=["documents", "metadatas", "distances"])
    vector_rows = [_row(i, text, meta, 1 - distance) for i, text, meta, distance in
                   zip(vector["ids"][0], vector["documents"][0], vector["metadatas"][0], vector["distances"][0])]
    candidates = rrf(keyword, vector_rows)
    model = reranker()
    if model and len(candidates) > 1:
        scores = model.predict([(query, item["content"]) for item in candidates])
        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(score)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates[:limit]
