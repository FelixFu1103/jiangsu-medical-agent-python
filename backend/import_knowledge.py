from pathlib import Path
import yaml
from app.db import close_pool, connection, open_pool
from app.ingestion import content_hash, split_text
from app.rag import embed_documents


ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "documents"


def read_document(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        return {"title": path.stem}, raw.strip()
    _, header, body = raw.split("---", 2)
    return yaml.safe_load(header) or {}, body.strip()


def import_all() -> None:
    paths = sorted(ROOT.glob("*.md"))
    open_pool()
    total_chunks = 0
    try:
        for number, path in enumerate(paths, 1):
            metadata, body = read_document(path)
            title = metadata.get("title", path.stem)
            department = metadata.get("department", "江苏省医疗保障局")
            source = metadata.get("source", "")
            chunks = split_text(body)
            searchable = [f"{title} {metadata.get('topic', '')} {metadata.get('keywords', '')} {chunk}" for chunk in chunks]
            vectors = embed_documents(searchable)
            with connection() as conn, conn.transaction():
                document_id = conn.execute("""INSERT INTO knowledge_documents
                  (content_hash,title,department,region,source_url,raw_content,status)
                  VALUES (%s,%s,%s,%s,%s,%s,'published')
                  ON CONFLICT(content_hash) DO UPDATE SET title=EXCLUDED.title,department=EXCLUDED.department,
                    region=EXCLUDED.region,source_url=EXCLUDED.source_url,raw_content=EXCLUDED.raw_content,
                    status='published',updated_at=NOW() RETURNING id""",
                    (content_hash(body), title, department, metadata.get("region", "江苏省"), source, body)).fetchone()[0]
                conn.execute("DELETE FROM knowledge_chunks WHERE document_id=%s", (document_id,))
                with conn.cursor() as cursor:
                    cursor.executemany("""INSERT INTO knowledge_chunks
                      (document_id,chunk_index,content,embedding) VALUES (%s,%s,%s,%s)""",
                      [(document_id, index, chunk, vector) for index, (chunk, vector) in enumerate(zip(chunks, vectors))])
            total_chunks += len(chunks)
            print(f"[{number}/{len(paths)}] {title}: {len(chunks)} chunks", flush=True)
    finally:
        close_pool()
    print(f"Imported {len(paths)} documents and {total_chunks} vector chunks.")


if __name__ == "__main__":
    import_all()
