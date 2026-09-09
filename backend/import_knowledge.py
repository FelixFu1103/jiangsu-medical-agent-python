from pathlib import Path
import yaml
from app.main import publish_knowledge, save_draft


ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "documents"


def read_document(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        return {"title": path.stem}, raw.strip()
    _, header, body = raw.split("---", 2)
    return yaml.safe_load(header) or {}, body.strip()


def import_all() -> None:
    paths = sorted(ROOT.glob("*.md"))
    total_chunks = 0
    for number, path in enumerate(paths, 1):
        metadata, body = read_document(path)
        result = save_draft(metadata.get("title", path.stem), metadata.get("department", "江苏省医疗保障局"),
                            metadata.get("source", ""), body)
        publish_knowledge(result["id"])
        total_chunks += result["chunks"]
        print(f"[{number}/{len(paths)}] {result['title']}: {result['chunks']} chunks", flush=True)
    print(f"Imported {len(paths)} documents and {total_chunks} vector chunks.")


if __name__ == "__main__":
    import_all()
