from __future__ import annotations
import asyncio
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .agent import agent_graph
from .config import get_settings
from .db import chunks as chunk_store, close_db, documents as document_store, open_db
from .ingestion import content_hash, extract_text, split_text
from .rag import embed_documents


SYSTEM_PROMPT = """你是江苏医保智能咨询Agent。只能依据提供的已发布资料回答，不得编造政策、材料、费用或时限。
严格使用“## 结论”“## 办理建议”“## 需要确认”三个Markdown标题，每段最多3点，不要输出其他标题或大段连续文字。关键结论使用[资料1]标注依据。知识库没有答案时明确说明。
不得要求身份证号、银行卡号、密码或验证码，个案结果以当地医保部门答复为准。"""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


class MedicalAnswer(BaseModel):
    conclusion: str
    steps: list[str] = Field(default_factory=list)
    confirmations: list[str] = Field(default_factory=list)


def structured_answer(answer: str) -> dict:
    clean = re.sub(r"\*\*(结论|办理建议|需要确认)\*\*", r"\1", answer)
    parts = re.split(r"(?:^|\n)#{0,3}\s*(结论|办理建议|需要确认)[：:]?\s*", clean)
    sections = {parts[index]: parts[index + 1].strip() for index in range(1, len(parts) - 1, 2)}

    def items(name: str) -> list[str]:
        return [re.sub(r"^[-*•\d.、\s]+", "", line).strip() for line in sections.get(name, "").splitlines() if line.strip()]

    conclusion = " ".join(items("结论")) or answer.strip()
    return MedicalAnswer(conclusion=conclusion, steps=items("办理建议"),
                         confirmations=items("需要确认")).model_dump()


def agent_metadata(state: dict) -> dict:
    return {"responseType": state.get("response_type"), "missingSlots": state.get("missing_slots", []),
            "evidenceReason": state.get("evidence_reason"), "retryCount": state.get("retry_count", 0),
            "selectedTool": state.get("selected_tool"), "toolCalls": state.get("tool_calls", []),
            "trace": state.get("trace", [])}


def event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def immediate_stream(state: dict) -> StreamingResponse:
    message = state.get("response_message", "暂时无法处理该问题。")

    async def stream():
        yield event("delta", {"text": message})
        yield event("done", {"answer": message, "structured": structured_answer(message), "intent": state.get("intent"),
                             "slots": state.get("slots", {}), "sources": [], "agent": agent_metadata(state)})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def save_draft(title: str, department: str, source_url: str, text: str) -> dict:
    if len(text) < 30:
        raise ValueError("未提取到足够的正文")
    chunks = split_text(text)
    vectors = embed_documents([f"{title} {department} {part}" for part in chunks])
    digest = content_hash(text)
    now = datetime.now(timezone.utc).isoformat()
    existing = document_store().get(ids=[digest], include=["metadatas"])
    created_at = existing["metadatas"][0].get("created_at", now) if existing["ids"] else now
    metadata = {"content_hash": digest, "title": title, "department": department, "region": "江苏省",
                "source_url": source_url, "status": "draft", "created_at": created_at, "updated_at": now}
    document_store().upsert(ids=[digest], embeddings=[[0.0]], documents=[text], metadatas=[metadata])
    store = chunk_store()
    old = store.get(where={"document_id": digest})
    if old["ids"]:
        store.delete(ids=old["ids"])
    chunk_metadata = [{"document_id": digest, "title": title, "department": department,
                       "source_url": source_url, "status": "draft", "chunk_index": index}
                      for index in range(len(chunks))]
    store.upsert(ids=[f"{digest}:{index}" for index in range(len(chunks))], embeddings=vectors,
                 documents=chunks, metadatas=chunk_metadata)
    return {"id": digest, "title": title, "status": "draft", "characters": len(text), "chunks": len(chunks)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    open_db()
    yield
    close_db()


app = FastAPI(title="江苏医保 Agent API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    published = document_store().get(where={"status": "published"})
    return {"ok": True, "database": "ChromaDB", "publishedDocuments": len(published["ids"]),
            "chunks": chunk_store().count(), "embeddingModel": get_settings().embedding_model,
            "framework": "FastAPI + LangChain + LangGraph + ChromaDB"}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise HTTPException(503, "尚未配置DEEPSEEK_API_KEY")
    state = await asyncio.to_thread(agent_graph.invoke, {"message": body.message, "history": body.history[-10:]})
    if state.get("response_type") != "answer":
        return immediate_stream(state)
    documents = state.get("documents", [])
    context = "\n\n".join(
        f"[资料{i}]\n标题：{doc['title']}\n部门：{doc['department']}\n来源：{doc['source_url']}\n内容：{doc['content']}"
        for i, doc in enumerate(documents, 1)
    ) or "未检索到相关资料。"
    model = ChatOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url,
                       model=settings.deepseek_model, temperature=0, streaming=True)
    prompt = f"意图：{state.get('intent')}\n已提取信息：{state.get('slots')}\n证据判断：{state.get('evidence_reason')}\n检索资料：\n{context}\n\n用户问题：{body.message}"

    async def stream():
        answer = ""
        try:
            async for chunk in model.astream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    answer += text
                    yield event("delta", {"text": text})
            yield event("done", {"answer": answer, "structured": structured_answer(answer),
                                  "intent": state.get("intent"), "slots": state.get("slots"),
                                  "sources": [{"title": d["title"], "url": d["source_url"], "chunkId": d["chunk_id"]} for d in documents],
                                  "agent": agent_metadata(state)})
        except Exception:
            yield event("error", {"error": "AI服务暂时不可用"})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/api/knowledge/upload")
async def upload_knowledge(title: str = Form(...), department: str = Form("江苏省医疗保障局"),
                           source_url: str = Form(""), file: Optional[UploadFile] = File(None)):
    try:
        if file:
            content = await file.read(10 * 1024 * 1024 + 1)
            if len(content) > 10 * 1024 * 1024:
                raise ValueError("文件不能超过10MB")
            text = await asyncio.to_thread(extract_text, file.filename or "upload.txt", content)
        elif source_url:
            host = (urlparse(source_url).hostname or "").lower()
            if not source_url.startswith("https://") or not (host.endswith(".gov.cn") or host == "gov.cn"):
                raise ValueError("仅允许HTTPS政府网站链接")
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(source_url)
                response.raise_for_status()
                if len(response.content) > 10 * 1024 * 1024:
                    raise ValueError("网页内容不能超过10MB")
                text = extract_text("page.html", response.content)
        else:
            raise ValueError("请选择文件或填写政府网页链接")
        return await asyncio.to_thread(save_draft, title.strip(), department.strip(), source_url.strip(), text)
    except (ValueError, httpx.HTTPError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/knowledge")
def list_knowledge():
    docs = document_store().get(include=["metadatas"])
    stored_chunks = chunk_store().get(include=["metadatas"])
    counts = {}
    for metadata in stored_chunks["metadatas"]:
        counts[metadata["document_id"]] = counts.get(metadata["document_id"], 0) + 1
    rows = [{"id": doc_id, "title": meta["title"], "department": meta["department"],
             "source": meta["source_url"], "status": meta["status"], "updatedAt": meta["updated_at"],
             "chunks": counts.get(doc_id, 0)} for doc_id, meta in zip(docs["ids"], docs["metadatas"])]
    return sorted(rows, key=lambda row: (row["status"] != "draft", row["updatedAt"]), reverse=False)[:100]


@app.get("/api/knowledge/{document_id}")
def get_knowledge(document_id: str):
    document = document_store().get(ids=[document_id], include=["documents", "metadatas"])
    if not document["ids"]:
        raise HTTPException(404, "资料不存在")
    meta = document["metadatas"][0]
    stored = chunk_store().get(where={"document_id": document_id}, include=["documents", "metadatas"])
    pieces = sorted(({"id": chunk_id, "chunkIndex": item["chunk_index"], "content": content, "vectorized": True}
                     for chunk_id, content, item in zip(stored["ids"], stored["documents"], stored["metadatas"])),
                    key=lambda item: item["chunkIndex"])
    return {"id": document_id, "title": meta["title"], "department": meta["department"],
            "source": meta["source_url"], "status": meta["status"], "body": document["documents"][0], "chunks": pieces}


@app.post("/api/knowledge/{document_id}/publish")
def publish_knowledge(document_id: str):
    document = document_store().get(ids=[document_id], include=["documents", "metadatas"])
    if not document["ids"] or document["metadatas"][0]["status"] != "draft":
        raise HTTPException(409, "资料不存在或不是草稿")
    metadata = {**document["metadatas"][0], "status": "published", "updated_at": datetime.now(timezone.utc).isoformat()}
    document_store().update(ids=[document_id], metadatas=[metadata])
    stored = chunk_store().get(where={"document_id": document_id}, include=["metadatas"])
    if stored["ids"]:
        chunk_store().update(ids=stored["ids"], metadatas=[{**item, "status": "published"} for item in stored["metadatas"]])
    return {"id": document_id, "title": metadata["title"], "status": "published"}
