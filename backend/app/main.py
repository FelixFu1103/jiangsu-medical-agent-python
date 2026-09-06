from __future__ import annotations
import asyncio
import json
from contextlib import asynccontextmanager
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
from .db import close_pool, connection, open_pool
from .ingestion import content_hash, extract_text, split_text
from .rag import embed_documents


SYSTEM_PROMPT = """你是江苏医保智能咨询Agent。只能依据提供的已发布资料回答，不得编造政策、材料、费用或时限。
用简洁 Markdown 输出，“结论”“办理建议”“需要确认”各自单独换行，每段最多3点；不要输出大段连续文字。关键结论使用[资料1]标注依据。知识库没有答案时明确说明。
不得要求身份证号、银行卡号、密码或验证码，个案结果以当地医保部门答复为准。"""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


def event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def immediate_stream(state: dict) -> StreamingResponse:
    message = state.get("response_message", "暂时无法处理该问题。")

    async def stream():
        yield event("delta", {"text": message})
        yield event("done", {"answer": message, "intent": state.get("intent"), "slots": state.get("slots", {}),
                             "sources": [], "agent": {"responseType": state.get("response_type"),
                             "missingSlots": state.get("missing_slots", []), "evidenceReason": state.get("evidence_reason"),
                             "retryCount": state.get("retry_count", 0), "trace": state.get("trace", [])}})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


def save_draft(title: str, department: str, source_url: str, text: str) -> dict:
    if len(text) < 30:
        raise ValueError("未提取到足够的正文")
    chunks = split_text(text)
    vectors = embed_documents([f"{title} {department} {part}" for part in chunks])
    digest = content_hash(text)
    with connection() as conn:
        with conn.transaction():
            row = conn.execute("""INSERT INTO knowledge_documents
              (content_hash,title,department,source_url,raw_content,status)
              VALUES (%s,%s,%s,%s,%s,'draft')
              ON CONFLICT(content_hash) DO UPDATE SET title=EXCLUDED.title, department=EXCLUDED.department,
                source_url=EXCLUDED.source_url, raw_content=EXCLUDED.raw_content, updated_at=NOW()
              RETURNING id,status""", (digest, title, department, source_url, text)).fetchone()
            document_id, status = row
            conn.execute("DELETE FROM knowledge_chunks WHERE document_id=%s", (document_id,))
            with conn.cursor() as cursor:
                cursor.executemany("""INSERT INTO knowledge_chunks
                  (document_id,chunk_index,content,embedding) VALUES (%s,%s,%s,%s)""",
                  [(document_id, index, chunk, vector) for index, (chunk, vector) in enumerate(zip(chunks, vectors))])
    return {"id": str(document_id), "title": title, "status": status, "characters": len(text), "chunks": len(chunks)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    open_pool()
    yield
    close_pool()


app = FastAPI(title="江苏医保 Agent API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    with connection() as conn:
        documents, chunks = conn.execute("""SELECT
          (SELECT count(*) FROM knowledge_documents WHERE status='published'),
          (SELECT count(*) FROM knowledge_chunks)""").fetchone()
    return {"ok": True, "database": True, "publishedDocuments": documents, "chunks": chunks,
            "embeddingModel": get_settings().embedding_model, "framework": "FastAPI + LangChain + LangGraph"}


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
            yield event("done", {"answer": answer, "intent": state.get("intent"), "slots": state.get("slots"),
                                  "sources": [{"title": d["title"], "url": d["source_url"], "chunkId": d["chunk_id"]} for d in documents],
                                  "agent": {"responseType": "answer", "evidenceReason": state.get("evidence_reason"),
                                  "retryCount": state.get("retry_count", 0), "trace": state.get("trace", [])}})
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
    with connection() as conn:
        rows = conn.execute("""SELECT d.id,d.title,d.department,d.source_url,d.status,d.updated_at,count(c.id)
          FROM knowledge_documents d LEFT JOIN knowledge_chunks c ON c.document_id=d.id
          GROUP BY d.id ORDER BY CASE d.status WHEN 'draft' THEN 0 WHEN 'published' THEN 1 ELSE 2 END,d.updated_at DESC LIMIT 100""").fetchall()
    return [{"id": str(r[0]), "title": r[1], "department": r[2], "source": r[3], "status": r[4], "updatedAt": r[5], "chunks": r[6]} for r in rows]


@app.get("/api/knowledge/{document_id}")
def get_knowledge(document_id: int):
    with connection() as conn:
        document = conn.execute("SELECT id,title,department,source_url,status,raw_content FROM knowledge_documents WHERE id=%s", (document_id,)).fetchone()
        if not document:
            raise HTTPException(404, "资料不存在")
        chunks = conn.execute("SELECT id,chunk_index,content FROM knowledge_chunks WHERE document_id=%s ORDER BY chunk_index", (document_id,)).fetchall()
    return {"id": str(document[0]), "title": document[1], "department": document[2], "source": document[3],
            "status": document[4], "body": document[5], "chunks": [{"id": str(c[0]), "chunkIndex": c[1], "content": c[2], "vectorized": True} for c in chunks]}


@app.post("/api/knowledge/{document_id}/publish")
def publish_knowledge(document_id: int):
    with connection() as conn:
        row = conn.execute("UPDATE knowledge_documents SET status='published',updated_at=NOW() WHERE id=%s AND status='draft' RETURNING id,title,status", (document_id,)).fetchone()
        conn.commit()
    if not row:
        raise HTTPException(409, "资料不存在或不是草稿")
    return {"id": str(row[0]), "title": row[1], "status": row[2]}
