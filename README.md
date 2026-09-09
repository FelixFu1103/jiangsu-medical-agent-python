# 江苏医保 Agent（Python版）

这是与原Node.js项目并列的独立实现：Vue 3前端，FastAPI后端，LangGraph编排，LangChain连接DeepSeek与HuggingFace，ChromaDB本地持久化文档、状态、片段和向量。

## 已实现

- HuggingFace `BAAI/bge-m3` 本地1024维Embedding
- ChromaDB余弦向量检索 + 词面检索 + RRF融合
- 可选 `BAAI/bge-reranker-v2-m3` Cross-Encoder精排
- LangGraph条件工作流：范围判断、意图识别、槽位追问、政策检索、证据判断、一次Query改写重试与安全降级
- 8个医保知识工具及LangGraph工具选择/执行节点
- 结构化回答接口：结论、办理建议、需要确认
- 前端Agent执行详情：工具、路径与证据判断
- DeepSeek通过LangChain进行SSE流式回答
- PDF、DOCX、TXT、Markdown、HTML和政府HTTPS网页导入
- 上传为草稿、正文与片段预览、人工确认发布
- 只有`published`资料参与RAG

## 启动

要求Python 3.9+和Node.js 20+，不再需要Docker Desktop或PostgreSQL。

```bash
cd /Users/macpor/Documents/Codex/2026-09-01/wo-2/outputs/jiangsu-medical-agent-python
cp .env.example .env
```

在`.env`中填写：

```env
DEEPSEEK_API_KEY=你的key
```

启动后端：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

第一次运行会下载`bge-m3`模型，模型较大，需要等待。启动前端：

```bash
cd ../frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。

导入随项目提供的原知识库：

```bash
cd backend
source .venv/bin/activate
python import_knowledge.py
```

## 验证

```bash
cd backend
source .venv/bin/activate
pytest -q
```

接口文档：`http://127.0.0.1:8000/docs`。

## 目录

```text
backend/app/main.py       FastAPI、SSE、知识库审核接口
backend/app/agent.py      LangGraph工作流、意图与槽位
backend/app/rag.py        bge-m3、混合检索、RRF与Reranker
backend/app/ingestion.py  多格式解析与LangChain切分
backend/app/db.py         ChromaDB持久化客户端与集合
frontend/src/App.vue      问答、上传、预览、发布页面
```

当前会话状态随每次请求携带，尚未增加Redis持久化；权限系统也尚未实现，生产环境发布接口必须增加登录、角色和审计。

当前LangGraph采用受控有限循环：异地就医缺少参保地或就医地时直接追问；非医保问题直接拒绝；证据分数不足时只改写一次Query，第二次仍不足则拒绝给出确定结论。节点执行轨迹、证据原因和重试次数会随SSE的`done`事件返回。
