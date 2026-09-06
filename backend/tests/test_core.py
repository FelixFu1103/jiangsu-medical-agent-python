import app.agent as agent
from app.agent import agent_graph, assess_evidence, check_scope, check_slots, classify, extract_slots, route_evidence
from app.ingestion import split_text
from app.rag import as_pgvector, rrf
from import_knowledge import read_document


def test_intent_and_slots():
    assert classify({"message": "南京职工医保去上海住院怎么备案"})["intent"] == "cross_region"
    slots = extract_slots({"message": "南京职工医保去上海住院", "history": []})["slots"]
    assert slots == {"insured_city": "南京", "care_city": "上海", "insurance": "职工医保"}


def test_agent_routes(monkeypatch):
    assert check_scope({"message": "今天天气怎么样", "intent": "unknown"})["response_type"] == "reject"
    assert check_slots({"intent": "cross_region", "slots": {"care_city": "上海"}})["missing_slots"] == ["insured_city"]
    strong = assess_evidence({"documents": [{"vector_score": 0.7}], "retry_count": 0})
    assert strong["evidence_sufficient"] and route_evidence(strong) == "answer"
    monkeypatch.setattr(agent, "retrieve", lambda _: [{"chunk_id": 1, "vector_score": 0.7}])
    answered = agent_graph.invoke({"message": "南京职工医保去上海住院怎么备案", "history": []})
    assert answered["response_type"] == "answer" and answered["trace"][-1] == "assess_evidence"


def test_region_rules(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda _: [{"chunk_id": 1, "vector_score": 0.7}])
    cross_province = agent_graph.invoke({"message": "我参加南京职工医保，准备去上海住院，怎么备案？", "history": []})
    assert cross_province["slots"]["region_type"] == "cross_province"
    assert "跨省异地就医" in cross_province["query"]

    intra_province = agent_graph.invoke({"message": "我参加南京医保，准备去苏州住院，怎么备案？", "history": []})
    assert intra_province["slots"]["region_type"] == "intra_province"

    local = agent_graph.invoke({"message": "我参加南京医保，在南京住院怎么备案？", "history": []})
    assert local["response_type"] == "local_care"

    outside = agent_graph.invoke({"message": "我参加上海医保，去南京住院怎么备案？", "history": []})
    assert outside["response_type"] == "reject"


def test_agent_rewrites_once_then_falls_back(monkeypatch):
    monkeypatch.setattr(agent, "retrieve", lambda _: [{"chunk_id": 1, "vector_score": 0.1}])
    state = agent_graph.invoke({"message": "江苏医保关系转移怎么办", "history": []})
    assert state["response_type"] == "fallback"
    assert state["retry_count"] == 1 and state["trace"].count("retrieve_policy") == 2


def test_split_and_rrf():
    assert len(split_text("第一条。" * 200)) > 1
    result = rrf([{"chunk_id": 1}], [{"chunk_id": 1}, {"chunk_id": 2}])
    assert result[0]["chunk_id"] == 1 and result[0]["channels"] == ["keyword", "vector"]


def test_frontmatter(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("---\ntitle: 测试政策\ndepartment: 江苏省医疗保障局\n---\n正文内容", encoding="utf-8")
    metadata, body = read_document(path)
    assert metadata["title"] == "测试政策" and body == "正文内容"


def test_pgvector_query_parameter():
    vector = as_pgvector([0.1] * 1024)
    assert vector.dtype.name == "float32" and vector.shape == (1024,)
