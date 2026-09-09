import re
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from .tools import run_tool

INTENTS = {
    "cross_region": r"异地|跨省|备案|转诊", "family_mutual": r"家庭共济|共济|个人账户.*家人",
    "maternity_allowance": r"生育津贴|产假工资", "maternity_medical": r"生育医疗|产检|分娩",
    "chronic_disease": r"慢特病|慢性病|特殊病", "drug": r"双通道|药品目录|特药",
    "transfer": r"转移接续|医保关系转移", "reimbursement": r"报销|结算|费用",
    "enrollment": r"参保|缴费|医保登记",
}
INTENT_LABELS = {
    "cross_region": "江苏医保异地就医备案", "family_mutual": "江苏职工医保个人账户家庭共济",
    "maternity_allowance": "江苏生育津贴", "maternity_medical": "江苏生育医疗费用",
    "chronic_disease": "江苏医保门诊慢特病", "drug": "江苏医保双通道药品",
    "transfer": "江苏基本医疗保险关系转移接续", "reimbursement": "江苏医保费用报销",
    "enrollment": "江苏基本医疗保险参保",
}
MEDICAL_SCOPE = re.compile(r"医保|医疗保险|异地就医|生育|参保|报销|共济|慢特病|双通道|药品目录")
JIANGSU_CITIES = {"南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁"}
# 只收录用于槽位识别的常见就医城市；政策归属始终以参保地为准。
OTHER_CITIES = {"上海", "北京", "天津", "重庆", "杭州", "宁波", "合肥", "济南", "青岛", "郑州", "武汉", "长沙", "南昌", "福州", "厦门", "广州", "深圳", "成都", "西安"}
CITY_PATTERN = "|".join(sorted(JIANGSU_CITIES | OTHER_CITIES, key=len, reverse=True))


class AgentState(TypedDict, total=False):
    message: str
    history: list[dict]
    intent: str
    slots: dict
    missing_slots: list[str]
    query: str
    documents: list[dict]
    retry_count: int
    evidence_sufficient: bool
    evidence_reason: str
    response_type: str
    response_message: str
    retrieval_error: str
    trace: list[str]
    selected_tool: str
    tool_calls: list[dict]


def traced(state: AgentState, node: str, **updates) -> dict:
    return {**updates, "trace": [*state.get("trace", []), node]}


def classify(state: AgentState) -> dict:
    intent = next((name for name, pattern in INTENTS.items() if re.search(pattern, state["message"])), "unknown")
    return traced(state, "classify_intent", intent=intent, retry_count=0)


def check_scope(state: AgentState) -> dict:
    in_scope = state["intent"] != "unknown" or bool(MEDICAL_SCOPE.search(state["message"]))
    return traced(state, "check_scope", response_type="continue" if in_scope else "reject")


def reject_out_of_scope(state: AgentState) -> dict:
    return traced(state, "reject_out_of_scope", response_type="reject",
                  response_message="当前只支持江苏省基本医疗保险相关咨询。您可以询问参保、异地就医、费用报销、生育待遇、家庭共济、门诊慢特病或双通道药品等事项。")


def extract_slots(state: AgentState) -> dict:
    text = " ".join([item.get("content", "") for item in state.get("history", [])[-6:]] + [state["message"]])
    cities = re.findall(f"({CITY_PATTERN})市?", text)
    insured_city, care_city = (cities[0] if cities else None), (cities[1] if len(cities) > 1 else None)
    if len(cities) == 1 and re.search(fr"(?:去|到|前往){cities[0]}", text):
        insured_city, care_city = None, cities[0]
    insurance = "职工医保" if "职工医保" in text else "居民医保" if "居民医保" in text else None
    return traced(state, "extract_slots", slots={"insured_city": insured_city, "care_city": care_city, "insurance": insurance})


def check_slots(state: AgentState) -> dict:
    required = {"cross_region": ["insured_city", "care_city"]}.get(state["intent"], [])
    missing = [name for name in required if not state.get("slots", {}).get(name)]
    questions = {"insured_city": "请问您的医保参保城市是哪里？", "care_city": "请问您准备去哪个城市就医？"}
    if missing:
        return traced(state, "check_slots", missing_slots=missing, response_type="clarify",
                      response_message=questions[missing[0]])

    slots = state.get("slots", {})
    insured_city, care_city = slots.get("insured_city"), slots.get("care_city")
    if state["intent"] == "cross_region" and insured_city not in JIANGSU_CITIES:
        return traced(state, "check_slots", missing_slots=[], response_type="reject",
                      response_message=f"您的参保地是{insured_city}，不属于江苏省。当前 Agent 只提供江苏参保人员的医保政策咨询，请咨询参保地医保部门。")
    if state["intent"] == "cross_region" and insured_city == care_city:
        return traced(state, "check_slots", missing_slots=[], response_type="local_care",
                      response_message=f"您的参保地和就医地均为{insured_city}，不属于异地就医，一般无需办理异地就医备案。具体结算规则请以当地医保部门和定点医疗机构为准。")
    if state["intent"] == "cross_region":
        region_type = "intra_province" if care_city in JIANGSU_CITIES else "cross_province"
        slots = {**slots, "region_type": region_type}
    return traced(state, "check_slots", slots=slots, missing_slots=[], response_type="continue", response_message="")


def build_query(state: AgentState) -> dict:
    history = " ".join(item.get("content", "") for item in state.get("history", [])[-4:] if item.get("role") == "user")
    region_label = {"intra_province": "江苏省内异地就医", "cross_province": "跨省异地就医"}.get(
        state.get("slots", {}).get("region_type"), "")
    return traced(state, "build_query", query=f"{region_label} {history} {state['message']}".strip())


def select_tool(state: AgentState) -> dict:
    text = state["message"]
    if re.search(r"报销比例|起付线|封顶线|试算", text):
        tool = "calculate_reimbursement"
    elif re.search(r"有效|失效|废止|现行", text):
        tool = "verify_policy_status"
    elif re.search(r"材料|清单", text):
        tool = "generate_material_list"
    elif re.search(r"药品|双通道|特药|目录", text):
        tool = "query_drug_catalog"
    elif re.search(r"医院|医疗机构", text):
        tool = "query_hospital"
    elif re.search(r"药店", text):
        tool = "query_pharmacy"
    elif state["intent"] in {"cross_region", "transfer", "enrollment", "chronic_disease"}:
        tool = "get_service_guide"
    else:
        tool = "search_policy"
    return traced(state, "select_tool", selected_tool=tool, tool_calls=[])


def execute_tool(state: AgentState) -> dict:
    name = state["selected_tool"]
    try:
        documents = run_tool(name, state["query"])
        call = {"name": name, "status": "success", "results": len(documents)}
        return traced(state, "execute_tool", documents=documents, retrieval_error="", tool_calls=[*state.get("tool_calls", []), call])
    except Exception as error:
        call = {"name": name, "status": "error", "results": 0}
        return traced(state, "execute_tool", documents=[], retrieval_error=type(error).__name__, tool_calls=[*state.get("tool_calls", []), call])


def assess_evidence(state: AgentState) -> dict:
    documents = state.get("documents", [])
    keyword_score = max((float(item.get("keyword_score", 0)) for item in documents), default=0)
    vector_score = max((float(item.get("vector_score", 0)) for item in documents), default=0)
    sufficient = bool(documents) and (keyword_score >= 0.08 or vector_score >= 0.42)
    if state.get("retrieval_error"):
        reason = f"检索服务异常：{state['retrieval_error']}"
    elif not documents:
        reason = "没有检索到已发布资料"
    elif sufficient:
        reason = "关键词或语义相关度达到回答阈值"
    else:
        reason = "候选资料相关度不足"
    return traced(state, "assess_evidence", evidence_sufficient=sufficient, evidence_reason=reason,
                  response_type="answer" if sufficient else "retry")


def rewrite_query(state: AgentState) -> dict:
    slots = " ".join(value for value in state.get("slots", {}).values() if value)
    query = f"{INTENT_LABELS.get(state['intent'], '江苏医保')} {slots} {state['message']} 办理条件 材料 流程".strip()
    return traced(state, "rewrite_query", query=query, retry_count=state.get("retry_count", 0) + 1)


def fallback(state: AgentState) -> dict:
    return traced(state, "fallback", response_type="fallback",
                  response_message="当前知识库没有检索到足够可靠的江苏医保依据，暂时无法给出确定结论。请补充具体医保事项和参保城市，或通过当地医保部门官方渠道核验。")


def route_scope(state: AgentState) -> str:
    return "continue" if state["response_type"] == "continue" else "reject"


def route_slots(state: AgentState) -> str:
    return "continue" if state.get("response_type") == "continue" else "stop"


def route_evidence(state: AgentState) -> str:
    if state.get("evidence_sufficient"):
        return "answer"
    return "rewrite" if state.get("retry_count", 0) < 1 and not state.get("retrieval_error") else "fallback"


graph = StateGraph(AgentState)
for name, node in {
    "classify_intent": classify, "check_scope": check_scope, "reject_out_of_scope": reject_out_of_scope,
    "extract_slots": extract_slots, "check_slots": check_slots, "build_query": build_query,
    "select_tool": select_tool, "execute_tool": execute_tool, "assess_evidence": assess_evidence, "rewrite_query": rewrite_query,
    "fallback": fallback,
}.items():
    graph.add_node(name, node)
graph.add_edge(START, "classify_intent")
graph.add_edge("classify_intent", "check_scope")
graph.add_conditional_edges("check_scope", route_scope, {"continue": "extract_slots", "reject": "reject_out_of_scope"})
graph.add_edge("reject_out_of_scope", END)
graph.add_edge("extract_slots", "check_slots")
graph.add_conditional_edges("check_slots", route_slots, {"continue": "build_query", "stop": END})
graph.add_edge("build_query", "select_tool")
graph.add_edge("select_tool", "execute_tool")
graph.add_edge("execute_tool", "assess_evidence")
graph.add_conditional_edges("assess_evidence", route_evidence, {"answer": END, "rewrite": "rewrite_query", "fallback": "fallback"})
graph.add_edge("rewrite_query", "execute_tool")
graph.add_edge("fallback", END)
agent_graph = graph.compile()
