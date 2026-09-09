from .rag import retrieve


TOOL_HINTS = {
    "search_policy": "江苏医保政策依据",
    "query_drug_catalog": "江苏医保药品目录 双通道 国谈药",
    "query_hospital": "江苏医保定点医疗机构 医院 异地联网结算",
    "query_pharmacy": "江苏医保定点零售药店 双通道药店",
    "get_service_guide": "江苏医保政务服务办事指南 办理条件 流程",
    "calculate_reimbursement": "江苏医保报销待遇 起付线 支付比例 封顶线",
    "verify_policy_status": "江苏医保现行政策 生效日期 有效期 废止",
    "generate_material_list": "江苏医保办理材料 申请材料 清单",
}


def run_tool(name: str, query: str) -> list[dict]:
    if name not in TOOL_HINTS:
        raise ValueError(f"未知工具：{name}")
    return retrieve(f"{TOOL_HINTS[name]} {query}")
