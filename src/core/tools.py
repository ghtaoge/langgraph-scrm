"""通用工具注册 — 搜索/计算等基础工具"""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """网页搜索工具（用于知识问答模块的补充检索）

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    # 生产环境应接入真实搜索 API（如 Tavily/SerpAPI）
    # 这里返回占位结果，方便学习和测试
    return f"[搜索结果] 关键词 '{query}' 的搜索结果（生产环境请接入真实搜索 API）"


@tool
def calculate(expression: str) -> str:
    """简单计算工具

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return f"计算错误: 无法计算 '{expression}'"


# 工具名称映射（供 ToolNode 使用）
COMMON_TOOLS = [web_search, calculate]
