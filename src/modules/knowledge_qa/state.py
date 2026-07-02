"""知识库问答模块 State 定义"""
from typing import Annotated, Optional, TypedDict, operator


class KnowledgeQAState(TypedDict):
    """知识库问答状态 — Corrective RAG 模式

    LangGraph 知识点:
    - RAG 流程的 state 需要追踪文档、答案和验证结果
    - documents 和 web_results 使用 reducer 追加模式
    """
    question: str                           # 用户问题
    documents: Annotated[list[str], operator.add]   # 检索到的文档片段
    web_results: Annotated[list[str], operator.add] # 补充网页搜索结果
    answer: str                             # 生成的回答
    citations: Annotated[list[dict], operator.add]  # 引用来源
    verification: str                       # 自检结果: passed/failed
    retries: int                            # 生成重试次数
    error: Optional[str]
    error_node: Optional[str]
