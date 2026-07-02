"""知识库问答模块节点函数 — Corrective RAG 模式

LangGraph 知识点:
- ToolNode：LangGraph 内置的工具执行节点
- RAG 流程：retrieve → grade → generate → verify → (循环重试)
"""
import logging

from langchain_core.messages import HumanMessage

from src.config.prompts import (
    KNOWLEDGE_QA_GENERATE_PROMPT,
    KNOWLEDGE_QA_GRADE_PROMPT,
    KNOWLEDGE_QA_VERIFY_PROMPT,
)
from src.core.llm import get_llm, safe_llm_call
from src.modules.knowledge_qa.tools import get_retriever_tool

logger = logging.getLogger("langgraph-scrm.knowledge_qa")


@safe_llm_call
def retrieve_node(state: dict) -> dict:
    """检索节点 — 向量检索相关文档"""
    retriever_tool = get_retriever_tool()
    docs = retriever_tool.invoke(state["question"])
    return {"documents": [docs]}


@safe_llm_call
def grade_docs_node(state: dict) -> dict:
    """文档评估节点 — 评估检索文档相关性"""
    llm = get_llm()
    prompt = KNOWLEDGE_QA_GRADE_PROMPT.format(
        question=state["question"],
        documents="\n".join(state.get("documents", [])),
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    if "irrelevant" in response.content.lower():
        logger.info("文档相关性不足，将补充网页搜索")
        return {"verification": "docs_irrelevant"}
    return {"verification": "docs_relevant"}


@safe_llm_call
def web_search_node(state: dict) -> dict:
    """网页搜索节点 — 文档不足时的补充检索"""
    from src.core.tools import web_search
    result = web_search.invoke(state["question"])
    return {"web_results": [result]}


@safe_llm_call
def generate_node(state: dict) -> dict:
    """回答生成节点 — 基于文档和搜索结果生成回答"""
    llm = get_llm()
    all_docs = "\n".join(state.get("documents", []) + state.get("web_results", []))
    prompt = KNOWLEDGE_QA_GENERATE_PROMPT.format(
        question=state["question"],
        documents=all_docs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content, "retries": state.get("retries", 0) + 1}


@safe_llm_call
def verify_node(state: dict) -> dict:
    """自检节点 — 检查回答是否有幻觉"""
    llm = get_llm()
    all_docs = "\n".join(state.get("documents", []) + state.get("web_results", []))
    prompt = KNOWLEDGE_QA_VERIFY_PROMPT.format(
        question=state["question"],
        answer=state["answer"],
        documents=all_docs,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    verification = "passed" if "passed" in response.content.lower() else "failed"
    return {"verification": verification}


def respond_node(state: dict) -> dict:
    """最终回复节点"""
    return {}


# ── 条件路由函数 ──
def route_after_grade(state: dict) -> str:
    """文档评估后路由"""
    if state.get("verification") == "docs_irrelevant":
        return "web_search"
    return "generate"


def route_after_verify(state: dict) -> str:
    """自检后路由 — 重试上限 3 次"""
    if state.get("verification") == "failed" and state.get("retries", 0) < 3:
        return "generate"
    return "respond"
