"""LLM 抽象层 — 统一接口，支持 OpenAI + Doubao 双后端切换"""

import asyncio
from functools import wraps
from typing import Callable, Optional

from langchain_openai import ChatOpenAI

from src.config.settings import LLMProvider, settings


def get_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """获取 LLM 实例

    Args:
        provider: LLM 提供者，默认从 Settings 读取
        model: 模型名，默认从 Settings 读取
        base_url: API base URL，默认从 Settings 读取
        temperature: 温度参数，默认 0（确定性输出）

    Returns:
        ChatOpenAI 实例（Doubao 也通过 OpenAI 兼容接口接入）
    """
    provider = provider or settings.LLM_PROVIDER
    if provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=model or settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=base_url or settings.OPENAI_BASE_URL,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    elif provider == LLMProvider.DOUBAO:
        # Doubao 使用 OpenAI 兼容接口接入
        return ChatOpenAI(
            model=model or settings.DOUBAO_MODEL,
            api_key=settings.DOUBAO_API_KEY,
            base_url=base_url or settings.DOUBAO_ENDPOINT,
            temperature=temperature,
            max_retries=settings.LLM_MAX_RETRIES,
        )
    else:
        raise ValueError(f"不支持的 LLM 提供者: {provider}")


def safe_llm_call(func: Callable) -> Callable:
    """LLM 调用安全装饰器 — 捕获异常并返回错误 state 字段

    用法：
        @safe_llm_call
        def my_node(state):
            result = llm.invoke(...)
            return {"response": result.content}
    """

    @wraps(func)
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            node_name = func.__name__
            return {
                "error": f"节点 {node_name} 执行失败: {str(e)}",
                "error_node": node_name,
            }

    @wraps(func)
    async def async_wrapper(state):
        try:
            return await func(state)
        except Exception as e:
            node_name = func.__name__
            return {
                "error": f"节点 {node_name} 执行失败: {str(e)}",
                "error_node": node_name,
            }

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


def create_error_state(error_msg: str, node_name: str) -> dict:
    """创建错误 state 字段（用于节点内部异常处理）"""
    return {"error": error_msg, "error_node": node_name}
