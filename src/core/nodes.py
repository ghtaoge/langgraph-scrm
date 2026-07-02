"""通用节点函数基类 — 提供错误处理和日志记录的标准模式"""
import logging

from src.core.llm import create_error_state

logger = logging.getLogger("langgraph-scrm")


def create_error_node(error_msg: str, node_name: str) -> dict:
    """创建错误节点返回值（各模块节点内部使用）

    Args:
        error_msg: 错误信息
        node_name: 节点名

    Returns:
        包含 error 和 error_node 的 dict
    """
    logger.error(f"节点 {node_name} 错误: {error_msg}")
    return create_error_state(error_msg, node_name)
