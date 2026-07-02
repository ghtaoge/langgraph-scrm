"""State 定义基类 — 所有模块共享的错误字段"""

from typing import Optional, TypedDict


class BaseState(TypedDict):
    """所有模块 State 的基类字段

    每个模块的 State 定义应继承或包含这些字段。
    LangGraph 的 TypedDict 不能直接继承（需要全部字段在同一个 TypedDict 中），
    所以这个基类只作为文档参考 — 各模块需显式包含 error 和 error_node 字段。
    """

    error: Optional[str]  # 错误信息（null = 无错误）
    error_node: Optional[str]  # 发生错误的节点名
