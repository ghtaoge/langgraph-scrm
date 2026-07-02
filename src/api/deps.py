"""API 共享依赖 — 进程内共享 Checkpointer（用于 interrupt 模块跨请求恢复）"""
import uuid

from langgraph.checkpoint.memory import InMemorySaver

# 进程级共享 checkpointer：同一 thread_id 的 interrupt 暂停状态可跨请求恢复
_shared_checkpointer = InMemorySaver()


def get_shared_checkpointer() -> InMemorySaver:
    """获取共享 InMemorySaver 实例"""
    return _shared_checkpointer


def new_thread_id() -> str:
    """生成新的 thread_id"""
    return uuid.uuid4().hex


def run_graph_with_interrupt(graph, input_state: dict, thread_id: str):
    """运行可能含 interrupt 的图，返回 (completed, state_snapshot)

    - completed=True：图已执行到 END
    - completed=False：图在 interrupt 处暂停，需用 Command(resume=...) 恢复
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(input_state, config=config)
    state = graph.get_state(config)
    # next 非空表示仍有待执行节点（即处于 interrupt 暂停）
    completed = not state.next
    return completed, state.values


def resume_graph(graph, resume_payload: dict, thread_id: str):
    """恢复处于 interrupt 暂停的图"""
    from langgraph.types import Command
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume=resume_payload), config=config)
    state = graph.get_state(config)
    completed = not state.next
    return completed, state.values
