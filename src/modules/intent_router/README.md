# 模块 1：意图路由（Intent Router）

> 🎯 **LangGraph 知识点**：`StateGraph`、`add_node`、`add_edge`、`add_conditional_edges`

## 业务场景

客户发消息 → AI 分类意图 → 自动分配到对应技能组

## 图结构

```
START → classify → (条件路由)
  ├── consult → respond_consult → END
  ├── complaint → escalate_complaint → END
  ├── after_sale → handle_after_sale → END
  └── other → handle_other → END
```

## 知识点详解

### StateGraph

LangGraph 的核心概念。用 `StateGraph(StateSchema)` 创建一个状态图，图的每个节点共享同一个 State（TypedDict）。

### add_node

添加节点：`graph.add_node("name", func)`。节点函数接收 state dict，返回 state 更新 dict。

### add_conditional_edges

条件路由：`graph.add_conditional_edges("from_node", condition_fn)`。条件函数根据 state 返回下一个节点名。

## 运行

```bash
python -m src.modules.intent_router.graph
```
