# 模块 6：微信风控（WeChat Risk Control）

> 🎯 **LangGraph 知识点**：四路条件分支、`interrupt()` 上报审批、风险阈值路由

## 业务场景

员工微信消息 → AI 分类（正常/业务/敏感/违规）→ 敏感/违规评估风险分 → 高风险上报主管审批（暂停）→ 拦截/提醒/放行

## 图结构

```
START → receive_message → classify → (四路路由)
  ├── normal → allow → END
  ├── business → log_only → END
  ├── sensitive/violation → risk_assess → (条件路由)
      ├── 高风险(≥80) → escalate (interrupt) → block → log → END
      └── 低风险 → warn → END
```

## LangGraph 核心知识点

### 四路条件分支
`add_conditional_edges("classify", route_after_classify)` 根据消息类型分流到 4 条不同处理路径。

### interrupt 上报审批
高风险消息进入 `escalate` 节点，`interrupt()` 暂停等待主管 `Command(resume={"decision":"approve_block"})` 决策后继续拦截流程。

### 风险阈值路由
`risk_assess` 后按 `risk_score >= 80` 二次路由，决定上报或直接提醒。

## 运行

```bash
python -m src.modules.wechat_risk.graph
```
