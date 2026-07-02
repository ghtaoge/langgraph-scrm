# 模块 3：知识库问答（Knowledge QA）

> 🎯 **LangGraph 知识点**：Corrective RAG、ToolNode、Embedding 向量检索、自检重试循环

## 业务场景

客户提问 → 向量检索知识库 → 评估文档相关性 → 生成回答 → 自检防幻觉 → 不通过则重试/补充搜索

## 图结构

```
START → retrieve → grade_docs → (条件路由)
  ├── docs_irrelevant → web_search → retrieve (补充检索)
  └── docs_relevant → generate → verify → (条件路由)
      ├── failed → generate (重试，上限 3 次)
      └── passed → respond → END
```

## LangGraph 核心知识点

### Corrective RAG
检索结果不理想时，自动转向网页搜索补充，再回到检索-评估流程，形成自我纠错闭环。

### Annotated reducer 累积文档
`documents` 与 `web_results` 使用 `Annotated[list, operator.add]`，多轮检索结果自动累积。

### 自检重试循环
`verify` 节点检查回答是否基于文档（防幻觉），不通过则回到 `generate` 重试，受 `retries` 上限保护。

## 运行

```bash
python -m src.modules.knowledge_qa.graph
```
