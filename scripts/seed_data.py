"""数据初始化脚本 — 向量索引 + SQLite 数据库"""
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config.settings import settings
from src.data.seed_db import init_db


def seed_vector_store():
    """初始化 Chroma 向量存储 — 加载示例文档"""
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

    # 示例 FAQ 文档
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "src", "data", "sample_docs")
    documents = []

    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".md"):
                path = os.path.join(docs_dir, filename)
                content = open(path, encoding="utf-8").read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "type": "faq"},
                ))

    if not documents:
        # 如果没有文件，使用内置示例
        documents = [
            Document(page_content="SCRM 系统支持微信客户管理、朋友圈营销、社群运营等功能。", metadata={"source": "builtin", "type": "faq"}),
            Document(page_content="退换货政策：7天无理由退货，15天质量问题换货，需保留原始包装。", metadata={"source": "builtin", "type": "policy"}),
            Document(page_content="订单追踪功能支持物流信息实时查询，异常订单自动告警。", metadata={"source": "builtin", "type": "faq"}),
        ]

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="scrm_knowledge",
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    print(f"[OK] 向量存储初始化完成 — {len(documents)} 份文档")


def main():
    """运行所有数据初始化"""
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    seed_vector_store()
    init_db()


if __name__ == "__main__":
    main()
