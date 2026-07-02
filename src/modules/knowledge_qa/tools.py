"""知识库问答模块工具 — 向量检索工具"""

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from src.config.settings import settings


def get_retriever_tool(collection_name: str = "scrm_knowledge"):
    """创建向量检索工具

    Args:
        collection_name: Chroma 集合名

    Returns:
        LangChain tool 对象
    """
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @tool
    def retriever_tool(query: str) -> str:
        """知识库检索 — 搜索 SCRM 产品文档和 FAQ

        Args:
            query: 搜索关键词

        Returns:
            相关文档片段
        """
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever_tool
