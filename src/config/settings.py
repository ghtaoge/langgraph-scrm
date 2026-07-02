"""全局配置加载 — 从 .env 读取所有环境变量，提供 Settings 单例"""
import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """LLM 提供者枚举"""
    OPENAI = "openai"
    DOUBAO = "doubao"


class Settings:
    """全局配置单例 — 所有模块共享"""

    # LLM 配置
    LLM_PROVIDER: LLMProvider = LLMProvider(os.getenv("LANGGRAPH_LLM_PROVIDER", "openai"))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    DOUBAO_API_KEY: str = os.getenv("DOUBAO_API_KEY", "")
    DOUBAO_ENDPOINT: str = os.getenv("DOUBAO_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3")
    DOUBAO_MODEL: str = os.getenv("DOUBAO_MODEL", "doubao-pro-32k")

    # 向量存储配置
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    # Checkpoint 配置
    CHECKPOINT_STORE: str = os.getenv("CHECKPOINT_STORE", "sqlite")
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/scrm.db")

    # LLM 重试配置
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

    # 业务配置
    LEAD_QUALIFIER_MIN_SCORE: float = float(os.getenv("LEAD_QUALIFIER_MIN_SCORE", "60.0"))
    LEAD_QUALIFIER_MAX_QUESTIONS: int = int(os.getenv("LEAD_QUALIFIER_MAX_QUESTIONS", "5"))
    AFTER_SALE_APPROVAL_TIMEOUT_HOURS: int = int(os.getenv("AFTER_SALE_APPROVAL_TIMEOUT_HOURS", "24"))


# 全局单例
settings = Settings()
