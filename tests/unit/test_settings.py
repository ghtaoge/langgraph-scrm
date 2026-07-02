"""测试 Settings 配置加载"""

from src.config.settings import LLMProvider, Settings


def test_default_provider_is_openai():
    """默认 LLM 提供者应为 OpenAI"""
    s = Settings()
    assert s.LLM_PROVIDER == LLMProvider.OPENAI


def test_provider_enum_values():
    """LLMProvider 枚举值正确"""
    assert LLMProvider.OPENAI.value == "openai"
    assert LLMProvider.DOUBAO.value == "doubao"


def test_default_checkpoint_is_sqlite():
    """默认 Checkpoint 存储应为 SQLite"""
    s = Settings()
    assert s.CHECKPOINT_STORE == "sqlite"


def test_default_vector_store_is_chroma():
    """默认向量存储应为 Chroma"""
    s = Settings()
    assert s.VECTOR_STORE == "chroma"
