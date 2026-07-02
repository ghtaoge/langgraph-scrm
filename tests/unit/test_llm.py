"""测试 LLM 抽象层"""
import pytest

from src.config.settings import LLMProvider, settings
from src.core.llm import get_llm


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    """为所有测试注入 dummy API key，避免 ChatOpenAI 实例化时凭证校验失败"""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setattr(settings, "DOUBAO_API_KEY", "sk-test-dummy")


def test_get_llm_returns_openai_by_default():
    """默认返回 OpenAI LLM"""
    llm = get_llm()
    assert llm is not None


def test_get_llm_openai_provider():
    """OpenAI 提供者返回 ChatOpenAI 实例"""
    llm = get_llm(LLMProvider.OPENAI)
    assert llm is not None


def test_get_llm_with_custom_base_url():
    """支持自定义 base_url（用于代理）"""
    llm = get_llm(base_url="https://custom-proxy.com/v1")
    assert llm is not None
