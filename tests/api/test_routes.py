"""API 路由测试 — 使用 httpx ASGITransport"""
import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from langchain_core.messages import AIMessage

from src.api.main import app


@pytest.fixture
async def client():
    """异步 HTTP 测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_root_endpoint(client):
    """根路径应返回项目信息"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LangGraph-SCRM"
    assert len(data["modules"]) == 6


async def test_intent_router_validation_error(client):
    """空消息应返回 422 校验错误（不触发 LLM）"""
    response = await client.post("/intent-router/", json={"message": ""})
    assert response.status_code == 422


@patch("src.modules.intent_router.nodes.get_llm")
async def test_intent_router_endpoint_with_mock(mock_get_llm, client):
    """意图路由端点端到端 — mock LLM"""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content=json.dumps({"intent": "consult", "confidence": 0.9})),
        AIMessage(content="我们的产品价格如下..."),
    ]
    mock_get_llm.return_value = mock_llm

    response = await client.post("/intent-router/", json={"message": "我想咨询产品价格"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "consult"
    assert data["skill_group"] == "产品咨询组"
