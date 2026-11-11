import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from convai.engine import ConversationEngine
from convai.service import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(ConversationEngine()))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_creates_conversation_and_answers(client):
    response = client.post("/chat", json={"message": "Where is my order ORD-123456?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "order_status"
    assert body["action"] == "execute"
    assert body["slots"]["order_id"] == "ORD-123456"
    assert body["conversation_id"]


def test_chat_continues_conversation(client):
    first = client.post("/chat", json={"message": "I want to return the Widget Pro"})
    conv_id = first.json()["conversation_id"]
    second = client.post("/chat", json={"conversation_id": conv_id,
                                        "message": "My order is ORD-654321"})
    assert second.json()["action"] == "ask_slot"


def test_chat_unknown_conversation_404(client):
    response = client.post("/chat", json={"conversation_id": "conv-9999",
                                          "message": "hello"})
    assert response.status_code == 404


def test_get_conversation_returns_transcript(client):
    created = client.post("/chat", json={"message": "Hello"})
    conv_id = created.json()["conversation_id"]
    response = client.get(f"/conversations/{conv_id}")
    assert response.status_code == 200
    assert "Hello" in response.json()["transcript"]


def test_analytics_endpoint(client):
    client.post("/chat", json={"message": "Where is my order ORD-123456?"})
    response = client.get("/analytics")
    assert response.json()["conversations"] == 1


def test_flows_endpoint_lists_versioned_flows(client):
    response = client.get("/flows")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"
    assert "refund_request" in response.json()["flows"]
