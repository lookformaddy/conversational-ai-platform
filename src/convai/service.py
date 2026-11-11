"""FastAPI service layer (optional dependency).

The core engine is pure stdlib; this module imports FastAPI only when
``create_app()`` is called, and pydantic only behind a guarded import, so
``import convai.service`` never fails on a machine without them installed.
Tests skip the API suite when FastAPI is absent (see ``tests/test_api.py``).
"""

from __future__ import annotations

from typing import Any

from convai.engine import ConversationEngine
from convai.flows import FLOW_VERSION

try:
    # Request/response models live at module level so FastAPI can resolve
    # the (stringified, via `from __future__ import annotations`) endpoint
    # annotations. Importing pydantic here is safe: the core never imports
    # this module, and create_app() still raises ImportError without FastAPI.
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        conversation_id: str | None = None
        message: str

    class ChatResponse(BaseModel):
        conversation_id: str
        intent: str
        confidence: float
        action: str
        response: str
        slots: dict[str, Any]

except ImportError:  # pragma: no cover - exercised only without pydantic
    ChatRequest = None  # type: ignore[assignment]
    ChatResponse = None  # type: ignore[assignment]


def create_app(engine: ConversationEngine | None = None):
    """Build the FastAPI app. Raises ImportError if FastAPI is not installed."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Conversational AI Platform", version=FLOW_VERSION)
    eng = engine or ConversationEngine()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "flow_version": FLOW_VERSION}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        conversation_id = payload.conversation_id or eng.start_conversation()
        if eng.get_conversation(conversation_id) is None:
            raise HTTPException(status_code=404, detail="Unknown conversation")
        turn = eng.handle_message(conversation_id, payload.message)
        state = eng.get_conversation(conversation_id).state
        return ChatResponse(
            conversation_id=conversation_id,
            intent=turn.intent,
            confidence=turn.confidence,
            action=turn.action.value,
            response=turn.response,
            slots=dict(state.slots),
        )

    @app.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = eng.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Unknown conversation")
        return {
            "conversation_id": conversation_id,
            "state": conversation.state.__dict__,
            "turns": [
                {"user": t.user_text_redacted, "assistant": t.response,
                 "intent": t.intent, "action": t.action.value}
                for t in conversation.turns
            ],
            "transcript": conversation.transcript,
        }

    @app.get("/analytics")
    def analytics() -> dict[str, Any]:
        return eng.analytics.metrics()

    @app.get("/flows")
    def flows() -> dict[str, Any]:
        from convai.flows import FLOW_REGISTRY

        return {
            "version": FLOW_VERSION,
            "flows": {
                name: {"required_slots": flow.required_slots,
                       "confirm_before_execute": flow.confirm_before_execute}
                for name, flow in FLOW_REGISTRY[FLOW_VERSION].items()
            },
        }

    return app


def main() -> None:
    """Run the service with uvicorn: ``python -m convai.service``."""
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
