"""Core data models for the conversational AI platform.

All models are plain dataclasses (no pydantic dependency in the core) so the
dialogue engine runs on the standard library alone. The FastAPI layer
translates these into response models at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    RESPOND = "respond"
    ASK_SLOT = "ask_slot"
    CLARIFY_INTENT = "clarify_intent"
    DISAMBIGUATE = "disambiguate"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    HANDOFF = "handoff"
    REFUSE = "refuse"
    GREET = "greet"
    CLOSE = "close"


@dataclass
class IntentResult:
    """Result of intent classification for one utterance."""

    intent: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def runner_up(self) -> tuple[str, float] | None:
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[1] if len(ranked) > 1 else None

    @property
    def margin(self) -> float:
        """Probability gap between the top intent and its runner-up."""
        ranked = sorted(self.scores.values(), reverse=True)
        if len(ranked) < 2:
            return 1.0
        return ranked[0] - ranked[1]


@dataclass
class Action:
    """The next action the dialogue policy selected."""

    type: ActionType
    intent: str = "unknown"
    slot: str | None = None
    candidates: list[str] = field(default_factory=list)
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """One user turn and the platform's response to it."""

    turn_index: int
    user_text: str
    user_text_redacted: str
    intent: str
    confidence: float
    slots: dict[str, Any]
    action: ActionType
    response: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DialogueState:
    """Mutable per-conversation state tracked across turns."""

    conversation_id: str
    slots: dict[str, Any] = field(default_factory=dict)
    intent_history: list[str] = field(default_factory=list)
    pending_action: str | None = None
    pending_confirmation: dict[str, Any] | None = None
    active_flow: str | None = None
    clarification_count: int = 0
    turn_count: int = 0
    handoff_requested: bool = False
    resolved: bool = False
    resolution: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    """A full conversation: state plus its turn transcript."""

    conversation_id: str
    state: DialogueState
    turns: list[Turn] = field(default_factory=list)
    handoff_transcript: str | None = None

    @property
    def transcript(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"Customer: {turn.user_text_redacted}")
            lines.append(f"Assistant: {turn.response}")
        return "\n".join(lines)


@dataclass
class Handoff:
    """Payload handed to a human agent: transcript + structured context."""

    conversation_id: str
    reason: str
    summary: str
    slots: dict[str, Any]
    intent_history: list[str]
    transcript: str
    priority: str = "normal"
