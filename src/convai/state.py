"""Dialogue state tracking with context carryover.

The tracker is the conversation's memory: slots filled in earlier turns stay
filled (carryover) unless explicitly replaced, intent history is kept for
analytics and disambiguation, and a pending confirmation survives until the
user answers it. State is a plain dataclass, so it serialises trivially.
"""

from __future__ import annotations

from typing import Any

from convai.models import DialogueState


class StateTracker:
    """Owns one conversation's :class:`DialogueState`."""

    def __init__(self, conversation_id: str) -> None:
        self.state = DialogueState(conversation_id=conversation_id)

    def apply_turn(self, intent: str, slots: dict[str, Any]) -> DialogueState:
        """Merge a turn's NLU output into state (slots carry over)."""
        self.state.turn_count += 1
        self.state.intent_history.append(intent)
        # New values replace old ones; absent slots keep their carried value.
        self.state.slots.update(slots)
        self.state.touch()
        return self.state

    def set_pending_confirmation(self, action: str, payload: dict[str, Any]) -> None:
        self.state.pending_action = action
        self.state.pending_confirmation = {"action": action, **payload}

    def clear_pending_confirmation(self) -> None:
        self.state.pending_action = None
        self.state.pending_confirmation = None

    def mark_clarification(self) -> None:
        self.state.clarification_count += 1

    def mark_resolved(self, resolution: str) -> None:
        self.state.resolved = True
        self.state.resolution = resolution
        self.state.touch()

    def request_handoff(self) -> None:
        self.state.handoff_requested = True
        self.state.touch()

    def missing_slots(self, required: list[str]) -> list[str]:
        return [slot for slot in required if slot not in self.state.slots]

    def snapshot(self) -> dict[str, Any]:
        return {
            "conversation_id": self.state.conversation_id,
            "slots": dict(self.state.slots),
            "intent_history": list(self.state.intent_history),
            "pending_action": self.state.pending_action,
            "clarification_count": self.state.clarification_count,
            "turn_count": self.state.turn_count,
            "handoff_requested": self.state.handoff_requested,
            "resolved": self.state.resolved,
            "resolution": self.state.resolution,
        }
