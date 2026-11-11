"""Conversation analytics: containment, handoff, and intent mix.

Analytics consume finished conversations only (resolved, closed, or handed
off) and never store raw user text — summaries carry intents, slot names,
and counts. Recording the same conversation twice updates its entry rather
than double-counting it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationSummary:
    conversation_id: str
    turns: int
    intents: list[str]
    slot_names: list[str]
    clarifications: int
    handed_off: bool
    resolved: bool
    resolution: str | None = None


@dataclass
class Analytics:
    summaries: list[ConversationSummary] = field(default_factory=list)

    def record(self, conversation: Any) -> ConversationSummary:
        state = conversation.state
        summary = ConversationSummary(
            conversation_id=conversation.conversation_id,
            turns=len(conversation.turns),
            intents=list(state.intent_history),
            slot_names=sorted(state.slots),
            clarifications=state.clarification_count,
            handed_off=state.handoff_requested,
            resolved=state.resolved,
            resolution=state.resolution,
        )
        for index, existing in enumerate(self.summaries):
            if existing.conversation_id == summary.conversation_id:
                self.summaries[index] = summary
                return summary
        self.summaries.append(summary)
        return summary

    # -- aggregate metrics ------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        total = len(self.summaries)
        if total == 0:
            return {
                "conversations": 0, "avg_turns": 0.0, "containment_rate": 0.0,
                "handoff_rate": 0.0, "resolution_rate": 0.0,
                "avg_clarifications": 0.0, "intent_distribution": {},
                "top_intents": [],
            }
        contained = sum(1 for s in self.summaries if s.resolved and not s.handed_off)
        handed_off = sum(1 for s in self.summaries if s.handed_off)
        resolved = sum(1 for s in self.summaries if s.resolved)
        intent_counts = Counter(intent for s in self.summaries for intent in s.intents)
        return {
            "conversations": total,
            "avg_turns": round(sum(s.turns for s in self.summaries) / total, 2),
            "containment_rate": round(contained / total, 3),
            "handoff_rate": round(handed_off / total, 3),
            "resolution_rate": round(resolved / total, 3),
            "avg_clarifications": round(sum(s.clarifications for s in self.summaries) / total, 2),
            "intent_distribution": dict(intent_counts),
            "top_intents": [intent for intent, _ in intent_counts.most_common(5)],
        }
