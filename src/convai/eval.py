"""Eval harness over the synthetic dialogues in :mod:`convai.data`.

Runs each scenario through a fresh engine and reports:
- ``intent_accuracy`` — over turns whose expected intent is a real intent
  (``"unknown"`` expectations, e.g. bare confirmations, are excluded because
  the confirmation path answers from pending state, not from NLU).
- ``slot_accuracy`` — over turns with expected slots, checked against the
  *cumulative* carried-over state (so carryover is tested, not just extraction).
- ``final_action_accuracy`` — the policy's last action per dialogue.
- Per-dialogue detail, so a regression names the failing scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from convai.data import EVAL_DIALOGUES
from convai.engine import ConversationEngine


@dataclass
class DialogueEvalResult:
    dialogue_id: str
    intent_correct: int
    intent_total: int
    slot_correct: int
    slot_total: int
    final_action: str
    expected_final_action: str
    final_action_correct: bool
    turns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalReport:
    dialogues: list[DialogueEvalResult]
    intent_accuracy: float
    slot_accuracy: float
    final_action_accuracy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "dialogues_evaluated": len(self.dialogues),
            "intent_accuracy": round(self.intent_accuracy, 3),
            "slot_accuracy": round(self.slot_accuracy, 3),
            "final_action_accuracy": round(self.final_action_accuracy, 3),
            "dialogues": [
                {
                    "id": d.dialogue_id,
                    "intent": f"{d.intent_correct}/{d.intent_total}",
                    "slots": f"{d.slot_correct}/{d.slot_total}",
                    "final_action": d.final_action,
                    "expected_final_action": d.expected_final_action,
                    "final_action_correct": d.final_action_correct,
                }
                for d in self.dialogues
            ],
        }


def evaluate_dialogue(engine: ConversationEngine, scenario: dict[str, Any]) -> DialogueEvalResult:
    conversation_id = engine.start_conversation()
    intent_correct = intent_total = slot_correct = slot_total = 0
    turn_details: list[dict[str, Any]] = []
    last_action = ""
    for step in scenario["turns"]:
        turn = engine.handle_message(conversation_id, step["user"])
        last_action = turn.action.value
        expected_intent = step.get("expected_intent", "unknown")
        if expected_intent != "unknown":
            intent_total += 1
            intent_correct += turn.intent == expected_intent
        expected_slots = step.get("expected_slots", {})
        state = engine.get_conversation(conversation_id).state
        for slot, value in expected_slots.items():
            slot_total += 1
            slot_correct += state.slots.get(slot) == value
        turn_details.append({
            "user": step["user"], "intent": turn.intent,
            "expected_intent": expected_intent, "action": turn.action.value,
            "response": turn.response,
        })
    expected_final = scenario["expected_final_action"]
    return DialogueEvalResult(
        dialogue_id=scenario["id"],
        intent_correct=intent_correct, intent_total=intent_total,
        slot_correct=slot_correct, slot_total=slot_total,
        final_action=last_action, expected_final_action=expected_final,
        final_action_correct=last_action == expected_final,
        turns=turn_details,
    )


def run_eval(scenarios: list[dict[str, Any]] | None = None) -> EvalReport:
    scenarios = scenarios if scenarios is not None else EVAL_DIALOGUES
    results = [evaluate_dialogue(ConversationEngine(), s) for s in scenarios]
    intent_total = sum(r.intent_total for r in results)
    intent_correct = sum(r.intent_correct for r in results)
    slot_total = sum(r.slot_total for r in results)
    slot_correct = sum(r.slot_correct for r in results)
    return EvalReport(
        dialogues=results,
        intent_accuracy=(intent_correct / intent_total) if intent_total else 0.0,
        slot_accuracy=(slot_correct / slot_total) if slot_total else 0.0,
        final_action_accuracy=(
            sum(r.final_action_correct for r in results) / len(results) if results else 0.0
        ),
    )
