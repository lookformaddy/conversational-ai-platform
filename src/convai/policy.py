"""Policy-driven next-action selection.

The policy is a pure function of (state, NLU result, extracted slots,
safety result): no LLM, no hidden state, fully testable. Decision order
matters and is deliberate — safety first, then handoff, then understanding
(clarify / disambiguate), then slot completeness, then confirmation, then
execution. Every branch is covered by tests.
"""

from __future__ import annotations

from convai.flows import get_flow
from convai.models import Action, ActionType, DialogueState, IntentResult
from convai.safety import SafetyResult

# Below this confidence the classifier's top intent is not trusted.
# Matches the classifier's UNKNOWN_CONFIDENCE_THRESHOLD.
LOW_CONFIDENCE = 0.20
# Below this margin the top two intents are too close to pick between.
AMBIGUITY_MARGIN = 0.08
# After this many failed clarifications, hand off instead of looping.
MAX_CLARIFICATIONS = 3

# Intents that are answered directly once understood (no execution step).
RESPOND_INTENTS = {"account_help", "product_info", "billing_question", "shipping_info", "complaint"}
# Intents whose fulfillment is a (simulated) backend action.
EXECUTE_INTENTS = {"order_status", "return_request", "refund_request", "cancel_subscription"}


def decide(
    state: DialogueState,
    intent_result: IntentResult,
    safety: SafetyResult,
    *,
    frustrated: bool = False,
    flow_version: str = "1.0.0",
) -> Action:
    if safety.blocked:
        return Action(
            type=ActionType.REFUSE,
            intent=intent_result.intent,
            summary=f"Blocked by safety screen: {safety.category}",
        )

    if intent_result.intent == "human_handoff" or frustrated:
        return Action(
            type=ActionType.HANDOFF,
            intent=intent_result.intent,
            summary="Handoff requested by the customer.",
        )

    if intent_result.intent == "unknown" or intent_result.confidence < LOW_CONFIDENCE:
        if state.clarification_count >= MAX_CLARIFICATIONS:
            return Action(
                type=ActionType.HANDOFF,
                intent="unknown",
                summary="Too many failed clarifications; handing off to a human.",
            )
        return Action(
            type=ActionType.CLARIFY_INTENT,
            intent="unknown",
            summary="Could not confidently identify the intent.",
        )

    runner_up = intent_result.runner_up
    if (
        runner_up is not None
        and intent_result.margin < AMBIGUITY_MARGIN
        and runner_up[1] >= 0.15
    ):
        return Action(
            type=ActionType.DISAMBIGUATE,
            intent=intent_result.intent,
            candidates=[intent_result.intent, runner_up[0]],
            summary="Top two intents are too close to decide.",
        )

    intent = intent_result.intent
    if intent == "greeting":
        return Action(type=ActionType.GREET, intent=intent, summary="Greet the customer.")
    if intent == "goodbye":
        return Action(type=ActionType.CLOSE, intent=intent, summary="Close the conversation.")

    flow = get_flow(intent, version=flow_version)
    if flow is not None:
        missing = [s for s in flow.required_slots if s not in state.slots]
        if missing:
            slot = missing[0]
            return Action(
                type=ActionType.ASK_SLOT,
                intent=intent,
                slot=slot,
                summary=flow.slot_prompts.get(slot, f"Please provide {slot}."),
            )
        if flow.confirm_before_execute:
            return Action(
                type=ActionType.CONFIRM,
                intent=intent,
                summary=f"Confirmation required before executing {intent}.",
            )

    if intent in EXECUTE_INTENTS:
        return Action(type=ActionType.EXECUTE, intent=intent, summary=f"Execute {intent}.")
    return Action(type=ActionType.RESPOND, intent=intent, summary=f"Answer {intent}.")
