"""Conversation engine: safety -> NLU -> state -> policy -> response.

The engine owns conversations end to end and is the only component that
mutates state. Every turn is deterministic given the same inputs (the
classifier is seeded by its fixed training data; the simulated backend
derives order status from the order number, not from randomness).
"""

from __future__ import annotations

import itertools
from typing import Any

from convai.analytics import Analytics
from convai.data import PRODUCT_CATALOG
from convai.flows import FLOW_VERSION, get_flow
from convai.models import ActionType, Conversation, Handoff, IntentResult, Turn
from convai.nlu import classify
from convai.policy import decide
from convai.safety import SAFE_REFUSAL, redact, screen
from convai.slots import extract_slots, is_affirmative, is_negative
from convai.state import StateTracker

FRUSTRATION_WORDS = ["frustrated", "angry", "furious", "livid", "fed up"]

_ORDER_STATUSES = ["shipped — in transit", "out for delivery", "delivered"]


def _simulated_order_status(order_id: str) -> str:
    digits = "".join(ch for ch in order_id if ch.isdigit())
    index = (sum(int(d) for d in digits) if digits else 0) % len(_ORDER_STATUSES)
    return _ORDER_STATUSES[index]


class ConversationEngine:
    def __init__(self, flow_version: str = FLOW_VERSION) -> None:
        self.flow_version = flow_version
        self.analytics = Analytics()
        self._conversations: dict[str, Conversation] = {}
        self._trackers: dict[str, StateTracker] = {}
        self._counter = itertools.count(1)

    # -- conversation lifecycle -------------------------------------------------
    def start_conversation(self) -> str:
        conversation_id = f"conv-{next(self._counter):04d}"
        tracker = StateTracker(conversation_id)
        self._trackers[conversation_id] = tracker
        self._conversations[conversation_id] = Conversation(
            conversation_id=conversation_id, state=tracker.state
        )
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    # -- main entry point -------------------------------------------------------
    def handle_message(self, conversation_id: str, text: str) -> Turn:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        tracker = self._trackers[conversation_id]
        state = tracker.state

        safety = screen(text)
        if safety.blocked:
            return self._record(conversation, text, "unknown", 0.0, {}, ActionType.REFUSE, SAFE_REFUSAL)

        # A pending confirmation is answered before any new NLU work: the
        # user's yes/no is about the pending action, not a new request.
        if state.pending_confirmation is not None:
            if is_affirmative(text):
                pending = state.pending_confirmation
                intent = pending["action"]
                tracker.clear_pending_confirmation()
                response = self._execution_response(intent, state.slots)
                tracker.mark_resolved(f"{intent} executed after confirmation")
                turn = self._record(conversation, text, intent, 1.0, {}, ActionType.EXECUTE, response)
                self.analytics.record(conversation)
                return turn
            if is_negative(text):
                intent = state.pending_confirmation["action"]
                tracker.clear_pending_confirmation()
                response = "No problem — I won't go ahead with that. Is there anything else I can help with?"
                return self._record(conversation, text, intent, 1.0, {}, ActionType.RESPOND, response)
            tracker.clear_pending_confirmation()  # New topic supersedes the pending action.

        intent_result = classify(text)
        slots = extract_slots(text)
        frustrated = any(word in text.lower() for word in FRUSTRATION_WORDS)

        # Active-flow continuation: if the previous turn left a flow waiting
        # for slots and this turn fills one of them, the user is answering
        # that flow's question — not starting a new request. (Greeting,
        # goodbye, and handoff always take precedence over continuation.)
        if (
            state.active_flow is not None
            and intent_result.intent not in {"greeting", "goodbye", "human_handoff"}
        ):
            flow = get_flow(state.active_flow, version=self.flow_version)
            if flow is not None:
                missing = [s for s in flow.required_slots if s not in state.slots]
                if missing and any(slot in slots for slot in missing):
                    # A slot-bearing answer to a pending question is strong
                    # evidence of continuation, so it also lifts confidence
                    # past the low-confidence clarify threshold.
                    intent_result = IntentResult(
                        intent=state.active_flow,
                        confidence=0.9,
                        scores={state.active_flow: 0.9, "unknown": 0.1},
                    )

        tracker.apply_turn(intent_result.intent, slots)
        action = decide(state, intent_result, safety, frustrated=frustrated,
                        flow_version=self.flow_version)

        if action.type is ActionType.CLARIFY_INTENT:
            tracker.mark_clarification()
        if action.type is ActionType.ASK_SLOT:
            state.active_flow = action.intent
        elif action.type in (ActionType.EXECUTE, ActionType.CONFIRM, ActionType.RESPOND,
                             ActionType.GREET, ActionType.CLOSE, ActionType.HANDOFF):
            state.active_flow = None
        if action.type is ActionType.CONFIRM:
            tracker.set_pending_confirmation(action.intent, {"slots": dict(state.slots)})

        response = self._respond(action, state, intent_result)
        if action.type is ActionType.HANDOFF:
            tracker.request_handoff()
            conversation.handoff_transcript = conversation.transcript + f"\nAssistant: {response}"
        turn = self._record(conversation, text, intent_result.intent, intent_result.confidence,
                            slots, action.type, response)
        if action.type in (ActionType.EXECUTE, ActionType.CLOSE, ActionType.HANDOFF):
            if action.type is ActionType.EXECUTE:
                tracker.mark_resolved(f"{action.intent} executed")
            elif action.type is ActionType.CLOSE:
                tracker.mark_resolved("customer said goodbye")
            self.analytics.record(conversation)
        return turn

    # -- responses --------------------------------------------------------------
    def _execution_response(self, intent: str, slots: dict[str, Any]) -> str:
        if intent == "order_status":
            order_id = slots.get("order_id", "your order")
            return (f"Order {order_id} is {_simulated_order_status(order_id)}. "
                    "You can follow its progress with the tracking link we emailed you.")
        if intent == "return_request":
            return (f"Your return for the {slots.get('product', 'item')} "
                    f"(order {slots.get('order_id', 'unknown')}) is started. "
                    "A prepaid label is on its way to your email.")
        if intent == "refund_request":
            return (f"Done — a refund for order {slots.get('order_id', 'unknown')} "
                    "has been issued to your original payment method. "
                    "It usually arrives within 3–5 business days.")
        if intent == "cancel_subscription":
            return "Done — your subscription is cancelled and will not renew. You keep access until the end of the current billing period."
        return "Done."

    def _respond(self, action, state, intent_result) -> str:
        intent = action.intent
        if action.type is ActionType.REFUSE:
            return SAFE_REFUSAL
        if action.type is ActionType.GREET:
            return ("Hello! I'm the support assistant. I can check an order, start a return "
                    "or refund, answer product and shipping questions, or connect you with "
                    "a human agent. How can I help?")
        if action.type is ActionType.CLOSE:
            return "You're welcome! Have a great day — goodbye."
        if action.type is ActionType.CLARIFY_INTENT:
            return ("I'm not sure I understood that. Could you rephrase it? For example, "
                    "you can ask about an order, a return, a refund, or a product.")
        if action.type is ActionType.DISAMBIGUATE:
            options = " or ".join(action.candidates)
            return f"Just to check — did you mean {options}? Could you say a little more?"
        if action.type is ActionType.ASK_SLOT:
            flow = get_flow(intent, version=self.flow_version)
            if flow is not None:
                return flow.slot_prompts.get(action.slot or "", f"Please provide {action.slot}.")
            return f"Please provide {action.slot}."
        if action.type is ActionType.CONFIRM:
            if intent == "refund_request":
                return (f"Just to confirm: refund order {state.slots.get('order_id')} "
                        "to the original payment method? Please reply yes to confirm, or no to cancel.")
            if intent == "cancel_subscription":
                return ("Just to confirm: cancel your subscription so it does not renew? "
                        "Please reply yes to confirm, or no to cancel.")
            return "Please confirm you would like me to go ahead. Reply yes or no."
        if action.type is ActionType.EXECUTE:
            return self._execution_response(intent, state.slots)
        if action.type is ActionType.HANDOFF:
            return ("Of course — I'm connecting you with a human agent now. "
                    "I've shared our conversation so far, so you won't need to repeat yourself.")
        if intent == "product_info":
            product = state.slots.get("product")
            catalog = ", ".join(PRODUCT_CATALOG)
            if product:
                return (f"The {product} is one of our most popular products, with a 2-year "
                        "warranty and free returns within 30 days. Would you like pricing or stock details?")
            return f"We carry {catalog}. Which product would you like to know about?"
        if intent == "shipping_info":
            return ("Standard delivery takes 3–5 business days and is free over $50; "
                    "express delivery takes 1–2 business days. We ship across the United States and Canada.")
        if intent == "billing_question":
            return ("I can look into a charge for you. Charges appear under the name of the "
                    "store on your statement; if a charge looks unfamiliar, share the amount "
                    "and date and I'll check it — or I can connect you with a human agent.")
        if intent == "account_help":
            return ("For account access, use the 'Forgot password' link on the sign-in page "
                    "to reset your password. If your account is locked, I can connect you "
                    "with a human agent to verify your identity.")
        if intent == "complaint":
            return ("I'm sorry to hear that — that isn't the experience we want you to have. "
                    "I've noted your feedback. Would you like me to connect you with a human "
                    "agent to make this right?")
        return "Thanks — how else can I help?"

    def _record(self, conversation, text, intent, confidence, slots, action_type, response) -> Turn:
        turn = Turn(
            turn_index=len(conversation.turns) + 1,
            user_text=text,
            user_text_redacted=redact(text),
            intent=intent,
            confidence=confidence,
            slots=dict(slots),
            action=action_type,
            response=response,
        )
        conversation.turns.append(turn)
        return turn

    # -- handoff ----------------------------------------------------------------
    def build_handoff(self, conversation_id: str, reason: str = "customer request") -> Handoff:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        state = conversation.state
        return Handoff(
            conversation_id=conversation_id,
            reason=reason,
            summary=(f"{len(conversation.turns)} turns; intents: "
                     f"{', '.join(state.intent_history) or 'none'}; "
                     f"slots: {state.slots}"),
            slots=dict(state.slots),
            intent_history=list(state.intent_history),
            transcript=conversation.transcript,
            priority="high" if state.clarification_count >= 3 else "normal",
        )
