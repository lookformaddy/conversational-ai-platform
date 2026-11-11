from convai.models import DialogueState, IntentResult
from convai.policy import decide
from convai.safety import SafetyResult
from convai.state import StateTracker


def _state(**kwargs):
    return DialogueState(conversation_id="conv-test", **kwargs)


def _intent(intent, confidence=0.9, scores=None):
    scores = scores or {intent: confidence, "other": 1.0 - confidence}
    return IntentResult(intent=intent, confidence=confidence, scores=scores)


def test_tracker_carries_slots_over():
    tracker = StateTracker("c1")
    tracker.apply_turn("order_status", {"order_id": "ORD-123456"})
    tracker.apply_turn("shipping_info", {})
    assert tracker.state.slots["order_id"] == "ORD-123456"
    assert tracker.state.turn_count == 2
    assert tracker.state.intent_history == ["order_status", "shipping_info"]


def test_tracker_missing_slots():
    tracker = StateTracker("c1")
    tracker.apply_turn("return_request", {"product": "Widget Pro"})
    assert tracker.missing_slots(["product", "order_id", "reason"]) == ["order_id", "reason"]


def test_tracker_confirmation_lifecycle():
    tracker = StateTracker("c1")
    tracker.set_pending_confirmation("refund_request", {"order_id": "ORD-1"})
    assert tracker.state.pending_action == "refund_request"
    tracker.clear_pending_confirmation()
    assert tracker.state.pending_action is None
    assert tracker.state.pending_confirmation is None


def test_policy_refuses_blocked_message():
    safety = SafetyResult(True, "prompt_injection", "blocked")
    action = decide(_state(), _intent("order_status"), safety)
    assert action.type.value == "refuse"


def test_policy_handoff_on_request():
    action = decide(_state(), _intent("human_handoff"), SafetyResult(False))
    assert action.type.value == "handoff"


def test_policy_handoff_on_frustration():
    action = decide(_state(), _intent("complaint"), SafetyResult(False), frustrated=True)
    assert action.type.value == "handoff"


def test_policy_clarifies_unknown():
    action = decide(_state(), _intent("unknown", 1.0, {"unknown": 1.0}), SafetyResult(False))
    assert action.type.value == "clarify_intent"


def test_policy_hands_off_after_repeated_clarifications():
    state = _state(clarification_count=3)
    action = decide(state, _intent("unknown", 1.0, {"unknown": 1.0}), SafetyResult(False))
    assert action.type.value == "handoff"


def test_policy_disambiguates_close_intents():
    result = IntentResult(intent="refund_request", confidence=0.42,
                          scores={"refund_request": 0.42, "return_request": 0.38,
                                  "order_status": 0.20})
    action = decide(_state(), result, SafetyResult(False))
    assert action.type.value == "disambiguate"
    assert set(action.candidates) == {"refund_request", "return_request"}


def test_policy_asks_for_missing_slot():
    state = _state(slots={"product": "Widget Pro"})
    action = decide(state, _intent("return_request"), SafetyResult(False))
    assert action.type.value == "ask_slot"
    assert action.slot == "order_id"


def test_policy_confirms_before_refund():
    state = _state(slots={"order_id": "ORD-123456"})
    action = decide(state, _intent("refund_request"), SafetyResult(False))
    assert action.type.value == "confirm"


def test_policy_executes_complete_order_status():
    state = _state(slots={"order_id": "ORD-123456"})
    action = decide(state, _intent("order_status"), SafetyResult(False))
    assert action.type.value == "execute"


def test_policy_greets_and_closes():
    assert decide(_state(), _intent("greeting"), SafetyResult(False)).type.value == "greet"
    assert decide(_state(), _intent("goodbye"), SafetyResult(False)).type.value == "close"
