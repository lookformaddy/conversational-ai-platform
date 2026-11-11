from convai.analytics import Analytics
from convai.engine import ConversationEngine
from convai.eval import run_eval


def _resolved_engine():
    engine = ConversationEngine()
    conv = engine.start_conversation()
    engine.handle_message(conv, "Where is my order ORD-123456?")
    return engine


def test_analytics_empty_metrics():
    metrics = Analytics().metrics()
    assert metrics["conversations"] == 0
    assert metrics["containment_rate"] == 0.0


def test_analytics_counts_handoff():
    engine = ConversationEngine()
    conv = engine.start_conversation()
    engine.handle_message(conv, "Let me talk to a human")
    metrics = engine.analytics.metrics()
    assert metrics["conversations"] == 1
    assert metrics["handoff_rate"] == 1.0
    assert metrics["containment_rate"] == 0.0


def test_analytics_dedupes_repeat_records():
    engine = _resolved_engine()
    conversation = engine.get_conversation("conv-0001")
    engine.analytics.record(conversation)
    engine.analytics.record(conversation)
    assert engine.analytics.metrics()["conversations"] == 1


def test_analytics_intent_distribution():
    engine = _resolved_engine()
    metrics = engine.analytics.metrics()
    assert metrics["intent_distribution"]["order_status"] == 1
    assert metrics["top_intents"] == ["order_status"]
    assert metrics["avg_turns"] == 1.0


def test_eval_harness_all_scenarios_pass():
    report = run_eval()
    assert report.intent_accuracy == 1.0
    assert report.slot_accuracy == 1.0
    assert report.final_action_accuracy == 1.0
    assert len(report.dialogues) == 8


def test_eval_report_dict_shape():
    report = run_eval().as_dict()
    assert report["dialogues_evaluated"] == 8
    assert report["intent_accuracy"] == 1.0
    assert all(d["final_action_correct"] for d in report["dialogues"])
