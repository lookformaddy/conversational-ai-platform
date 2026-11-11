from convai.nlu import IntentClassifier, classify, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Where Is My ORDER?!") == ["where", "is", "my", "order"]


def test_greeting_classified():
    assert classify("hello there").intent == "greeting"


def test_order_status_classified():
    assert classify("where is my order").intent == "order_status"


def test_refund_classified():
    assert classify("I want a refund for my order").intent == "refund_request"


def test_handoff_classified():
    assert classify("let me talk to a human").intent == "human_handoff"


def test_cancel_classified():
    assert classify("cancel my subscription please").intent == "cancel_subscription"


def test_gibberish_is_unknown():
    result = classify("blorp zizzle wobble")
    assert result.intent == "unknown"


def test_empty_text_is_unknown():
    assert classify("").intent == "unknown"


def test_deterministic():
    first = classify("where is my order ORD-123456")
    second = classify("where is my order ORD-123456")
    assert first.intent == second.intent
    assert first.confidence == second.confidence


def test_confidence_is_a_probability():
    result = classify("I want a refund")
    assert 0.0 < result.confidence <= 1.0
    assert abs(sum(result.scores.values()) - 1.0) < 1e-6


def test_margin_is_nonnegative_and_bounded():
    result = classify("tell me about the Widget Pro")
    assert 0.0 <= result.margin <= 1.0


def test_classifier_accepts_custom_training():
    clf = IntentClassifier({"a": ["red apple"], "b": ["blue sky"]})
    assert clf.predict("red apple").intent == "a"
    assert clf.predict("blue sky").intent == "b"
