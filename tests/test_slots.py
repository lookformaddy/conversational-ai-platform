from convai.slots import extract_slots, is_affirmative, is_negative


def test_order_id_with_prefix():
    assert extract_slots("Where is my order ORD-123456?")["order_id"] == "ORD-123456"


def test_order_id_with_hash():
    assert extract_slots("Check #654321 please")["order_id"] == "ORD-654321"


def test_order_id_without_prefix():
    assert extract_slots("My order is 123456")["order_id"] == "ORD-123456"


def test_email_extracted():
    slots = extract_slots("My email is jane@example.com")
    assert slots["email"] == "jane@example.com"


def test_product_matched_case_insensitively():
    assert extract_slots("Tell me about the widget pro")["product"] == "Widget Pro"


def test_reason_damaged():
    assert extract_slots("It arrived damaged")["reason"] == "damaged"


def test_reason_wrong_item():
    assert extract_slots("This is the wrong item")["reason"] == "wrong_item"


def test_amount_extracted():
    assert extract_slots("I was charged $49.99")["amount"] == 49.99


def test_relative_date():
    assert extract_slots("Deliver it tomorrow")["date"] == "tomorrow"


def test_no_slots_for_plain_greeting():
    assert extract_slots("Hello there") == {}


def test_affirmative_and_negative():
    assert is_affirmative("Yes, please go ahead")
    assert is_affirmative("sure")
    assert not is_affirmative("no thanks")
    assert is_negative("No, cancel that")
    assert not is_negative("yes please")
