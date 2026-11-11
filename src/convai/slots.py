"""Deterministic slot filling with regular expressions and catalog lookup.

Slots are extracted per turn and merged into dialogue state by the state
tracker (carryover). Nothing here calls an external service; patterns are
anchored to the synthetic domain (fake order ids, generic product names).
"""

from __future__ import annotations

import re
from typing import Any

from convai.data import PRODUCT_CATALOG

_ORDER_RE = re.compile(r"\bord[- ]?(\d{5,})\b", re.IGNORECASE)
_HASH_ORDER_RE = re.compile(r"#(\d{5,})\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MONEY_RE = re.compile(r"\$\s?(\d+(?:\.\d{2})?)")
_DATE_ISO_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RELATIVE_DATE_RE = re.compile(r"\b(today|tomorrow|next week|next monday|next friday)\b", re.IGNORECASE)
_QUANTITY_RE = re.compile(r"\b(\d+)\s+(?:items?|units?|pieces?)\b", re.IGNORECASE)

REASON_KEYWORDS: dict[str, list[str]] = {
    "damaged": ["damaged", "broken", "cracked", "defective"],
    "wrong_item": ["wrong item", "wrong product", "not what I ordered", "incorrect item"],
    "changed_mind": ["changed my mind", "no longer need", "do not want it anymore", "dont want it"],
    "late_delivery": ["arrived late", "too late", "late delivery", "delayed"],
    "not_as_described": ["not as described", "does not match", "different from the description"],
}

YES_RE = re.compile(r"^\s*(yes|yeah|yep|sure|ok|okay|go ahead|confirm|please do|do it)\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(no|nope|not now|cancel that|never mind|do not|don't)\b", re.IGNORECASE)


def is_affirmative(text: str) -> bool:
    return bool(YES_RE.search(text.strip()))


def is_negative(text: str) -> bool:
    return bool(NO_RE.search(text.strip()))


def extract_slots(text: str) -> dict[str, Any]:
    """Extract all slots present in one utterance (empty dict if none)."""
    slots: dict[str, Any] = {}
    lowered = text.lower()

    match = _ORDER_RE.search(text) or _HASH_ORDER_RE.search(text)
    if match is None and "order" in lowered:
        # Bare number after the word "order": "My order is 123456".
        bare = re.search(r"\b(\d{5,})\b", text)
        if bare:
            match = bare
    if match:
        slots["order_id"] = f"ORD-{match.group(1)}"

    email = _EMAIL_RE.search(text)
    if email:
        slots["email"] = email.group(0)

    for product in PRODUCT_CATALOG:
        if product.lower() in lowered:
            slots["product"] = product
            break

    for reason, keywords in REASON_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            slots["reason"] = reason
            break

    money = _MONEY_RE.search(text)
    if money:
        slots["amount"] = float(money.group(1))

    date_match = _DATE_ISO_RE.search(text) or _RELATIVE_DATE_RE.search(text)
    if date_match:
        slots["date"] = date_match.group(1).lower()

    quantity = _QUANTITY_RE.search(text)
    if quantity:
        slots["quantity"] = int(quantity.group(1))

    if "canada" in lowered:
        slots["country"] = "Canada"
    elif "united states" in lowered or " usa" in lowered or " us " in f" {lowered} ":
        slots["country"] = "United States"

    return slots
