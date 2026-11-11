"""Fail-closed safety screening and PII redaction.

Design principle: when in doubt, block. A blocked turn never reaches the
dialogue policy, never fills slots, and never gets a generated answer beyond
a fixed safe refusal. Screening runs on the raw user text *before* NLU so a
prompt-injection attempt cannot steer intent classification or slot filling.

Categories:
- ``prompt_injection`` — attempts to override instructions or role.
- ``sensitive_data_request`` — asking the assistant to reveal passwords,
  secrets, or another customer's data.
- ``pii_in_message`` — the user pasted a card number / SSN; the message is
  refused and the value is never stored (redaction is a second line of defence).
- ``abuse`` — profanity / harassment directed at the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous |your |the )*instructions?",
    r"disregard (all )?(previous |your |the )*instructions?",
    r"you are now",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"do anything now",
    r"reveal (your |the )?(system prompt|instructions|secret)",
    r"bypass (the )?(safety|filter|guardrail)",
    r"override (your )?(instructions|rules|policy)",
]

_SENSITIVE_REQUEST_PATTERNS = [
    r"\bpassword\b.*\b(for|of)\b.*\baccount\b",
    r"give me (the )?(password|api key|secret)",
    r"show me (another|other) customer",
    r"what is (the )?(password|pin|secret)",
    r"share (my|the) (password|credit card)",
]

_ABUSE_WORDS = ["idiot", "stupid bot", "useless bot", "shut up", "hate you"]


@dataclass
class SafetyResult:
    blocked: bool
    category: str | None = None
    reason: str = ""


def screen(text: str) -> SafetyResult:
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return SafetyResult(True, "prompt_injection",
                                "Message looks like an attempt to override assistant instructions.")
    for pattern in _SENSITIVE_REQUEST_PATTERNS:
        if re.search(pattern, lowered):
            return SafetyResult(True, "sensitive_data_request",
                                "Request for credentials or another customer's data.")
    if _SSN_RE.search(text):
        return SafetyResult(True, "pii_in_message",
                            "Message contains a social security number pattern.")
    if _CARD_RE.search(text):
        return SafetyResult(True, "pii_in_message",
                            "Message contains a payment-card number pattern.")
    if any(word in lowered for word in _ABUSE_WORDS):
        return SafetyResult(True, "abuse", "Message contains abusive language.")
    return SafetyResult(False)


def redact(text: str) -> str:
    """Redact PII from text before it is stored in transcripts or logs."""
    redacted = _SSN_RE.sub("[SSN REDACTED]", text)
    redacted = _EMAIL_RE.sub("[EMAIL REDACTED]", redacted)
    redacted = _CARD_RE.sub("[CARD REDACTED]", redacted)
    return redacted


SAFE_REFUSAL = (
    "I can't help with that request. If you need support with an order, "
    "return, refund, or your account, I can help — or I can connect you "
    "with a human agent."
)
