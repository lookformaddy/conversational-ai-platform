"""Conversational AI platform — multi-turn dialogue manager for enterprise support.

Pure-stdlib core: deterministic intent classification + slot filling built
from scratch (no sklearn / no LLM calls), dialogue state tracking with
context carryover, policy-driven next-action selection, fail-closed safety
screening, versioned dialogue flows, handoff, analytics, and an eval harness
over synthetic dialogues. FastAPI is an optional dependency for the service
layer only; the core and all non-API tests run on the standard library.
"""

from convai.models import Action, Conversation, DialogueState, IntentResult, Turn
from convai.engine import ConversationEngine

__all__ = ["Action", "Conversation", "ConversationEngine", "DialogueState", "IntentResult", "Turn"]
__version__ = "1.0.0"
