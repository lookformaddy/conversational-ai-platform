"""Versioned dialogue flows.

A flow is the contract for one intent: which slots are required, what to ask
for each missing slot, and whether the action needs an explicit confirmation
before it executes. Flows are versioned so a behaviour change is a new
version, not a silent edit — old conversations keep the version they started
with, and the eval harness pins the version it runs against.

Version 1.0.0 is the initial release. A v2 would live in ``FLOWS_V2`` and be
selected by passing ``flow_version="2.0.0"`` to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FLOW_VERSION = "1.0.0"


@dataclass
class Flow:
    intent: str
    required_slots: list[str] = field(default_factory=list)
    slot_prompts: dict[str, str] = field(default_factory=dict)
    confirm_before_execute: bool = False
    description: str = ""


FLOWS_V1: dict[str, Flow] = {
    "greeting": Flow(
        intent="greeting",
        description="Open the conversation and offer help.",
    ),
    "order_status": Flow(
        intent="order_status",
        required_slots=["order_id"],
        slot_prompts={"order_id": "Could you share your order number? It looks like ORD-123456."},
        description="Look up the delivery status of an order.",
    ),
    "return_request": Flow(
        intent="return_request",
        required_slots=["product", "order_id", "reason"],
        slot_prompts={
            "product": "Which product would you like to return?",
            "order_id": "What is the order number for the return?",
            "reason": "Could you tell me briefly why you are returning it — for example damaged, wrong item, or changed mind?",
        },
        description="Start a return; needs product, order, and reason.",
    ),
    "refund_request": Flow(
        intent="refund_request",
        required_slots=["order_id"],
        slot_prompts={"order_id": "What is the order number for the refund?"},
        confirm_before_execute=True,
        description="Refund a payment; confirmation required before executing.",
    ),
    "account_help": Flow(
        intent="account_help",
        required_slots=[],
        description="Help with login, password reset, or account details.",
    ),
    "product_info": Flow(
        intent="product_info",
        required_slots=["product"],
        slot_prompts={"product": "Which product would you like to know about?"},
        description="Answer questions about a product in the catalog.",
    ),
    "billing_question": Flow(
        intent="billing_question",
        required_slots=[],
        description="Explain a charge or invoice.",
    ),
    "cancel_subscription": Flow(
        intent="cancel_subscription",
        required_slots=[],
        confirm_before_execute=True,
        description="Cancel a subscription; confirmation required before executing.",
    ),
    "shipping_info": Flow(
        intent="shipping_info",
        required_slots=[],
        description="Answer shipping cost, timing, and coverage questions.",
    ),
    "complaint": Flow(
        intent="complaint",
        required_slots=[],
        description="Acknowledge a complaint and offer a handoff.",
    ),
    "goodbye": Flow(
        intent="goodbye",
        description="Close the conversation politely.",
    ),
    "human_handoff": Flow(
        intent="human_handoff",
        description="Hand the conversation to a human agent with transcript.",
    ),
}

FLOW_REGISTRY: dict[str, dict[str, Flow]] = {"1.0.0": FLOWS_V1}


def get_flow(intent: str, version: str = FLOW_VERSION) -> Flow | None:
    return FLOW_REGISTRY.get(version, FLOWS_V1).get(intent)


def flow_version() -> str:
    return FLOW_VERSION
