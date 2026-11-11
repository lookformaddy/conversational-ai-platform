"""Synthetic training data, product catalog, and eval dialogues.

Everything here is invented for this project: generic product names, fake
order ids, and hand-written example utterances. No real customer data, no
employer data, no scraped conversations.
"""

from __future__ import annotations

PRODUCT_CATALOG: list[str] = [
    "Widget Pro",
    "Gadget Max",
    "Home Hub",
    "Power Bank Plus",
    "Smart Lamp",
    "Travel Charger",
]

# Training utterances per intent for the from-scratch Naive Bayes classifier.
# Kept deliberately small and deterministic; the eval harness measures the
# accuracy this yields on held-out synthetic dialogues.
TRAINING_UTTERANCES: dict[str, list[str]] = {
    "greeting": [
        "hello", "hi there", "good morning", "hey", "good afternoon",
        "hi, I need some help", "hello there", "good evening",
    ],
    "order_status": [
        "where is my order", "track my order", "what is the status of my order",
        "has my order shipped yet", "when will my order arrive",
        "check order status for order ORD-123456", "where is my package",
        "delivery status please",
    ],
    "return_request": [
        "I want to return my item", "how do I return this product",
        "I need to send this back", "start a return please",
        "return my order, it arrived damaged", "can I return the Widget Pro",
        "I would like to return my purchase", "initiate a return",
    ],
    "refund_request": [
        "I want a refund", "please refund my money", "give me my money back",
        "request a refund for my order", "I was charged twice, refund me",
        "refund the payment for order ORD-123456", "how do I get a refund",
        "I need a refund for the damaged item",
    ],
    "account_help": [
        "I cannot log in", "reset my password", "I forgot my password",
        "my account is locked", "help me update my email address",
        "change my account details", "I cannot access my account",
        "update my phone number on my account",
    ],
    "product_info": [
        "tell me about the Widget Pro", "what are the features of the Gadget Max",
        "how much does the Home Hub cost", "is the Smart Lamp in stock",
        "what is the warranty on the Power Bank Plus",
        "compare the Widget Pro and the Gadget Max",
        "does the Travel Charger work with my phone",
        "product specifications for the Home Hub",
    ],
    "billing_question": [
        "why was I charged twice", "what is this charge on my bill",
        "explain the charge on my statement", "my invoice looks wrong",
        "I do not recognise this payment", "billing question about my last invoice",
        "when will I be charged", "my card was charged incorrectly",
    ],
    "cancel_subscription": [
        "cancel my subscription", "I want to cancel my plan",
        "stop my membership please", "end my subscription",
        "cancel the monthly plan", "do not renew my subscription",
        "turn off auto renewal", "close my subscription",
    ],
    "shipping_info": [
        "how much is shipping", "do you ship internationally",
        "what are the delivery options", "how long does delivery take",
        "can I change my delivery address", "shipping costs to Canada",
        "do you offer express delivery", "what countries do you ship to",
        "what is the shipping cost", "what about the shipping cost",
        "how much does shipping cost", "shipping cost for my order",
    ],
    "complaint": [
        "I am very unhappy with the service", "this is unacceptable",
        "I want to complain about my experience", "your product broke after one day",
        "terrible experience, I am disappointed", "I am frustrated with the delay",
        "this is the worst service I have had", "I want to file a complaint",
    ],
    "goodbye": [
        "thank you, goodbye", "thanks, that is all", "bye", "that solves it, thanks",
        "goodbye", "thanks for your help", "no more questions, bye",
        "that is everything, thank you",
    ],
    "human_handoff": [
        "let me talk to a human", "connect me to an agent",
        "I want to speak to a real person", "transfer me to support",
        "speak to a human agent please", "get me a live agent",
        "I need a human to help me", "escalate this to a person",
    ],
}

INTENTS: list[str] = list(TRAINING_UTTERANCES) + ["unknown"]

# Synthetic multi-turn dialogues for the eval harness. Each turn states the
# expected intent and the slots the turn should fill; the final action is
# checked at the end of the dialogue.
EVAL_DIALOGUES: list[dict] = [
    {
        "id": "eval-order-status",
        "turns": [
            {"user": "Hi", "expected_intent": "greeting", "expected_slots": {}},
            {"user": "Where is my order ORD-123456?", "expected_intent": "order_status",
             "expected_slots": {"order_id": "ORD-123456"}},
        ],
        "expected_final_action": "execute",
    },
    {
        "id": "eval-return-flow",
        "turns": [
            {"user": "I want to return the Widget Pro", "expected_intent": "return_request",
             "expected_slots": {"product": "Widget Pro"}},
            {"user": "My order is ORD-654321", "expected_intent": "return_request",
             "expected_slots": {"order_id": "ORD-654321"}},
        ],
        "expected_final_action": "ask_slot",
    },
    {
        "id": "eval-refund-confirm",
        "turns": [
            {"user": "I want a refund for order ORD-111111", "expected_intent": "refund_request",
             "expected_slots": {"order_id": "ORD-111111"}},
            {"user": "Yes, please go ahead", "expected_intent": "unknown",
             "expected_slots": {}},
        ],
        "expected_final_action": "execute",
    },
    {
        "id": "eval-handoff",
        "turns": [
            {"user": "Let me talk to a human", "expected_intent": "human_handoff",
             "expected_slots": {}},
        ],
        "expected_final_action": "handoff",
    },
    {
        "id": "eval-clarify-unknown",
        "turns": [
            {"user": "blorp zizzle wobble", "expected_intent": "unknown",
             "expected_slots": {}},
        ],
        "expected_final_action": "clarify_intent",
    },
    {
        "id": "eval-cancel-confirm",
        "turns": [
            {"user": "Cancel my subscription", "expected_intent": "cancel_subscription",
             "expected_slots": {}},
            {"user": "Yes", "expected_intent": "unknown", "expected_slots": {}},
        ],
        "expected_final_action": "execute",
    },
    {
        "id": "eval-carryover",
        "turns": [
            {"user": "Where is my order ORD-222222?", "expected_intent": "order_status",
             "expected_slots": {"order_id": "ORD-222222"}},
            {"user": "And what about the shipping cost?", "expected_intent": "shipping_info",
             "expected_slots": {"order_id": "ORD-222222"}},
        ],
        "expected_final_action": "respond",
    },
    {
        "id": "eval-goodbye",
        "turns": [
            {"user": "Thanks, that is all, goodbye", "expected_intent": "goodbye",
             "expected_slots": {}},
        ],
        "expected_final_action": "close",
    },
]
