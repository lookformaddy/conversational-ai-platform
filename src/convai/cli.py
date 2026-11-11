"""Command-line interface: interactive REPL, single message, demo, eval.

Usage:
    python -m convai.cli               # interactive REPL
    python -m convai.cli --message "Where is my order ORD-123456?"
    python -m convai.cli --demo        # scripted multi-turn demo
    python -m convai.cli --eval        # run the synthetic-dialogue eval harness
"""

from __future__ import annotations

import argparse
import sys

from convai.engine import ConversationEngine
from convai.eval import run_eval

DEMO_SCRIPT: list[str] = [
    "Hello",
    "Where is my order ORD-123456?",
    "I want to return the Widget Pro",
    "My order is ORD-123456",
    "It arrived damaged",
    "I want a refund for order ORD-123456",
    "Yes, please go ahead",
    "Thanks, goodbye",
]


def run_demo(engine: ConversationEngine | None = None) -> None:
    engine = engine or ConversationEngine()
    conversation_id = engine.start_conversation()
    print(f"Conversation: {conversation_id}\n")
    for message in DEMO_SCRIPT:
        turn = engine.handle_message(conversation_id, message)
        print(f"You: {message}")
        print(f"Assistant [{turn.intent} · {turn.action.value} · "
              f"conf={turn.confidence:.2f}]: {turn.response}\n")
    print("Analytics:", engine.analytics.metrics())


def run_repl(engine: ConversationEngine | None = None) -> None:
    engine = engine or ConversationEngine()
    conversation_id = engine.start_conversation()
    print("Conversational AI Platform — type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if message.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not message:
            continue
        turn = engine.handle_message(conversation_id, message)
        print(f"Assistant [{turn.intent} · {turn.action.value}]: {turn.response}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conversational AI platform CLI")
    parser.add_argument("--message", help="Send a single message and exit")
    parser.add_argument("--demo", action="store_true", help="Run the scripted demo")
    parser.add_argument("--eval", action="store_true", help="Run the eval harness")
    args = parser.parse_args(argv)

    if args.eval:
        report = run_eval()
        print(f"Dialogues: {len(report.dialogues)}")
        print(f"Intent accuracy: {report.intent_accuracy:.3f}")
        print(f"Slot accuracy: {report.slot_accuracy:.3f}")
        print(f"Final-action accuracy: {report.final_action_accuracy:.3f}")
        for dialogue in report.dialogues:
            mark = "OK " if dialogue.final_action_correct else "FAIL"
            print(f"  [{mark}] {dialogue.dialogue_id}: final={dialogue.final_action} "
                  f"(expected {dialogue.expected_final_action})")
        return 0 if report.final_action_accuracy == 1.0 else 1

    if args.demo:
        run_demo()
        return 0

    engine = ConversationEngine()
    if args.message:
        conversation_id = engine.start_conversation()
        turn = engine.handle_message(conversation_id, args.message)
        print(f"[{turn.intent} · {turn.action.value} · conf={turn.confidence:.2f}] {turn.response}")
        return 0

    run_repl(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
