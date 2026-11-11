# Conversational AI Platform

A multi-turn dialogue manager for enterprise customer support — deterministic
intent classification + slot filling built from scratch, dialogue state
tracking with context carryover, policy-driven next-action selection,
clarification / disambiguation handling, human handoff with transcript,
conversation analytics, a fail-closed safety screen, versioned dialogue
flows, and an eval harness over synthetic dialogues.

No LLM calls, no sklearn, no external services at runtime. The core is pure
Python standard library; FastAPI is an optional dependency for the HTTP
service only.

## The enterprise problem

Support chatbots fail in predictable ways: they forget what the customer said
two turns ago, they act on a misunderstood request (refunding the wrong
order), they loop on clarification forever, and when they do hand off, the
human agent starts from zero. This platform addresses each failure directly:

- **State tracking with carryover** — slots filled in earlier turns stay
  filled, so "And what about the shipping cost?" still knows the order.
- **Active-flow continuation** — an answer to a slot question ("My order is
  ORD-123456") continues the pending return flow instead of being
  re-classified as a brand-new order-status request.
- **Confirmation before irreversible actions** — refunds and subscription
  cancellations require an explicit yes.
- **Fail-closed safety** — prompt injection, credential requests, and pasted
  card/SSN numbers are blocked before NLU runs; PII is redacted from every
  stored transcript.
- **Handoff with transcript** — the human agent receives the full transcript,
  slots, and intent history.
- **Measurable** — analytics (containment, handoff rate, intent mix) and an
  eval harness over synthetic dialogues, so regressions are numbers, not vibes.

## Architecture

```
User text
   │  1. Safety screen (fail-closed: injection / credentials / PII / abuse)
   ▼
   │  2. NLU: from-scratch Naive Bayes intent classifier (unigrams + bigrams,
   │     Laplace smoothing) + regex/catalog slot filling
   ▼
   │  3. State tracker: slots merge with carryover, intent history, pending
   │     confirmation, active flow
   ▼
   │  4. Policy (pure function): refuse → handoff → clarify → disambiguate →
   │     ask slot → confirm → execute / respond
   ▼
Response (templated) + turn recorded (PII-redacted)
```

Module map (`src/convai/`):

| Module | Responsibility |
|---|---|
| `nlu.py` | Multinomial Naive Bayes classifier, trained on synthetic utterances |
| `slots.py` | Regex + catalog slot extraction (order id, email, product, reason, …) |
| `safety.py` | Fail-closed screen + PII redaction |
| `state.py` | Dialogue state tracker (carryover, confirmations, clarifications) |
| `flows.py` | Versioned dialogue flows (required slots, prompts, confirmations) |
| `policy.py` | Pure-function next-action selection |
| `engine.py` | Orchestrates the pipeline; owns conversations and handoff |
| `analytics.py` | Containment / handoff / intent-mix metrics |
| `eval.py` | Eval harness over synthetic multi-turn dialogues |
| `service.py` | FastAPI service (optional dependency) |
| `cli.py` | REPL, single message, demo, eval |
| `data.py` | Synthetic training utterances, catalog, eval dialogues |

Full detail: [docs/architecture.md](docs/architecture.md).

## How to run

```bash
python -m venv /tmp/convai-venv
source /tmp/convai-venv/bin/activate
pip install -r requirements.txt

# Interactive chat
python -m convai.cli

# Scripted multi-turn demo (order → return → refund with confirmation)
python -m convai.cli --demo

# Eval harness over synthetic dialogues
python -m convai.cli --eval

# Single message
python -m convai.cli --message "Where is my order ORD-123456?"

# HTTP service (needs FastAPI)
python -m convai.service   # http://localhost:8000/health
```

Set `PYTHONPATH=src` (or install with `pip install -e .`) so `convai` is
importable. Tests: `pytest` (API tests auto-skip without FastAPI/httpx).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status + flow version |
| `POST /chat` | `{"conversation_id"?, "message"}` → intent, confidence, action, response, slots |
| `GET /conversations/{id}` | State, turns, redacted transcript |
| `GET /analytics` | Containment, handoff, resolution rates, intent mix |
| `GET /flows` | Versioned flow definitions |

## Design decisions and trade-offs

- **From-scratch Naive Bayes instead of an LLM or sklearn.** Deterministic,
  auditable, zero inference cost, and no data leaves the process. Trade-off:
  accuracy is bounded by the training utterances — the eval harness measures
  exactly that, and new intents are added by adding utterances, not by
  re-prompting a model.
- **Templated responses instead of generation.** No hallucinated order
  statuses or policies. Trade-off: less conversational variety; acceptable
  for support, where correctness beats charm.
- **Active-flow continuation wins over re-classification** when a turn fills
  a missing slot of the pending flow. Trade-off: a user who genuinely
  switches topics mid-flow with a slot-bearing message is treated as
  continuing; they can switch by naming the new intent without slot values.
- **Confirmation is a state, not a prompt.** The pending confirmation is
  answered before any new NLU work, so "yes" can never be misread as a new
  request.
- **Fail-closed safety.** A blocked turn never reaches NLU or the policy, and
  the refusal text is fixed — the model cannot be talked into a different one.
- **Versioned flows.** Behaviour changes ship as a new flow version; the eval
  harness pins the version it runs against.
- **Simulated backend.** Order status is derived deterministically from the
  order number; refunds/returns/cancellations are simulated executions. A
  real deployment swaps `_execution_response` for service calls behind the
  same policy boundary.

## Eval results

`python -m convai.cli --eval` over 8 synthetic dialogues: intent accuracy
1.000, slot accuracy 1.000, final-action accuracy 1.000 (see
`tests/test_analytics_eval.py`, which asserts all three).

## What it demonstrates

Deterministic NLU built from scratch · dialogue state tracking and context
carryover · policy-as-pure-function design · confirmation gates for
irreversible actions · fail-closed safety + PII redaction · human handoff
with full context · conversation analytics · versioned dialogue flows ·
eval-driven development with a synthetic harness · clean layered
architecture with an optional-dependency service boundary.
