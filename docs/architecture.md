# Architecture — Conversational AI Platform

## Overview

The platform is a layered pipeline. Each layer is a pure function (or a thin
owner of state) over the previous layer's output, so every decision can be
tested in isolation and the whole turn is reproducible.

```
                    ┌─────────────────────┐
                    │  CLI / FastAPI      │  transport only
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  ConversationEngine │  owns conversations + state
                    └─────────┬───────────┘
        ┌─────────────┬───────┴───────┬──────────────┐
        ▼             ▼               ▼              ▼
   ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐
   │ Safety  │  │   NLU    │  │   State    │  │  Policy   │
   │ screen  │  │ classify │  │  tracker   │  │  decide   │
   │ + redact│  │ + slots  │  │  + flows   │  │  (pure fn)│
   └─────────┘  └──────────┘  └────────────┘  └───────────┘
        │             │               │              │
        └─────────────┴───────┬───────┴──────────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Response + Turn    │  templated, redacted
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  Analytics / Eval   │  metrics over finished convs
                    └─────────────────────┘
```

## Turn pipeline (one message)

1. **Safety screen** (`safety.py`) — regex-based, runs on raw text *before*
   NLU. Categories: prompt injection, sensitive-data requests, pasted
   card/SSN numbers, abuse. Any hit → fixed safe refusal, no slot filling,
   no state mutation beyond recording the refused turn. Fail-closed by
   design: an unknown pattern is not a pass, it is simply not a known block
   — the screen is one layer, and templated responses bound what a
   successful bypass could ever make the system say.
2. **Pending confirmation** — if a confirmation is pending, yes/no is
   answered from state, not re-classified. Any other message clears the
   pending confirmation and is processed as a new request.
3. **NLU** (`nlu.py`, `slots.py`) — Naive Bayes over unigram + bigram
   features with Laplace smoothing; softmax confidence; slot extraction by
   regex and catalog lookup. Trained on the synthetic utterances in
   `data.py`; no external model, no network.
4. **Active-flow continuation** (`engine.py`) — if the previous turn left a
   flow waiting for slots and this turn fills one of them, the turn is
   treated as continuing that flow (intent overridden, confidence lifted).
   Greeting / goodbye / handoff always take precedence. This is the single
   most important multi-turn behaviour: without it, answering a slot
   question looks like a brand-new request.
5. **State update** (`state.py`) — slots merge with carryover (new values
   replace, absent slots persist), intent appended to history, turn counted.
6. **Policy** (`policy.py`) — pure function, decision order: refuse →
   handoff (explicit or frustrated) → clarify (unknown / low confidence;
   handoff after 3 failed clarifications) → disambiguate (margin < 0.08
   with a strong runner-up) → ask for the next missing slot → confirm
   (refund / cancel) → execute / respond.
7. **Response** — templated per action/intent. Execution responses are
   simulated deterministically (order status derived from the order
   number). The turn is recorded with PII redacted.
8. **Bookkeeping** — resolved / closed / handed-off conversations are
   recorded once in analytics (repeat records update, never double-count).

## Dialogue flows (versioned)

`flows.py` defines one flow per intent: required slots, the prompt for each
missing slot, and whether execution needs confirmation. Flows live in a
registry keyed by version (`"1.0.0"` today); the engine and the eval harness
pin a version, so a behaviour change is a new version, not a silent edit.

## Handoff

`engine.build_handoff()` packages conversation id, reason, a one-line
summary, slots, intent history, and the full redacted transcript. Priority
is `high` after 3+ clarifications, otherwise `normal`. The handoff message
tells the customer they will not have to repeat themselves — and the
transcript makes that true.

## Analytics

`analytics.py` aggregates finished conversations only and stores no raw
user text. Metrics: conversation count, average turns, containment rate
(resolved without handoff), handoff rate, resolution rate, average
clarifications, intent distribution, top intents.

## Eval harness

`eval.py` runs the 8 synthetic dialogues in `data.py` through fresh
engines and reports intent accuracy (excluding bare-confirmation turns,
which are answered from pending state), slot accuracy against cumulative
carried-over state, and final-action accuracy per dialogue. Current
results: 1.000 / 1.000 / 1.000.

## Data flow and privacy

- User text enters through the CLI or the FastAPI `/chat` endpoint.
- PII is redacted before any turn is stored; transcripts shown to agents
  or in analytics contain only redacted text.
- No user data leaves the process: no LLM calls, no external APIs, no
  persistence beyond in-memory conversations (a deployment would add a
  store behind the engine boundary).
- All training and eval data is synthetic and hand-written for this project.

## Failure modes and limits

- **NLU coverage** is bounded by the training utterances; genuinely novel
  phrasing falls back to clarification, then handoff after 3 attempts.
  This is the intended degradation path, not a bug.
- **Topic switching mid-flow** with a slot-bearing message is treated as
  flow continuation (see README trade-offs).
- **Simulated backend** — order status, refunds, returns, and cancellations
  are not real. The swap point is `ConversationEngine._execution_response`.
- **In-memory state** — conversations do not survive a restart; add a
  repository behind the engine for production use.
- **Safety screen is pattern-based** — it catches known injection and PII
  shapes, not adversarial paraphrases. Defence in depth comes from
  templated responses (the system can only say what templates allow).
