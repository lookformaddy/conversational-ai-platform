"""Deterministic intent classification, built from scratch.

Multinomial Naive Bayes with Laplace smoothing, trained on the synthetic
utterances in :mod:`convai.data`. No sklearn, no embeddings, no LLM calls —
classification is a pure function of the training data, so results are
reproducible turn after turn. Confidence is the softmax of log-likelihoods;
the margin between the top two intents drives disambiguation in the policy.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from convai.data import TRAINING_UTTERANCES
from convai.models import IntentResult

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Utterances whose token overlap is so weak that no intent should claim them.
# 0.20 separates genuine single-word inputs ("hello" scores ~0.23) from
# out-of-vocabulary gibberish (top intent ~0.13); keep in sync with the
# policy's LOW_CONFIDENCE.
UNKNOWN_CONFIDENCE_THRESHOLD = 0.20


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class IntentClassifier:
    """Multinomial Naive Bayes over word unigrams (and bigrams for phrases)."""

    def __init__(self, training: dict[str, list[str]] | None = None) -> None:
        self.training = training or TRAINING_UTTERANCES
        self.intents: list[str] = sorted(self.training)
        self._doc_counts: Counter[str] = Counter()
        self._word_counts: dict[str, Counter[str]] = {}
        self._total_words: dict[str, int] = {}
        self._vocab: set[str] = set()
        self._fit()

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _fit(self) -> None:
        for intent, utterances in self.training.items():
            counts: Counter[str] = Counter()
            for utterance in utterances:
                feats = self._features(utterance)
                counts.update(feats)
                self._vocab.update(feats)
            self._word_counts[intent] = counts
            self._total_words[intent] = sum(counts.values())
            self._doc_counts[intent] = len(utterances)

    def _log_likelihood(self, intent: str, features: list[str]) -> float:
        total_docs = sum(self._doc_counts.values())
        log_prob = math.log(self._doc_counts[intent] / total_docs)
        vocab_size = len(self._vocab)
        denom = self._total_words[intent] + vocab_size  # Laplace smoothing
        for feat in features:
            count = self._word_counts[intent].get(feat, 0)
            log_prob += math.log((count + 1) / denom)
        return log_prob

    def predict(self, text: str) -> IntentResult:
        features = self._features(text)
        if not features:
            return IntentResult(intent="unknown", confidence=1.0, scores={"unknown": 1.0})
        log_probs = {intent: self._log_likelihood(intent, features) for intent in self.intents}
        top = max(log_probs.values())
        # Softmax over log-likelihoods for a calibrated-ish confidence.
        exp_scores = {k: math.exp(v - top) for k, v in log_probs.items()}
        total = sum(exp_scores.values())
        probs = {k: v / total for k, v in exp_scores.items()}
        best_intent = max(probs, key=probs.get)
        best_prob = probs[best_intent]
        if best_prob < UNKNOWN_CONFIDENCE_THRESHOLD:
            return IntentResult(intent="unknown", confidence=1.0 - best_prob, scores=probs)
        return IntentResult(intent=best_intent, confidence=best_prob, scores=probs)


# Module-level singleton: training is fixed, so one classifier is enough.
_DEFAULT_CLASSIFIER = IntentClassifier()


def classify(text: str) -> IntentResult:
    return _DEFAULT_CLASSIFIER.predict(text)
