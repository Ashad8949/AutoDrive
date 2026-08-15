"""
AutoDrive RAG v2.0 — Intent Classifier
Classifies user queries into predefined intents to route them
to the optimal retrieval/generation pipeline.

Intents:
  - inventory_search: Looking for cars matching criteria
  - comparison: Comparing two or more cars
  - recommendation: Asking for suggestions
  - booking: Wanting to book a test drive
  - specs_lookup: Asking about specific car details
  - general_chat: Greeting, small talk, off-topic
  - price_query: Specifically about pricing/deals
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("chatbot.query.intent")


class Intent(str, Enum):
    """Supported user intents."""

    INVENTORY_SEARCH = "inventory_search"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    BOOKING = "booking"
    SPECS_LOOKUP = "specs_lookup"
    GENERAL_CHAT = "general_chat"
    PRICE_QUERY = "price_query"


# ── Rule-based patterns for intent classification ───────────────────

INTENT_PATTERNS = {
    Intent.BOOKING: [
        r"book\s*(a\s*)?test\s*drive",
        r"schedule\s*(a\s*)?test\s*drive",
        r"test\s*drive",
        r"want\s*to\s*drive",
        r"can\s*i\s*try",
        r"visit\s*(the\s*)?showroom",
    ],
    Intent.COMPARISON: [
        r"compare\b",
        r"comparison\b",
        r"versus\b",
        r"\bvs\.?\b",
        r"difference\s*between",
        r"which\s*(one\s*)?(is\s*)?better",
        r"head\s*to\s*head",
        r"\bor\b.*\bwhich\b",
    ],
    Intent.RECOMMENDATION: [
        r"recommend",
        r"suggest",
        r"what\s*(car\s*)?(should|would|do)\s*(you|i)",
        r"best\s*(car|option|choice)",
        r"which\s*(car\s*)?should",
        r"help\s*me\s*(choose|pick|find|decide)",
        r"looking\s*for\s*(a|some)",
        r"what\'?s?\s*good",
        r"any\s*good",
    ],
    Intent.PRICE_QUERY: [
        r"(how\s*much|what\'?s?\s*the\s*price|pricing|cost|rate)",
        r"(discount|deal|offer|emi|finance|loan|installment)",
        r"(affordable|cheap|budget|value\s*for\s*money|bang\s*for)",
        r"under\s*\d+",
        r"₹\s*\d+",
        r"lakh",
    ],
    Intent.SPECS_LOOKUP: [
        r"(spec|specification|feature|detail|info)",
        r"(mileage|fuel\s*economy|range|power|torque|engine|bhp|hp)",
        r"(boot\s*space|ground\s*clearance|dimension|weight)",
        r"(color|colour|variant|trim|top\s*model|base\s*model)",
        r"(safety\s*rating|ncap|airbag|abs|adas)",
        r"tell\s*me\s*(more\s*)?about",
    ],
    Intent.INVENTORY_SEARCH: [
        r"show\s*me",
        r"(list|display|find)\s*(all|me)?",
        r"(any|have|got)\s*(any)?\s*(car|vehicle|suv|sedan|hatchback)",
        r"available\s*(car|vehicle|in)",
        r"(diesel|petrol|electric|automatic|manual)\s*(car|suv|sedan)?",
        r"in\s*(delhi|mumbai|bangalore|pune|chennai|hyderabad|kolkata)",
    ],
    Intent.GENERAL_CHAT: [
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|thanks|thank\s*you)\b",
        r"^(how\s*are\s*you|what\'?s?\s*up|who\s*are\s*you)",
        r"^(bye|goodbye|see\s*you|take\s*care)",
        r"^(ok|okay|sure|got\s*it|alright|cool|nice|great|awesome)",
    ],
}

# LLM-based classification prompt
CLASSIFY_PROMPT = """\
Classify the following user query into exactly ONE of these intents:
- inventory_search: Looking for cars matching certain criteria
- comparison: Comparing two or more cars
- recommendation: Asking for car suggestions/advice
- booking: Wanting to book or schedule a test drive
- specs_lookup: Asking about specific car details/specifications
- price_query: Questions specifically about pricing, deals, or financing
- general_chat: Greetings, small talk, or off-topic

User query: "{query}"

Respond with ONLY the intent name (e.g., "inventory_search"). Nothing else."""


class IntentClassifier:
    """
    Classifies user queries into predefined intents.

    Supports two modes:
      1. Rule-based (default, fast, free) — regex pattern matching
      2. LLM-based (optional, more accurate) — few-shot classification

    Attributes:
        llm: Optional LLM for LLM-based classification.
        use_llm: Whether to use LLM classification.
    """

    def __init__(self, llm=None, use_llm: bool = False) -> None:
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def classify(self, query: str) -> Intent:
        """
        Classify a user query into an intent.

        Args:
            query: The user's natural language query.

        Returns:
            The classified Intent enum value.
        """
        if self.use_llm:
            return self._classify_llm(query)
        return self._classify_rules(query)

    def classify_with_confidence(self, query: str) -> dict:
        """
        Classify with confidence scores for each intent.

        Args:
            query: The user's natural language query.

        Returns:
            Dict with 'intent', 'confidence', and 'all_scores'.
        """
        scores = self._score_all_intents(query)
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        return {
            "intent": best_intent,
            "confidence": best_score,
            "all_scores": scores,
        }

    def _classify_rules(self, query: str) -> Intent:
        """Rule-based classification using regex patterns."""
        scores = self._score_all_intents(query)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else Intent.INVENTORY_SEARCH

    def _score_all_intents(self, query: str) -> dict[Intent, float]:
        """Score all intents using pattern matching."""
        query_lower = query.lower().strip()
        scores: dict[Intent, float] = {intent: 0.0 for intent in Intent}

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += 1.0

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for intent in scores:
                scores[intent] /= total

        return scores

    def _classify_llm(self, query: str) -> Intent:
        """LLM-based classification using few-shot prompting."""
        try:
            prompt = CLASSIFY_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip().lower()

            # Try to match to a valid intent
            for intent in Intent:
                if intent.value in text:
                    return intent

            logger.warning("LLM returned unrecognized intent: %s", text)
            return Intent.INVENTORY_SEARCH

        except Exception as e:
            logger.warning("LLM classification failed: %s — using rules", e)
            return self._classify_rules(query)

    def __repr__(self) -> str:
        mode = "LLM" if self.use_llm else "rule-based"
        return f"IntentClassifier(mode={mode})"
