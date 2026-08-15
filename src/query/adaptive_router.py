"""
AutoDrive RAG v2.0 — Adaptive Router
Dynamically routes queries to the optimal retrieval/generation pipeline
based on query complexity analysis.

Routes:
  SIMPLE  → Direct metadata lookup or cached response (fast, cheap)
  STANDARD → Hybrid retrieval → LLM generation (balanced)
  COMPLEX  → Agentic multi-step RAG with decomposition (thorough, expensive)

This optimizes latency and cost: simple queries don't need expensive
multi-hop retrieval, while complex queries get the full treatment.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

from .intent_classifier import Intent, IntentClassifier

logger = logging.getLogger("chatbot.query.router")


class QueryComplexity(str, Enum):
    """Query complexity levels determining pipeline routing."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


# Signals that indicate complex queries
COMPLEX_SIGNALS = [
    r"\bcompare\b",
    r"\bversus\b|\bvs\.?\b",
    r"\bdifference\b",
    r"\bbetter\b.*\bor\b",
    r"\bpros\s*(and|&)\s*cons\b",
    r"\btop\s*\d+\b",
    r"\brank\b",
    r"\ball\b.*\b(cars|vehicles|options)\b",
    r"\bwhich\b.*\band\b.*\band\b",
    r"\bif\b.*\bthen\b",
    r"\bfor\s*a\s*family\s*of\b",
    r"\blong\s*(road\s*)?trip\b",
    r"\bbest\b.*\bfor\b",
    r"\banalyze\b|\banalysis\b",
    r"\bdetailed\b|\bin\s*depth\b|\bcomprehensive\b",
]

# Signals that indicate simple queries
SIMPLE_SIGNALS = [
    r"^(hi|hello|hey|thanks|ok|bye)\b",
    r"^what\s*(is|\'s)\s*the\s*(price|cost|color|colour|mileage)\s*of",
    r"^(yes|no|sure|okay)\b",
    r"^how\s*much\s*(does|is)",
    r"^(show|list)\s*(me\s*)?car\s*(id|#|number)",
    r"^(tell\s*me\s*)?about\s*car\s*(id|#|number)?\s*\d+",
]


class AdaptiveRouter:
    """
    Routes queries to the optimal pipeline based on complexity.

    Analyzes query complexity using heuristics and optional LLM
    classification, then recommends the appropriate retrieval strategy.

    Attributes:
        intent_classifier: IntentClassifier for intent detection.
        llm: Optional LLM for more nuanced complexity assessment.
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        llm=None,
    ) -> None:
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.llm = llm

    def route(self, query: str) -> dict:
        """
        Analyze query and determine the optimal pipeline.

        Args:
            query: User's natural language query.

        Returns:
            Dict with routing decision:
              {
                'complexity': QueryComplexity,
                'intent': Intent,
                'pipeline': str,          # Recommended pipeline name
                'strategies': list[str],  # Query transformation strategies
                'retrieval_k': int,       # Suggested top-k for retrieval
                'use_reranker': bool,     # Whether to use cross-encoder
                'reasoning': str,         # Explanation of routing decision
              }
        """
        intent = self.intent_classifier.classify(query)
        complexity = self._assess_complexity(query, intent)

        # Determine pipeline based on complexity + intent
        if complexity == QueryComplexity.SIMPLE:
            pipeline = self._simple_pipeline(intent)
        elif complexity == QueryComplexity.COMPLEX:
            pipeline = self._complex_pipeline(intent)
        else:
            pipeline = self._standard_pipeline(intent)

        logger.info(
            "Routed query → complexity=%s, intent=%s, pipeline=%s",
            complexity.value,
            intent.value,
            pipeline["pipeline"],
        )
        return pipeline

    def _assess_complexity(self, query: str, intent: Intent) -> QueryComplexity:
        """Determine query complexity using heuristics."""
        query_lower = query.lower().strip()

        # Check for simple signals
        for pattern in SIMPLE_SIGNALS:
            if re.search(pattern, query_lower):
                return QueryComplexity.SIMPLE

        # Chat intents are always simple
        if intent == Intent.GENERAL_CHAT:
            return QueryComplexity.SIMPLE

        # Check for complex signals
        complex_score = 0
        for pattern in COMPLEX_SIGNALS:
            if re.search(pattern, query_lower):
                complex_score += 1

        # Multi-entity queries are complex
        if intent == Intent.COMPARISON:
            complex_score += 2

        # Long queries tend to be more complex
        word_count = len(query.split())
        if word_count > 20:
            complex_score += 1

        # Multiple question marks suggest multi-part queries
        if query.count("?") > 1:
            complex_score += 1

        if complex_score >= 2:
            return QueryComplexity.COMPLEX
        elif complex_score == 1:
            return QueryComplexity.STANDARD

        return QueryComplexity.STANDARD

    def _simple_pipeline(self, intent: Intent) -> dict:
        """Pipeline for simple queries: fast, minimal retrieval."""
        return {
            "complexity": QueryComplexity.SIMPLE,
            "intent": intent,
            "pipeline": "simple",
            "strategies": ["expand"],
            "retrieval_k": 3,
            "use_reranker": False,
            "use_query_transform": False,
            "reasoning": "Simple query — using lightweight retrieval for fast response",
        }

    def _standard_pipeline(self, intent: Intent) -> dict:
        """Pipeline for standard queries: balanced retrieval + generation."""
        strategies = ["expand"]

        if intent in (Intent.INVENTORY_SEARCH, Intent.PRICE_QUERY):
            strategies.append("expand")
        elif intent == Intent.SPECS_LOOKUP:
            strategies.append("step_back")

        return {
            "complexity": QueryComplexity.STANDARD,
            "intent": intent,
            "pipeline": "standard",
            "strategies": strategies,
            "retrieval_k": 5,
            "use_reranker": True,
            "use_query_transform": True,
            "reasoning": "Standard query — hybrid retrieval with re-ranking",
        }

    def _complex_pipeline(self, intent: Intent) -> dict:
        """Pipeline for complex queries: full agentic RAG."""
        strategies = ["multi_query", "expand"]

        if intent == Intent.COMPARISON:
            strategies.append("decompose")
        elif intent == Intent.RECOMMENDATION:
            strategies.extend(["hyde", "step_back"])

        return {
            "complexity": QueryComplexity.COMPLEX,
            "intent": intent,
            "pipeline": "agentic",
            "strategies": strategies,
            "retrieval_k": 10,
            "use_reranker": True,
            "use_query_transform": True,
            "reasoning": "Complex query — using agentic multi-step RAG with query decomposition",
        }

    def __repr__(self) -> str:
        return f"AdaptiveRouter(classifier={self.intent_classifier!r})"
