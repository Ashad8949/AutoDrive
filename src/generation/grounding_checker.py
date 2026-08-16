"""
AutoDrive RAG v2.0 — Grounding Checker
Verifies that generated answers are grounded in (supported by) the
retrieved source documents. Detects hallucinations.

A "grounded" answer only makes claims that can be verified in the
source documents. An "ungrounded" answer contains fabricated facts.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("chatbot.generation.grounding")

GROUNDING_PROMPT = """\
You are a hallucination detector. Given source documents and a generated \
answer, identify any claims in the answer that are NOT supported by the \
source documents.

Source documents:
{documents}

Generated answer:
{answer}

List ONLY the unsupported/hallucinated claims, one per line. \
If the answer is fully grounded, respond with "FULLY_GROUNDED".

Unsupported claims:"""


class GroundingChecker:
    """
    Checks if generated answers are grounded in source documents.

    Provides both LLM-based and heuristic methods for hallucination
    detection. Returns a groundedness score and list of unsupported claims.

    Attributes:
        llm: LLM for intelligent grounding checks.
        threshold: Minimum groundedness score to pass (0-1).
    """

    def __init__(self, llm=None, threshold: float = 0.7) -> None:
        self.llm = llm
        self.threshold = threshold

    def check(
        self,
        answer: str,
        documents: list[dict],
    ) -> dict:
        """
        Check if the answer is grounded in the documents.

        Args:
            answer: Generated answer text.
            documents: Source documents used for generation.

        Returns:
            Dict with:
              - 'is_grounded': bool
              - 'score': float (0-1)
              - 'unsupported_claims': list of ungrounded statements
              - 'method': 'llm' or 'heuristic'
        """
        if self.llm:
            return self._check_llm(answer, documents)
        return self._check_heuristic(answer, documents)

    def _check_llm(self, answer: str, documents: list[dict]) -> dict:
        """LLM-based grounding check."""
        try:
            docs_text = "\n---\n".join(d["text"][:500] for d in documents[:5])
            prompt = GROUNDING_PROMPT.format(
                documents=docs_text, answer=answer[:1000]
            )

            response = self.llm.invoke(prompt)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()

            if "FULLY_GROUNDED" in text.upper():
                return {
                    "is_grounded": True,
                    "score": 1.0,
                    "unsupported_claims": [],
                    "method": "llm",
                }

            claims = [
                line.strip().lstrip("- •")
                for line in text.split("\n")
                if line.strip() and len(line.strip()) > 5
            ]

            # Estimate score based on number of unsupported claims
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            total = max(len(sentences), 1)
            unsupported = len(claims)
            score = max(0, 1.0 - (unsupported / total))

            return {
                "is_grounded": score >= self.threshold,
                "score": score,
                "unsupported_claims": claims,
                "method": "llm",
            }

        except Exception as e:
            logger.warning("LLM grounding check failed: %s", e)
            return self._check_heuristic(answer, documents)

    def _check_heuristic(self, answer: str, documents: list[dict]) -> dict:
        """
        Heuristic grounding check using token overlap.

        Checks what fraction of content words in the answer appear
        in the source documents.
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "in", "on", "at", "to", "for", "of", "and", "or", "but",
            "not", "with", "this", "that", "it", "i", "you", "we", "they",
            "me", "my", "your", "our", "their", "its", "very", "so",
            "also", "just", "than", "then", "if", "when", "while",
        }

        answer_words = set(
            w.lower()
            for w in re.findall(r"\b\w+\b", answer)
            if w.lower() not in stop_words and len(w) > 2
        )

        doc_words = set()
        for doc in documents:
            doc_words.update(
                w.lower()
                for w in re.findall(r"\b\w+\b", doc["text"])
                if len(w) > 2
            )

        if not answer_words:
            return {
                "is_grounded": True,
                "score": 1.0,
                "unsupported_claims": [],
                "method": "heuristic",
            }

        grounded_words = answer_words & doc_words
        score = len(grounded_words) / len(answer_words)

        # Find sentences with low overlap
        unsupported = []
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            sent_words = set(
                w.lower()
                for w in re.findall(r"\b\w+\b", sentence)
                if w.lower() not in stop_words and len(w) > 2
            )
            if sent_words:
                sent_overlap = len(sent_words & doc_words) / len(sent_words)
                if sent_overlap < 0.3:
                    unsupported.append(sentence)

        return {
            "is_grounded": score >= self.threshold,
            "score": score,
            "unsupported_claims": unsupported,
            "method": "heuristic",
        }

    def __repr__(self) -> str:
        return (
            f"GroundingChecker(threshold={self.threshold}, "
            f"has_llm={self.llm is not None})"
        )
