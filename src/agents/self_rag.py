"""
AutoDrive RAG v2.0 — Self-RAG (Self-Reflective RAG)
The LLM self-critiques its own retrieval decisions and generated output,
detecting and correcting hallucinations through reflection tokens.

Self-RAG Reflection Points:
  1. RETRIEVE — Should I retrieve documents for this query at all?
  2. RELEVANT — Are the retrieved documents relevant?
  3. SUPPORTED — Is each sentence in my answer supported by the evidence?
  4. USEFUL — Is the final answer actually useful to the user?

Reference: Asai et al. (2023) — "Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("chatbot.agents.self_rag")


class ReflectionType(str, Enum):
    """Types of self-reflection checks."""

    RETRIEVE = "retrieve"
    RELEVANT = "relevant"
    SUPPORTED = "supported"
    USEFUL = "useful"


@dataclass
class ReflectionResult:
    """Result of a self-reflection check."""

    reflection_type: ReflectionType
    passed: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    action: str  # What to do: 'proceed', 'retry', 'regenerate', 'skip_retrieval'


# ── Reflection Prompts ──────────────────────────────────────────────

NEED_RETRIEVAL_PROMPT = """\
Given the following user query, determine if you need to retrieve external \
documents to answer it accurately, or if you can answer from general knowledge.

Query: "{query}"

Respond with exactly one word:
- "yes" — if you need specific factual information from documents
- "no" — if this is a greeting, general knowledge, or conversational query

Answer:"""

RELEVANCE_CHECK_PROMPT = """\
Given the user's question and a set of retrieved documents, evaluate whether \
the documents contain information relevant to answering the question.

Question: "{query}"

Retrieved documents:
{documents}

Rate relevance from 0 to 10 (0 = completely irrelevant, 10 = perfectly relevant).
Respond with a JSON object: {{"score": <number>, "reasoning": "<brief explanation>"}}

JSON:"""

SUPPORT_CHECK_PROMPT = """\
Given the generated answer and the source documents, check if the answer \
is fully supported by the evidence in the documents.

Source documents:
{documents}

Generated answer:
{answer}

For each claim in the answer, verify it against the source documents.
Rate support from 0 to 10 (0 = hallucinated, 10 = fully supported).
Respond with a JSON object: {{"score": <number>, "unsupported_claims": ["claim1", ...], "reasoning": "<brief explanation>"}}

JSON:"""

USEFULNESS_CHECK_PROMPT = """\
Given the user's question and the generated answer, rate how useful \
and complete the answer is.

Question: "{query}"
Answer: "{answer}"

Rate usefulness from 0 to 10 (0 = useless, 10 = perfect answer).
Respond with a JSON object: {{"score": <number>, "reasoning": "<brief explanation>"}}

JSON:"""


class SelfRAG:
    """
    Self-Reflective RAG — LLM critiques its own retrieval and generation.

    Implements 4 reflection checks that can be used at different
    stages of the RAG pipeline to improve answer quality and reduce
    hallucinations.

    Attributes:
        llm: LLM for performing reflection checks.
        retrieval_threshold: Min score to proceed without retrieval.
        relevance_threshold: Min relevance score to accept documents.
        support_threshold: Min support score to accept an answer.
        usefulness_threshold: Min usefulness score to return an answer.
        max_regenerations: Max times to regenerate before giving up.
    """

    def __init__(
        self,
        llm=None,
        retrieval_threshold: float = 0.5,
        relevance_threshold: float = 0.5,
        support_threshold: float = 0.6,
        usefulness_threshold: float = 0.5,
        max_regenerations: int = 2,
    ) -> None:
        self.llm = llm
        self.retrieval_threshold = retrieval_threshold
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold
        self.usefulness_threshold = usefulness_threshold
        self.max_regenerations = max_regenerations

    def should_retrieve(self, query: str) -> ReflectionResult:
        """
        Decide if this query needs document retrieval.

        Some queries (greetings, general knowledge) don't need retrieval.
        Skipping retrieval for these saves latency and cost.

        Args:
            query: User's query.

        Returns:
            ReflectionResult with action 'proceed' or 'skip_retrieval'.
        """
        if not self.llm:
            return ReflectionResult(
                reflection_type=ReflectionType.RETRIEVE,
                passed=True,
                score=1.0,
                reasoning="No LLM available — defaulting to retrieve",
                action="proceed",
            )

        try:
            prompt = NEED_RETRIEVAL_PROMPT.format(query=query)
            response = self._invoke_llm(prompt)
            needs_retrieval = "yes" in response.lower()

            return ReflectionResult(
                reflection_type=ReflectionType.RETRIEVE,
                passed=needs_retrieval,
                score=1.0 if needs_retrieval else 0.0,
                reasoning=f"Query {'needs' if needs_retrieval else 'does not need'} retrieval",
                action="proceed" if needs_retrieval else "skip_retrieval",
            )
        except Exception as e:
            logger.warning("Retrieval check failed: %s", e)
            return ReflectionResult(
                reflection_type=ReflectionType.RETRIEVE,
                passed=True,
                score=1.0,
                reasoning=f"Check failed ({e}) — defaulting to retrieve",
                action="proceed",
            )

    def check_relevance(
        self, query: str, documents: list[dict]
    ) -> ReflectionResult:
        """
        Check if retrieved documents are relevant to the query.

        Args:
            query: User's query.
            documents: Retrieved documents (list of dicts with 'text').

        Returns:
            ReflectionResult with relevance assessment.
        """
        if not self.llm or not documents:
            return ReflectionResult(
                reflection_type=ReflectionType.RELEVANT,
                passed=True,
                score=0.7,
                reasoning="No LLM or no documents — skipping relevance check",
                action="proceed",
            )

        try:
            docs_text = "\n---\n".join(
                d["text"][:500] for d in documents[:5]
            )
            prompt = RELEVANCE_CHECK_PROMPT.format(
                query=query, documents=docs_text
            )
            response = self._invoke_llm(prompt)
            parsed = self._parse_json_response(response)

            score = parsed.get("score", 5) / 10.0
            passed = score >= self.relevance_threshold

            return ReflectionResult(
                reflection_type=ReflectionType.RELEVANT,
                passed=passed,
                score=score,
                reasoning=parsed.get("reasoning", ""),
                action="proceed" if passed else "retry",
            )
        except Exception as e:
            logger.warning("Relevance check failed: %s", e)
            return ReflectionResult(
                reflection_type=ReflectionType.RELEVANT,
                passed=True,
                score=0.5,
                reasoning=f"Check failed ({e}) — proceeding anyway",
                action="proceed",
            )

    def check_support(
        self, answer: str, documents: list[dict]
    ) -> ReflectionResult:
        """
        Check if the generated answer is supported by the evidence.

        This is the critical hallucination detection step. Each claim
        in the answer is verified against the source documents.

        Args:
            answer: Generated answer text.
            documents: Source documents used for generation.

        Returns:
            ReflectionResult with support assessment and unsupported claims.
        """
        if not self.llm:
            return ReflectionResult(
                reflection_type=ReflectionType.SUPPORTED,
                passed=True,
                score=0.7,
                reasoning="No LLM — skipping support check",
                action="proceed",
            )

        try:
            docs_text = "\n---\n".join(
                d["text"][:500] for d in documents[:5]
            )
            prompt = SUPPORT_CHECK_PROMPT.format(
                documents=docs_text, answer=answer
            )
            response = self._invoke_llm(prompt)
            parsed = self._parse_json_response(response)

            score = parsed.get("score", 5) / 10.0
            passed = score >= self.support_threshold
            unsupported = parsed.get("unsupported_claims", [])

            action = "proceed" if passed else "regenerate"
            if unsupported:
                logger.warning(
                    "Self-RAG: %d unsupported claims detected", len(unsupported)
                )

            return ReflectionResult(
                reflection_type=ReflectionType.SUPPORTED,
                passed=passed,
                score=score,
                reasoning=parsed.get("reasoning", f"Unsupported claims: {unsupported}"),
                action=action,
            )
        except Exception as e:
            logger.warning("Support check failed: %s", e)
            return ReflectionResult(
                reflection_type=ReflectionType.SUPPORTED,
                passed=True,
                score=0.5,
                reasoning=f"Check failed ({e}) — proceeding",
                action="proceed",
            )

    def check_usefulness(self, query: str, answer: str) -> ReflectionResult:
        """
        Check if the answer is actually useful to the user.

        Args:
            query: Original user query.
            answer: Generated answer.

        Returns:
            ReflectionResult with usefulness assessment.
        """
        if not self.llm:
            return ReflectionResult(
                reflection_type=ReflectionType.USEFUL,
                passed=True,
                score=0.7,
                reasoning="No LLM — skipping usefulness check",
                action="proceed",
            )

        try:
            prompt = USEFULNESS_CHECK_PROMPT.format(
                query=query, answer=answer[:1000]
            )
            response = self._invoke_llm(prompt)
            parsed = self._parse_json_response(response)

            score = parsed.get("score", 5) / 10.0
            passed = score >= self.usefulness_threshold

            return ReflectionResult(
                reflection_type=ReflectionType.USEFUL,
                passed=passed,
                score=score,
                reasoning=parsed.get("reasoning", ""),
                action="proceed" if passed else "regenerate",
            )
        except Exception as e:
            logger.warning("Usefulness check failed: %s", e)
            return ReflectionResult(
                reflection_type=ReflectionType.USEFUL,
                passed=True,
                score=0.5,
                reasoning=f"Check failed ({e})",
                action="proceed",
            )

    def full_reflection(
        self,
        query: str,
        documents: list[dict],
        answer: str,
    ) -> dict:
        """
        Run all 4 reflection checks and return a comprehensive report.

        Args:
            query: User's query.
            documents: Retrieved documents.
            answer: Generated answer.

        Returns:
            Dict with all reflection results and overall assessment.
        """
        relevance = self.check_relevance(query, documents)
        support = self.check_support(answer, documents)
        usefulness = self.check_usefulness(query, answer)

        overall_score = (
            relevance.score * 0.3
            + support.score * 0.4
            + usefulness.score * 0.3
        )

        overall_passed = all([
            relevance.passed,
            support.passed,
            usefulness.passed,
        ])

        # Determine action based on which check failed
        if not support.passed:
            action = "regenerate"
        elif not relevance.passed:
            action = "retry"
        elif not usefulness.passed:
            action = "regenerate"
        else:
            action = "proceed"

        return {
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "action": action,
            "reflections": {
                "relevance": {
                    "passed": relevance.passed,
                    "score": relevance.score,
                    "reasoning": relevance.reasoning,
                },
                "support": {
                    "passed": support.passed,
                    "score": support.score,
                    "reasoning": support.reasoning,
                },
                "usefulness": {
                    "passed": usefulness.passed,
                    "score": usefulness.score,
                    "reasoning": usefulness.reasoning,
                },
            },
        }

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM and return text response."""
        response = self.llm.invoke(prompt)
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        ).strip()

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Best-effort JSON parsing from LLM response."""
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract score
        score_match = re.search(r"(\d+(?:\.\d+)?)", text)
        score = float(score_match.group(1)) if score_match else 5.0

        return {"score": min(score, 10), "reasoning": text[:200]}

    def __repr__(self) -> str:
        return (
            f"SelfRAG(support_threshold={self.support_threshold}, "
            f"has_llm={self.llm is not None})"
        )
