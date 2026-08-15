"""
AutoDrive RAG v2.0 — Corrective RAG (CRAG)
Evaluates retrieval quality and takes corrective action when retrieved
documents are insufficient or irrelevant.

CRAG Pipeline:
  1. Retrieve documents normally
  2. Grade each document for relevance (LLM or heuristic)
  3. If relevant → proceed to generation
  4. If ambiguous → rewrite query and re-retrieve
  5. If irrelevant → fall back to web search

Reference: Yan et al. (2024) — "Corrective Retrieval Augmented Generation"
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger("chatbot.agents.crag")


class RelevanceGrade(str, Enum):
    """Document relevance grades."""

    RELEVANT = "relevant"
    AMBIGUOUS = "ambiguous"
    IRRELEVANT = "irrelevant"


# Prompt for LLM-based relevance grading
GRADE_PROMPT = """\
You are a relevance grader. Given a user question and a retrieved document, \
determine if the document contains information relevant to answering the question.

Grade the document as:
- "relevant" — The document directly helps answer the question
- "ambiguous" — The document is somewhat related but may not directly answer
- "irrelevant" — The document is not useful for answering the question

User question: {query}

Retrieved document:
{document}

Grade (respond with exactly one word: relevant, ambiguous, or irrelevant):"""

REWRITE_PROMPT = """\
You are a query rewriting expert. The original query did not retrieve \
good results. Rewrite the query to be more specific and likely to find \
relevant information in a car dealership's inventory.

Original query: {query}

Reason for rewriting: Retrieved documents were not relevant.

Rewritten query (one line only):"""


class CorrectiveRAG:
    """
    Corrective RAG — evaluates and corrects retrieval quality.

    Implements a retrieval evaluation loop that:
      1. Grades retrieved documents for relevance
      2. Rewrites queries when results are ambiguous
      3. Falls back to web search when local knowledge fails
      4. Limits correction attempts to avoid infinite loops

    Attributes:
        llm: LLM for relevance grading and query rewriting.
        retriever: The retrieval function/object to call.
        relevance_threshold: Min fraction of relevant docs to proceed.
        max_retries: Maximum number of query rewrite attempts.
    """

    def __init__(
        self,
        llm=None,
        relevance_threshold: float = 0.3,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize Corrective RAG.

        Args:
            llm: LangChain-compatible LLM for grading/rewriting.
            relevance_threshold: Min relevant doc ratio (0-1) to proceed.
            max_retries: Max query rewrite attempts before fallback.
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.max_retries = max_retries

    def grade_document(self, query: str, document: str) -> RelevanceGrade:
        """
        Grade a single document's relevance to the query.

        Args:
            query: User's question.
            document: Retrieved document text.

        Returns:
            RelevanceGrade enum value.
        """
        if self.llm:
            return self._grade_llm(query, document)
        return self._grade_heuristic(query, document)

    def grade_documents(
        self, query: str, documents: list[dict]
    ) -> list[dict]:
        """
        Grade all retrieved documents for relevance.

        Args:
            query: User's question.
            documents: List of retrieval result dicts (must have 'text').

        Returns:
            Same documents with added 'relevance_grade' field.
        """
        graded = []
        for doc in documents:
            grade = self.grade_document(query, doc["text"])
            graded.append({**doc, "relevance_grade": grade.value})

        relevant_count = sum(
            1 for d in graded if d["relevance_grade"] == RelevanceGrade.RELEVANT.value
        )
        logger.info(
            "CRAG grading: %d/%d documents relevant (threshold=%.0f%%)",
            relevant_count,
            len(graded),
            self.relevance_threshold * 100,
        )
        return graded

    def evaluate_retrieval(
        self, query: str, documents: list[dict]
    ) -> dict:
        """
        Evaluate retrieval quality and decide corrective action.

        Args:
            query: User's question.
            documents: Retrieved documents.

        Returns:
            Dict with:
              - 'action': 'proceed' | 'rewrite' | 'web_search'
              - 'graded_documents': Documents with grades
              - 'relevant_documents': Only the relevant ones
              - 'relevance_ratio': Fraction of relevant docs
              - 'rewritten_query': New query if action is 'rewrite'
        """
        graded = self.grade_documents(query, documents)

        relevant = [
            d for d in graded
            if d["relevance_grade"] == RelevanceGrade.RELEVANT.value
        ]
        ambiguous = [
            d for d in graded
            if d["relevance_grade"] == RelevanceGrade.AMBIGUOUS.value
        ]

        total = len(graded)
        relevance_ratio = len(relevant) / total if total > 0 else 0.0

        result = {
            "graded_documents": graded,
            "relevant_documents": relevant + ambiguous,  # Include ambiguous as fallback
            "relevance_ratio": relevance_ratio,
            "rewritten_query": None,
        }

        if relevance_ratio >= self.relevance_threshold:
            result["action"] = "proceed"
            logger.info("CRAG: Proceeding with %d relevant documents", len(relevant))
        elif relevant or ambiguous:
            # Some useful docs — try rewriting for better results
            result["action"] = "rewrite"
            result["rewritten_query"] = self.rewrite_query(query)
            logger.info("CRAG: Rewriting query for better results")
        else:
            result["action"] = "web_search"
            logger.info("CRAG: All documents irrelevant — falling back to web search")

        return result

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query that produced poor retrieval results.

        Args:
            query: The original underperforming query.

        Returns:
            A rewritten, more specific query.
        """
        if not self.llm:
            # Simple heuristic rewrite: add context
            return f"car dealership inventory {query}"

        try:
            prompt = REWRITE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            rewritten = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            rewritten = rewritten.strip().split("\n")[0]
            logger.debug("CRAG rewrite: '%s' → '%s'", query, rewritten)
            return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
            return query

    def web_search_fallback(self, query: str) -> list[dict]:
        """
        Perform web search as a fallback when local retrieval fails.

        Uses DuckDuckGo for free web search. Returns results formatted
        as retrieval documents.

        Args:
            query: The search query.

        Returns:
            List of search result dicts with 'text', 'score', 'metadata'.
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            web_docs = []
            for i, r in enumerate(results):
                web_docs.append({
                    "text": f"{r.get('title', '')}\n{r.get('body', '')}",
                    "score": 1.0 - (i * 0.1),
                    "rank": i + 1,
                    "metadata": {
                        "source": "web_search",
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                    },
                    "source": "web_search",
                })

            logger.info("CRAG web search: %d results for '%s'", len(web_docs), query[:50])
            return web_docs

        except ImportError:
            logger.warning("duckduckgo-search not installed — web fallback unavailable")
            return []
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return []

    def _grade_llm(self, query: str, document: str) -> RelevanceGrade:
        """Grade relevance using LLM."""
        try:
            # Truncate long documents
            doc_truncated = document[:2000]
            prompt = GRADE_PROMPT.format(query=query, document=doc_truncated)
            response = self.llm.invoke(prompt)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip().lower()

            for grade in RelevanceGrade:
                if grade.value in text:
                    return grade
            return RelevanceGrade.AMBIGUOUS

        except Exception as e:
            logger.warning("LLM grading failed: %s — using heuristic", e)
            return self._grade_heuristic(query, document)

    @staticmethod
    def _grade_heuristic(query: str, document: str) -> RelevanceGrade:
        """Grade relevance using keyword overlap heuristic."""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        # Remove stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not", "with", "this",
            "that", "it", "i", "me", "my", "you", "your", "we", "they",
            "what", "which", "who", "how", "show", "find", "get",
        }
        query_keywords = query_words - stop_words
        if not query_keywords:
            return RelevanceGrade.AMBIGUOUS

        overlap = query_keywords & doc_words
        overlap_ratio = len(overlap) / len(query_keywords)

        if overlap_ratio >= 0.5:
            return RelevanceGrade.RELEVANT
        elif overlap_ratio >= 0.2:
            return RelevanceGrade.AMBIGUOUS
        else:
            return RelevanceGrade.IRRELEVANT

    def __repr__(self) -> str:
        return (
            f"CorrectiveRAG(threshold={self.relevance_threshold}, "
            f"max_retries={self.max_retries}, has_llm={self.llm is not None})"
        )
