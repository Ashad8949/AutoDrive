"""
AutoDrive RAG v2.0 — Query Transformer
Implements advanced query transformation techniques to improve retrieval quality.

Techniques:
  1. HyDE (Hypothetical Document Embedding) — Generate a hypothetical answer,
     embed it, and use it for retrieval instead of the raw query.
  2. Multi-Query — Generate multiple query variations to capture different
     aspects of the user's intent, retrieve for each, merge results.
  3. Step-back Prompting — Generate a more abstract/general question to
     retrieve broader context before answering the specific question.
  4. Query Expansion — Add synonyms and related terms to the query.
  5. Query Decomposition — Break complex multi-part questions into sub-queries.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("chatbot.query.transformer")


# ── Prompt Templates ────────────────────────────────────────────────

HYDE_PROMPT = """\
You are a knowledgeable automotive expert. Please write a brief, \
factual passage (3-5 sentences) that would answer the following question \
about cars from a dealership inventory. Write as if you're describing \
a specific car listing.

Question: {query}

Passage:"""

MULTI_QUERY_PROMPT = """\
You are an AI assistant helping to improve search results. Given the \
user's question, generate {n} different versions of the same question \
that capture different aspects or phrasings. Each version should help \
retrieve relevant information from a car dealership's inventory.

Original question: {query}

Generate exactly {n} alternative questions, one per line. Do not number them.
Alternative questions:"""

STEPBACK_PROMPT = """\
You are an expert at generating abstract, higher-level questions. Given \
a specific question, generate a more general "step-back" question that \
would help retrieve broader background information useful for answering \
the original question.

Original question: {query}

Step-back question (one line only):"""

DECOMPOSE_PROMPT = """\
You are an expert at breaking down complex questions. Given a complex \
question, decompose it into 2-4 simpler sub-questions that, when \
answered together, would fully answer the original question.

Complex question: {query}

Sub-questions (one per line, no numbering):"""

EXPANSION_SYNONYMS = {
    "suv": ["suv", "sport utility vehicle", "crossover", "utility vehicle"],
    "sedan": ["sedan", "saloon", "four-door"],
    "hatchback": ["hatchback", "hatch", "hot hatch"],
    "ev": ["electric vehicle", "ev", "battery electric", "electric car", "bev"],
    "automatic": ["automatic", "auto", "at", "cvt", "dct", "dsg", "amt"],
    "manual": ["manual", "mt", "stick shift"],
    "petrol": ["petrol", "gasoline", "gas"],
    "diesel": ["diesel", "turbo diesel"],
    "cheap": ["affordable", "budget", "low cost", "economical", "value"],
    "expensive": ["premium", "luxury", "high-end"],
    "family": ["family", "spacious", "7 seater", "7-seater", "large"],
    "mileage": ["mileage", "fuel efficiency", "kmpl", "fuel economy"],
    "safe": ["safe", "safety", "adas", "airbags", "ncap"],
}


class QueryTransformer:
    """
    Transforms user queries to improve retrieval quality.

    Supports 5 transformation techniques that can be used individually
    or combined for maximum retrieval recall.

    Attributes:
        llm: LangChain-compatible LLM for generating transformations.
    """

    def __init__(self, llm=None) -> None:
        """
        Initialize the query transformer.

        Args:
            llm: LangChain-compatible LLM instance.
                 Required for HyDE, multi-query, step-back, decomposition.
                 Query expansion works without an LLM.
        """
        self.llm = llm

    def hyde(self, query: str) -> str:
        """
        HyDE: Hypothetical Document Embedding.

        Generates a hypothetical answer to the query, which is then
        used as the search query instead. The hypothesis is closer
        to the target document in embedding space than the raw question.

        Args:
            query: Original user query.

        Returns:
            Hypothetical document text (use this for retrieval).
        """
        if not self.llm:
            logger.warning("No LLM configured — returning original query for HyDE")
            return query

        try:
            prompt = HYDE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            hypothesis = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            logger.debug("HyDE hypothesis: %s", hypothesis[:100])
            return hypothesis.strip()
        except Exception as e:
            logger.warning("HyDE generation failed: %s — using original query", e)
            return query

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """
        Multi-Query: Generate multiple query variations.

        Creates n different phrasings of the same question to capture
        different aspects of the user's intent. Results from all
        variations are merged for higher recall.

        Args:
            query: Original user query.
            n: Number of variations to generate.

        Returns:
            List of query variations (includes original).
        """
        if not self.llm:
            logger.warning("No LLM — returning only expanded query")
            return [self.expand_query(query)]

        try:
            prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
            response = self.llm.invoke(prompt)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            variations = [
                line.strip()
                for line in text.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            # Always include the original query
            all_queries = [query] + variations[:n]
            logger.debug("Multi-query: %d variations generated", len(all_queries))
            return all_queries

        except Exception as e:
            logger.warning("Multi-query generation failed: %s", e)
            return [query]

    def step_back(self, query: str) -> str:
        """
        Step-back Prompting: Generate a more abstract question.

        The abstract question retrieves broader background information
        that helps contextualize the specific answer.

        Example:
          Query: "What's the mileage of the 2023 Hyundai Creta?"
          Step-back: "What are the specifications of the Hyundai Creta?"

        Args:
            query: Original specific query.

        Returns:
            A more general/abstract version of the query.
        """
        if not self.llm:
            return query

        try:
            prompt = STEPBACK_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            stepback = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            stepback = stepback.strip().split("\n")[0]  # Take first line only
            logger.debug("Step-back query: %s", stepback)
            return stepback

        except Exception as e:
            logger.warning("Step-back generation failed: %s", e)
            return query

    def decompose(self, query: str) -> list[str]:
        """
        Query Decomposition: Break complex query into sub-queries.

        Example:
          Query: "Compare the safety features and pricing of the Creta and Seltos"
          Sub-queries:
            - "What are the safety features of the Hyundai Creta?"
            - "What are the safety features of the Kia Seltos?"
            - "What is the price of the Hyundai Creta?"
            - "What is the price of the Kia Seltos?"

        Args:
            query: Complex multi-part query.

        Returns:
            List of simpler sub-queries.
        """
        if not self.llm:
            return [query]

        try:
            prompt = DECOMPOSE_PROMPT.format(query=query)
            response = self.llm.invoke(prompt)
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in text.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            logger.debug("Decomposed into %d sub-queries", len(sub_queries))
            return sub_queries if sub_queries else [query]

        except Exception as e:
            logger.warning("Query decomposition failed: %s", e)
            return [query]

    def expand_query(self, query: str) -> str:
        """
        Query Expansion: Add synonyms and related terms.

        This is a lightweight, LLM-free technique that adds relevant
        synonyms to improve keyword matching in sparse retrieval.

        Args:
            query: Original user query.

        Returns:
            Expanded query with additional relevant terms.
        """
        query_lower = query.lower()
        expansions = []

        for keyword, synonyms in EXPANSION_SYNONYMS.items():
            if keyword in query_lower:
                # Add synonyms that aren't already in the query
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        expansions.append(syn)

        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            logger.debug("Expanded query: %s", expanded[:100])
            return expanded

        return query

    def transform(
        self,
        query: str,
        strategies: Optional[list[str]] = None,
    ) -> dict:
        """
        Apply multiple transformation strategies to a query.

        Args:
            query: Original user query.
            strategies: List of strategies to apply.
                       Options: 'hyde', 'multi_query', 'step_back',
                                'decompose', 'expand'.
                       Default: ['expand', 'multi_query'].

        Returns:
            Dict with results from each strategy:
              {
                'original': str,
                'expanded': str,
                'hyde': str,
                'multi_query': list[str],
                'step_back': str,
                'sub_queries': list[str],
                'all_queries': list[str],  # merged unique queries for retrieval
              }
        """
        strategies = strategies or ["expand", "multi_query"]
        result = {"original": query, "all_queries": [query]}

        if "expand" in strategies:
            result["expanded"] = self.expand_query(query)
            if result["expanded"] != query:
                result["all_queries"].append(result["expanded"])

        if "hyde" in strategies:
            result["hyde"] = self.hyde(query)
            result["all_queries"].append(result["hyde"])

        if "multi_query" in strategies:
            result["multi_query"] = self.multi_query(query)
            result["all_queries"].extend(result["multi_query"])

        if "step_back" in strategies:
            result["step_back"] = self.step_back(query)
            result["all_queries"].append(result["step_back"])

        if "decompose" in strategies:
            result["sub_queries"] = self.decompose(query)
            result["all_queries"].extend(result["sub_queries"])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in result["all_queries"]:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        result["all_queries"] = unique

        return result

    def __repr__(self) -> str:
        has_llm = self.llm is not None
        return f"QueryTransformer(has_llm={has_llm})"
