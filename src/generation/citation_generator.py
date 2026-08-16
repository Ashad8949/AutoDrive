"""
AutoDrive RAG v2.0 — Citation Generator
Adds inline source citations to generated answers, enabling users
to verify claims by tracing them back to source documents.

Output format:
  {
    "answer": "The Hyundai Creta is priced at ₹14.50 lakh [1]...",
    "citations": [
      {"id": 1, "source": "inventory/car_2", "text": "Price: ₹14,50,000", "confidence": 0.95}
    ]
  }
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("chatbot.generation.citations")

CITATION_PROMPT = """\
You are an AI assistant that ALWAYS cites sources. Given the context documents \
and a user question, generate a concise answer with inline citations.

Rules:
- Add [N] after each claim, where N references the source document number.
- Only make claims supported by the provided documents.
- If you cannot answer from the documents, say so honestly.

Context documents:
{documents}

Question: {query}

Answer with inline citations:"""


class CitationGenerator:
    """
    Generates answers with inline source citations.

    Ensures every factual claim in the response is traceable to a
    specific source document, building user trust and enabling
    verification.

    Attributes:
        llm: LLM for citation-aware generation.
    """

    def __init__(self, llm=None) -> None:
        self.llm = llm

    def generate_with_citations(
        self,
        query: str,
        documents: list[dict],
        chat_history: list = None,
    ) -> dict:
        """
        Generate an answer with inline citations.

        Args:
            query: User's question.
            documents: Retrieved documents (each must have 'text' and optional 'metadata').
            chat_history: Previous conversation messages.

        Returns:
            Dict with 'answer' (text with [N] citations) and 'citations' list.
        """
        if not documents:
            return {
                "answer": "I don't have enough information to answer this question.",
                "citations": [],
            }

        # Build numbered document context
        doc_lines = []
        doc_map = {}
        for i, doc in enumerate(documents[:10], start=1):
            source = doc.get("metadata", {}).get("source", f"doc_{i}")
            doc_id = doc.get("metadata", {}).get("parent_doc_id", source)
            doc_text = doc["text"][:600]
            doc_lines.append(f"[{i}] (Source: {doc_id})\n{doc_text}")
            doc_map[i] = {
                "source": doc_id,
                "text": doc_text,
                "metadata": doc.get("metadata", {}),
            }

        documents_text = "\n\n".join(doc_lines)

        if not self.llm:
            # Without LLM, return raw context with source labels
            return {
                "answer": documents_text,
                "citations": [
                    {"id": i, **info} for i, info in doc_map.items()
                ],
            }

        try:
            prompt = CITATION_PROMPT.format(
                documents=documents_text, query=query
            )
            response = self.llm.invoke(prompt)
            answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            # Extract which citations were actually used
            used_citations = self._extract_used_citations(answer, doc_map)

            return {
                "answer": answer,
                "citations": used_citations,
            }

        except Exception as e:
            logger.error("Citation generation failed: %s", e)
            return {
                "answer": "I encountered an error generating a response.",
                "citations": [],
            }

    def _extract_used_citations(
        self, answer: str, doc_map: dict
    ) -> list[dict]:
        """Extract citation references from the generated answer."""
        used = []
        citation_ids = set(
            int(m) for m in re.findall(r"\[(\d+)\]", answer)
        )

        for cid in sorted(citation_ids):
            if cid in doc_map:
                used.append({
                    "id": cid,
                    "source": doc_map[cid]["source"],
                    "text": doc_map[cid]["text"][:200],
                    "confidence": 0.9,
                })

        return used

    def add_citations_post_hoc(
        self,
        answer: str,
        documents: list[dict],
    ) -> dict:
        """
        Add citations to an already-generated answer by matching
        claims to source documents using text overlap.

        Args:
            answer: Generated answer without citations.
            documents: Source documents.

        Returns:
            Dict with 'answer' (with citations added) and 'citations'.
        """
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        citations = []
        cited_answer_parts = []

        for sentence in sentences:
            best_match = None
            best_overlap = 0

            sentence_words = set(sentence.lower().split())

            for i, doc in enumerate(documents, start=1):
                doc_words = set(doc["text"].lower().split())
                overlap = len(sentence_words & doc_words) / max(
                    len(sentence_words), 1
                )

                if overlap > best_overlap and overlap > 0.3:
                    best_overlap = overlap
                    best_match = i

            if best_match:
                cited_answer_parts.append(f"{sentence} [{best_match}]")
                source = documents[best_match - 1].get("metadata", {}).get(
                    "source", f"doc_{best_match}"
                )
                citations.append({
                    "id": best_match,
                    "source": source,
                    "text": documents[best_match - 1]["text"][:200],
                    "confidence": best_overlap,
                })
            else:
                cited_answer_parts.append(sentence)

        return {
            "answer": " ".join(cited_answer_parts),
            "citations": citations,
        }

    def __repr__(self) -> str:
        return f"CitationGenerator(has_llm={self.llm is not None})"
