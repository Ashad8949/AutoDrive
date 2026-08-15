"""
AutoDrive RAG v2.0 — Contextual Chunker (Anthropic-style)
Prepends document-level context to each chunk before embedding,
dramatically reducing retrieval failures.

Standard chunking produces isolated chunks that lose document context:
  "The price starts at ₹14.50 lakh" — Which car? What document?

Contextual chunking enriches each chunk with surrounding context:
  "This chunk is from the Hyundai Creta product page. The document
   covers specifications, pricing, and features of the 2023 Creta SX(O).
   Content: The price starts at ₹14.50 lakh"

Reference: Anthropic (2024) — "Introducing Contextual Retrieval"
Claims up to 49% reduction in retrieval failures.
"""

from __future__ import annotations

import logging
from typing import Optional

from .chunker import Chunk, DocumentChunker

logger = logging.getLogger("chatbot.chunking.contextual")

# Template for contextual header generation
CONTEXT_PROMPT = """\
Here is the full document:
<document>
{document}
</document>

Here is the chunk we want to situate within the document:
<chunk>
{chunk}
</chunk>

Please give a short, succinct context (2-3 sentences max) to situate this \
chunk within the overall document. Focus on what specific topic or entity \
this chunk is about. Respond with ONLY the context, nothing else."""


class ContextualChunker:
    """
    Enriches chunks with document-level context before embedding.

    For each chunk, generates a brief context header using either:
      1. LLM-generated context (high quality, uses API calls)
      2. Rule-based context (free, uses document metadata)

    The contextual header is prepended to the chunk text before
    embedding, so the embedding captures both local and global meaning.

    Attributes:
        base_chunker: The underlying chunker for initial splitting.
        use_llm: Whether to use LLM for context generation.
    """

    def __init__(
        self,
        base_chunker: Optional[DocumentChunker] = None,
        use_llm: bool = False,
        llm=None,
    ) -> None:
        """
        Initialize the contextual chunker.

        Args:
            base_chunker: DocumentChunker for initial splitting.
                          Defaults to recursive chunking.
            use_llm: If True, use LLM to generate context headers.
                     If False, use rule-based context from metadata.
            llm: LangChain-compatible LLM instance (required if use_llm=True).
        """
        self.base_chunker = base_chunker or DocumentChunker(
            strategy="recursive",
            chunk_size=800,
            chunk_overlap=150,
        )
        self.use_llm = use_llm
        self.llm = llm

    def chunk_with_context(
        self,
        text: str,
        doc_id: str = "",
        doc_title: str = "",
        doc_summary: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Chunk a document and enrich each chunk with contextual headers.

        Args:
            text: Full document text.
            doc_id: Document identifier.
            doc_title: Title of the document (used for context).
            doc_summary: Summary of the document (used for context).
            metadata: Additional metadata for each chunk.

        Returns:
            List of enriched Chunk objects with contextual text.
        """
        base_metadata = metadata or {}

        # Step 1: Chunk the document
        chunks = self.base_chunker.chunk_document(
            text=text, doc_id=doc_id, metadata=base_metadata
        )

        if not chunks:
            return chunks

        # Step 2: Generate context for each chunk
        enriched_chunks = []
        for chunk in chunks:
            if self.use_llm and self.llm:
                context = self._generate_llm_context(text, chunk.text)
            else:
                context = self._generate_rule_context(
                    chunk=chunk,
                    doc_title=doc_title,
                    doc_summary=doc_summary,
                    total_chunks=len(chunks),
                )

            # Step 3: Prepend context to chunk text
            enriched_text = f"{context}\n\n{chunk.text}"

            enriched_chunks.append(
                Chunk(
                    text=enriched_text,
                    metadata={
                        **chunk.metadata,
                        "original_text": chunk.text,
                        "context_header": context,
                        "strategy": "contextual",
                    },
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    parent_doc_id=doc_id,
                    strategy="contextual",
                )
            )

        logger.info(
            "Contextual chunking: %d chunks enriched for document '%s'",
            len(enriched_chunks),
            doc_id[:30] or doc_title[:30],
        )
        return enriched_chunks

    def _generate_llm_context(self, full_doc: str, chunk_text: str) -> str:
        """
        Generate context using an LLM (high quality).

        Sends the full document + target chunk to the LLM and asks
        for a brief situating context.
        """
        try:
            # Truncate document if too long for LLM context window
            max_doc_len = 8000
            truncated_doc = full_doc[:max_doc_len]
            if len(full_doc) > max_doc_len:
                truncated_doc += "\n... [truncated]"

            prompt = CONTEXT_PROMPT.format(
                document=truncated_doc,
                chunk=chunk_text,
            )

            response = self.llm.invoke(prompt)
            context = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            return context.strip()

        except Exception as e:
            logger.warning("LLM context generation failed: %s", e)
            return self._generate_rule_context(
                chunk=Chunk(text=chunk_text),
                doc_title="",
                doc_summary="",
                total_chunks=1,
            )

    @staticmethod
    def _generate_rule_context(
        chunk: Chunk,
        doc_title: str = "",
        doc_summary: str = "",
        total_chunks: int = 1,
    ) -> str:
        """
        Generate context using rules and metadata (free, no LLM needed).

        Constructs a contextual header from available metadata fields.
        """
        parts = []

        if doc_title:
            parts.append(f"Document: {doc_title}")

        if doc_summary:
            # Truncate summary to keep header concise
            summary = doc_summary[:200]
            parts.append(f"Summary: {summary}")

        if chunk.parent_doc_id:
            parts.append(f"Source: {chunk.parent_doc_id}")

        if total_chunks > 1:
            parts.append(
                f"Section {chunk.chunk_index + 1} of {total_chunks}"
            )

        # Extract any entity info from metadata
        meta = chunk.metadata
        if meta.get("make"):
            parts.append(f"Car: {meta.get('make')} {meta.get('model', '')}")
        if meta.get("location"):
            parts.append(f"Location: {meta['location']}")

        if not parts:
            return "Context: General document content"

        return " | ".join(parts)

    def chunk_documents_with_context(
        self,
        documents: list[dict],
    ) -> list[Chunk]:
        """
        Contextually chunk multiple documents.

        Args:
            documents: List of dicts with 'text', 'doc_id',
                       optional 'title', 'summary', 'metadata'.

        Returns:
            Flat list of all enriched chunks from all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_with_context(
                text=doc["text"],
                doc_id=doc.get("doc_id", ""),
                doc_title=doc.get("title", ""),
                doc_summary=doc.get("summary", ""),
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        logger.info(
            "Contextual chunking complete: %d documents → %d chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    def __repr__(self) -> str:
        mode = "LLM" if self.use_llm else "rule-based"
        return f"ContextualChunker(mode={mode}, base={self.base_chunker!r})"
