"""
AutoDrive RAG v2.0 — Parent Document Retriever
Solves the precision-vs-context tradeoff by storing small child chunks
for precise search but returning larger parent chunks for richer LLM context.

Problem: Small chunks → better search precision, but too little context for LLM.
         Large chunks → rich context, but diluted search relevance.

Solution: Index small child chunks (200 tokens) for search.
          On retrieval, return their parent chunks (1000 tokens) for generation.
          Best of both worlds.

Reference: LangChain ParentDocumentRetriever pattern.
"""

from __future__ import annotations

import logging
from typing import Optional

from .chunker import DocumentChunker, Chunk

logger = logging.getLogger("chatbot.retrieval.parent_doc")


class ParentDocumentRetriever:
    """
    Two-level chunking with parent-child mapping for retrieval.

    Child chunks are small and precise (used for search).
    Parent chunks are larger and context-rich (returned to LLM).

    Attributes:
        parent_chunker: Chunker for creating parent-level chunks.
        child_chunker: Chunker for creating child-level chunks (smaller).
        parent_chunks: Dict mapping parent_id → parent Chunk.
        child_to_parent: Dict mapping child_id → parent_id.
        child_chunks: List of all child chunks (for indexing).
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_overlap: int = 200,
        child_chunk_size: int = 300,
        child_overlap: int = 50,
    ) -> None:
        """
        Initialize with separate chunking parameters for parents and children.

        Args:
            parent_chunk_size: Size of parent chunks (returned to LLM).
            parent_overlap: Overlap for parent chunks.
            child_chunk_size: Size of child chunks (used for search).
            child_overlap: Overlap for child chunks.
        """
        self.parent_chunker = DocumentChunker(
            strategy="recursive",
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self.child_chunker = DocumentChunker(
            strategy="sentence",
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            min_chunk_size=30,
        )

        # Stores
        self.parent_chunks: dict[str, Chunk] = {}
        self.child_to_parent: dict[int, str] = {}
        self.child_chunks: list[Chunk] = []

    def process_document(
        self,
        text: str,
        doc_id: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Split document into parent and child chunks, maintaining mapping.

        Args:
            text: Full document text.
            doc_id: Document identifier.
            metadata: Additional metadata.

        Returns:
            List of child chunks (these are what you index for search).
        """
        base_metadata = metadata or {}

        # Step 1: Create parent chunks
        parents = self.parent_chunker.chunk_document(
            text=text, doc_id=doc_id, metadata=base_metadata
        )

        all_children = []

        for parent in parents:
            parent_id = f"{doc_id}__parent_{parent.chunk_index}"
            self.parent_chunks[parent_id] = parent

            # Step 2: Split each parent into child chunks
            children = self.child_chunker.chunk_document(
                text=parent.text,
                doc_id=doc_id,
                metadata={
                    **base_metadata,
                    "parent_id": parent_id,
                    "parent_chunk_index": parent.chunk_index,
                },
            )

            for child in children:
                child_idx = len(self.child_chunks)
                self.child_to_parent[child_idx] = parent_id
                self.child_chunks.append(child)
                all_children.append(child)

        logger.info(
            "Parent-child chunking for '%s': %d parents → %d children",
            doc_id[:30],
            len(parents),
            len(all_children),
        )
        return all_children

    def process_documents(
        self,
        documents: list[dict],
    ) -> list[Chunk]:
        """
        Process multiple documents, building parent-child mappings.

        Args:
            documents: List of dicts with 'text', 'doc_id', optional 'metadata'.

        Returns:
            Flat list of all child chunks (index these for search).
        """
        all_children = []
        for doc in documents:
            children = self.process_document(
                text=doc["text"],
                doc_id=doc.get("doc_id", ""),
                metadata=doc.get("metadata", {}),
            )
            all_children.extend(children)

        logger.info(
            "Processed %d documents → %d parent chunks, %d child chunks",
            len(documents),
            len(self.parent_chunks),
            len(all_children),
        )
        return all_children

    def get_child_texts(self) -> tuple[list[str], list[dict]]:
        """
        Get all child chunk texts and metadata for indexing.

        Returns:
            Tuple of (texts, metadata_list) suitable for building retrieval index.
        """
        texts = [c.text for c in self.child_chunks]
        metadata = [c.metadata for c in self.child_chunks]
        return texts, metadata

    def retrieve_parents(
        self,
        child_indices: list[int],
        deduplicate: bool = True,
    ) -> list[Chunk]:
        """
        Given child indices from search results, return their parent chunks.

        Args:
            child_indices: Indices of matched child chunks.
            deduplicate: If True, return unique parents only.

        Returns:
            List of parent Chunk objects with richer context.
        """
        parent_ids_seen = set()
        parents = []

        for idx in child_indices:
            parent_id = self.child_to_parent.get(idx)
            if parent_id is None:
                continue

            if deduplicate and parent_id in parent_ids_seen:
                continue

            parent_ids_seen.add(parent_id)
            parent = self.parent_chunks.get(parent_id)
            if parent:
                parents.append(parent)

        logger.debug(
            "Retrieved %d parent chunks from %d child matches",
            len(parents),
            len(child_indices),
        )
        return parents

    def retrieve_parent_texts(
        self,
        search_results: list[dict],
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Given search results (with 'text' matching child chunks),
        find and return the corresponding parent chunks.

        Args:
            search_results: List of search result dicts from the retriever.
            deduplicate: If True, return unique parents only.

        Returns:
            List of result dicts with parent chunk text substituted.
        """
        # Build a lookup from child text → child index
        child_text_to_idx = {
            c.text: i for i, c in enumerate(self.child_chunks)
        }

        parent_ids_seen = set()
        parent_results = []

        for result in search_results:
            child_idx = child_text_to_idx.get(result.get("text"))
            if child_idx is None:
                # Not a child chunk — return as-is
                parent_results.append(result)
                continue

            parent_id = self.child_to_parent.get(child_idx)
            if parent_id is None:
                parent_results.append(result)
                continue

            if deduplicate and parent_id in parent_ids_seen:
                continue
            parent_ids_seen.add(parent_id)

            parent = self.parent_chunks.get(parent_id)
            if parent:
                parent_results.append({
                    **result,
                    "text": parent.text,
                    "child_text": result["text"],
                    "parent_id": parent_id,
                    "source": result.get("source", "parent_doc"),
                })

        return parent_results

    def __repr__(self) -> str:
        return (
            f"ParentDocumentRetriever("
            f"parents={len(self.parent_chunks)}, "
            f"children={len(self.child_chunks)})"
        )
