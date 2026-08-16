"""
AutoDrive RAG v2.0 — GraphRAG
Combines knowledge graph traversal with vector retrieval for
multi-hop reasoning that pure vector search would miss.
"""

from __future__ import annotations

import logging
from typing import Optional

from .graph_builder import KnowledgeGraphBuilder

logger = logging.getLogger("chatbot.knowledge.graph_rag")


class GraphRAG:
    """
    Combines knowledge graph queries with vector retrieval.

    Use cases:
      - "Cars with better safety than the Creta" → graph traversal
      - "All SUVs in Delhi with sunroof" → structured graph query
      - "Similar cars to car #5" → graph neighbor similarity
    """

    def __init__(self, kg: Optional[KnowledgeGraphBuilder] = None, retriever=None) -> None:
        self.kg = kg or KnowledgeGraphBuilder()
        self.retriever = retriever

    def hybrid_search(self, query: str, structured_filters: dict = None, top_k: int = 5) -> list[dict]:
        """
        Combine graph-based and vector-based retrieval.

        1. Try structured graph query first (if filters available)
        2. Enrich with vector search results
        3. Merge and deduplicate
        """
        graph_results = []
        vector_results = []

        # Graph-based retrieval
        if structured_filters and self.kg.graph:
            car_ids = self.kg.query_by_attributes(**structured_filters)
            for car_id in car_ids[:top_k]:
                node_data = self.kg.graph.nodes.get(car_id, {})
                neighbors = self.kg.get_car_neighbors(car_id)
                graph_results.append({
                    "text": f"{node_data.get('label', car_id)} | " + " | ".join(
                        f"{k}: {', '.join(v)}" for k, v in neighbors.items()
                    ),
                    "score": 1.0,
                    "metadata": {"car_id": car_id, **node_data},
                    "source": "graph",
                })

        # Vector-based retrieval
        if self.retriever:
            vector_results = self.retriever.search(query, top_k=top_k)

        # Merge (graph results first for precision)
        seen = set()
        merged = []
        for r in graph_results + vector_results:
            key = r.get("metadata", {}).get("car_id", r["text"][:50])
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged[:top_k]

    def find_similar(self, car_id: str) -> list[dict]:
        """Find cars similar to a given car using graph topology."""
        if not self.kg.graph:
            return []

        similar_ids = self.kg.find_similar_cars(f"car_{car_id}")
        results = []
        for sid in similar_ids:
            node_data = self.kg.graph.nodes.get(sid, {})
            results.append({
                "text": node_data.get("label", sid),
                "metadata": node_data,
                "source": "graph_similarity",
            })
        return results

    def multi_hop_query(self, query: str, start_entity: str, hops: int = 2) -> list[dict]:
        """
        Multi-hop graph traversal for complex queries.

        Example: "Cars with better safety than the Creta"
        → Find Creta → get its features → find cars with more safety features
        """
        if not self.kg.graph:
            return []

        visited = set()
        current_nodes = {start_entity}
        all_found = []

        for hop in range(hops):
            next_nodes = set()
            for node in current_nodes:
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in self.kg.graph.successors(node):
                    next_nodes.add(neighbor)
                for predecessor in self.kg.graph.predecessors(node):
                    next_nodes.add(predecessor)
            current_nodes = next_nodes

        # Collect car nodes found
        for node in visited | current_nodes:
            node_data = self.kg.graph.nodes.get(node, {})
            if node_data.get("type") == "car":
                all_found.append({
                    "text": node_data.get("label", node),
                    "metadata": node_data,
                    "source": "graph_multi_hop",
                })

        return all_found
