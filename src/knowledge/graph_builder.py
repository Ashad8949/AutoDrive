"""
AutoDrive RAG v2.0 — Knowledge Graph Builder
Builds a structured entity-relationship graph from car inventory data
for multi-hop reasoning and structured queries.

Graph structure:
  Car -[HAS_FEATURE]-> Feature
  Car -[LOCATED_IN]-> City
  Car -[FUEL_TYPE]-> FuelType
  Car -[BODY_TYPE]-> BodyType
  Car -[MADE_BY]-> Make
  Car -[PRICED_AT]-> PriceRange
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("chatbot.knowledge.graph")


class KnowledgeGraphBuilder:
    """
    Builds a knowledge graph from car inventory using NetworkX.

    Nodes: Cars, Makes, FuelTypes, BodyTypes, Cities, Features, PriceRanges
    Edges: Typed relationships between entities
    """

    def __init__(self) -> None:
        try:
            import networkx as nx
            self.graph = nx.DiGraph()
            self._nx = nx
        except ImportError:
            logger.warning("networkx not installed — KG disabled")
            self.graph = None
            self._nx = None

    def build_from_inventory(self, cars: list[dict]) -> None:
        """Build the knowledge graph from a list of car dicts."""
        if self.graph is None:
            return

        for car in cars:
            car_id = f"car_{car['id']}"
            car_label = f"{car.get('year', '')} {car.get('make', '')} {car.get('model', '')}"

            # Car node
            self.graph.add_node(car_id, type="car", label=car_label, **{
                k: v for k, v in car.items() if k not in ("features", "description")
            })

            # Make
            make = car.get("make", "Unknown")
            make_id = f"make_{make.lower().replace(' ', '_')}"
            self.graph.add_node(make_id, type="make", label=make)
            self.graph.add_edge(car_id, make_id, relation="MADE_BY")

            # Fuel type
            fuel = car.get("fuel_type", "Unknown")
            fuel_id = f"fuel_{fuel.lower()}"
            self.graph.add_node(fuel_id, type="fuel_type", label=fuel)
            self.graph.add_edge(car_id, fuel_id, relation="FUEL_TYPE")

            # Body type
            body = car.get("body_type", "Unknown")
            body_id = f"body_{body.lower()}"
            self.graph.add_node(body_id, type="body_type", label=body)
            self.graph.add_edge(car_id, body_id, relation="BODY_TYPE")

            # Location
            location = car.get("location", "Unknown")
            loc_id = f"city_{location.lower().replace(' ', '_')}"
            self.graph.add_node(loc_id, type="city", label=location)
            self.graph.add_edge(car_id, loc_id, relation="LOCATED_IN")

            # Features
            for feat in car.get("features", []):
                feat_id = f"feat_{feat.lower().replace(' ', '_')}"
                self.graph.add_node(feat_id, type="feature", label=feat)
                self.graph.add_edge(car_id, feat_id, relation="HAS_FEATURE")

            # Price range
            price = car.get("price", 0)
            if price < 500_000:
                pr = "under_5L"
            elif price < 1_000_000:
                pr = "5L_10L"
            elif price < 2_000_000:
                pr = "10L_20L"
            elif price < 5_000_000:
                pr = "20L_50L"
            else:
                pr = "above_50L"
            pr_id = f"price_{pr}"
            self.graph.add_node(pr_id, type="price_range", label=pr.replace("_", " "))
            self.graph.add_edge(car_id, pr_id, relation="PRICED_AT")

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def get_car_neighbors(self, car_id: str) -> dict:
        """Get all entities related to a car."""
        if self.graph is None:
            return {}

        result = {}
        for _, target, data in self.graph.out_edges(car_id, data=True):
            relation = data.get("relation", "RELATED")
            node_data = self.graph.nodes.get(target, {})
            result.setdefault(relation, []).append(node_data.get("label", target))
        return result

    def find_similar_cars(self, car_id: str) -> list[str]:
        """Find cars sharing the most features/attributes with given car."""
        if self.graph is None:
            return []

        car_neighbors = set(self.graph.successors(car_id))
        scores = {}

        for node in self.graph.nodes:
            if node == car_id or self.graph.nodes[node].get("type") != "car":
                continue
            node_neighbors = set(self.graph.successors(node))
            overlap = len(car_neighbors & node_neighbors)
            if overlap > 0:
                scores[node] = overlap

        sorted_cars = sorted(scores, key=scores.get, reverse=True)
        return sorted_cars[:5]

    def query_by_attributes(self, **kwargs) -> list[str]:
        """Find cars matching given attribute constraints via graph traversal."""
        if self.graph is None:
            return []

        candidates = set()
        first = True

        for attr_type, attr_value in kwargs.items():
            matching_cars = set()
            attr_value_lower = attr_value.lower().replace(" ", "_")

            for node, data in self.graph.nodes(data=True):
                if data.get("type") == attr_type or data.get("label", "").lower() == attr_value.lower():
                    predecessors = set(self.graph.predecessors(node))
                    car_preds = {n for n in predecessors if self.graph.nodes[n].get("type") == "car"}
                    matching_cars.update(car_preds)

            if first:
                candidates = matching_cars
                first = False
            else:
                candidates &= matching_cars

        return list(candidates)

    def get_stats(self) -> dict:
        if self.graph is None:
            return {"error": "Graph not available"}
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": type_counts,
        }
