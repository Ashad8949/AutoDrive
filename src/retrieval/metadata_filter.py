"""
AutoDrive RAG v2.0 — Metadata Filter
Provides structured pre-filtering of documents by metadata fields
BEFORE vector search, reducing the search space and improving precision.

Example use case:
  Query: "Show me diesel SUVs under ₹15 lakh in Mumbai"
  → Filter by: fuel_type="Diesel", body_type="SUV", price<=1500000, location="Mumbai"
  → Then vector search within the filtered subset
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("chatbot.retrieval.metadata_filter")


class MetadataFilter:
    """
    Filters documents by structured metadata before retrieval.

    Supports filtering by exact match, range, and list membership
    on metadata fields like fuel_type, body_type, price, location, etc.
    """

    # Common filter patterns extracted from user queries
    FUEL_TYPES = {"petrol", "diesel", "electric", "cng", "hybrid", "ev"}
    BODY_TYPES = {"suv", "sedan", "hatchback", "mpv", "coupe", "convertible", "pickup"}
    TRANSMISSIONS = {"manual", "automatic", "amt", "cvt", "dct", "dsg", "imt"}

    def filter_documents(
        self,
        documents: list[str],
        metadata: list[dict],
        filters: dict[str, Any],
    ) -> tuple[list[str], list[dict], list[int]]:
        """
        Filter documents by metadata criteria.

        Args:
            documents: List of document text strings.
            metadata: List of metadata dicts (parallel to documents).
            filters: Dict of filter criteria. Supported keys:
                - fuel_type: str — exact match (case-insensitive)
                - body_type: str — exact match (case-insensitive)
                - transmission: str — exact match (case-insensitive)
                - location: str — exact match (case-insensitive)
                - make: str — exact match (case-insensitive)
                - min_price: int/float — minimum price
                - max_price: int/float — maximum price
                - min_year: int — minimum year
                - max_year: int — maximum year
                - max_mileage: int — maximum mileage
                - max_owners: int — maximum number of previous owners

        Returns:
            Tuple of (filtered_documents, filtered_metadata, original_indices).
        """
        if not filters or not documents:
            return documents, metadata, list(range(len(documents)))

        filtered_docs = []
        filtered_meta = []
        original_indices = []

        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            if self._matches(meta, filters):
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                original_indices.append(i)

        logger.info(
            "Metadata filter: %d/%d documents matched filters %s",
            len(filtered_docs),
            len(documents),
            filters,
        )

        # If no documents match, fall back to unfiltered (avoid empty results)
        if not filtered_docs:
            logger.warning(
                "No documents matched filters — falling back to unfiltered"
            )
            return documents, metadata, list(range(len(documents)))

        return filtered_docs, filtered_meta, original_indices

    def _matches(self, meta: dict, filters: dict) -> bool:
        """Check if a single document's metadata matches all filter criteria."""
        for key, value in filters.items():
            if value is None:
                continue

            if key == "fuel_type":
                if meta.get("fuel_type", "").lower() != value.lower():
                    return False

            elif key == "body_type":
                if meta.get("body_type", "").lower() != value.lower():
                    return False

            elif key == "transmission":
                if meta.get("transmission", "").lower() != value.lower():
                    return False

            elif key == "location":
                if value.lower() not in meta.get("location", "").lower():
                    return False

            elif key == "make":
                if value.lower() not in meta.get("make", "").lower():
                    return False

            elif key == "min_price":
                price = meta.get("price", 0)
                if isinstance(price, (int, float)) and price < value:
                    return False

            elif key == "max_price":
                price = meta.get("price", float("inf"))
                if isinstance(price, (int, float)) and price > value:
                    return False

            elif key == "min_year":
                year = meta.get("year", 0)
                if isinstance(year, int) and year < value:
                    return False

            elif key == "max_year":
                year = meta.get("year", 9999)
                if isinstance(year, int) and year > value:
                    return False

            elif key == "max_mileage":
                mileage = meta.get("mileage", 0)
                if isinstance(mileage, (int, float)) and mileage > value:
                    return False

            elif key == "max_owners":
                owners = meta.get("owners", 1)
                if isinstance(owners, int) and owners > value:
                    return False

        return True

    def extract_filters_from_query(self, query: str) -> dict[str, Any]:
        """
        Heuristically extract structured filters from a natural language query.

        This is a lightweight rule-based extractor. For production use,
        consider using an LLM or NER model for more robust extraction.

        Args:
            query: Natural language user query.

        Returns:
            Dict of extracted filter criteria.
        """
        filters: dict[str, Any] = {}
        query_lower = query.lower()

        # Extract fuel type
        for fuel in self.FUEL_TYPES:
            if fuel in query_lower:
                filters["fuel_type"] = fuel.capitalize()
                if fuel == "ev":
                    filters["fuel_type"] = "Electric"
                break

        # Extract body type
        for body in self.BODY_TYPES:
            if body in query_lower:
                filters["body_type"] = body.upper() if body == "suv" else body.capitalize()
                break

        # Extract transmission
        for trans in self.TRANSMISSIONS:
            if trans in query_lower:
                filters["transmission"] = "Automatic" if trans in {
                    "automatic", "amt", "cvt", "dct", "dsg", "imt"
                } else "Manual"
                break

        # Extract price constraints (₹ / lakh / cr patterns)
        price_under = re.search(
            r"(?:under|below|less than|max|upto|up to|within|budget)\s*"
            r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:lakh|lac|l\b)",
            query_lower,
        )
        if price_under:
            filters["max_price"] = int(float(price_under.group(1)) * 100_000)

        price_above = re.search(
            r"(?:above|over|more than|min|starting|from)\s*"
            r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(?:lakh|lac|l\b)",
            query_lower,
        )
        if price_above:
            filters["min_price"] = int(float(price_above.group(1)) * 100_000)

        # Extract location
        known_cities = [
            "delhi", "mumbai", "bangalore", "bengaluru", "pune", "chennai",
            "hyderabad", "kolkata", "jaipur", "ahmedabad", "lucknow",
            "chandigarh", "gurgaon", "noida", "goa",
        ]
        for city in known_cities:
            if city in query_lower:
                filters["location"] = city.capitalize()
                if city == "bengaluru":
                    filters["location"] = "Bangalore"
                break

        # Extract make
        known_makes = [
            "maruti", "hyundai", "tata", "honda", "toyota", "kia", "mg",
            "mahindra", "volkswagen", "skoda", "bmw", "mercedes", "audi",
            "renault", "nissan", "ford", "jeep",
        ]
        for make in known_makes:
            if make in query_lower:
                filters["make"] = make.capitalize()
                if make == "bmw":
                    filters["make"] = "BMW"
                elif make == "mg":
                    filters["make"] = "MG"
                break

        return filters
