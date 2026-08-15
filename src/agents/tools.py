"""
AutoDrive RAG v2.0 — Agent Tools
Tool definitions for the LangGraph agentic RAG system.

Each tool is a callable function that the agent can invoke during
multi-step reasoning. Tools cover inventory search, comparison,
details lookup, booking, and web search fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("chatbot.agents.tools")


class AgentTools:
    """
    Collection of tools available to the RAG agent.

    Each tool is a method that takes structured input and returns
    a formatted string result that the agent can incorporate into
    its reasoning.

    Attributes:
        retriever: HybridRetriever instance for searching.
        metadata_filter: MetadataFilter for structured filtering.
        inventory_cache: InventoryCache for direct data access.
    """

    def __init__(
        self,
        retriever=None,
        metadata_filter=None,
        inventory_cache=None,
    ) -> None:
        self.retriever = retriever
        self.metadata_filter = metadata_filter
        self.inventory_cache = inventory_cache

    def search_inventory(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: int = 5,
    ) -> str:
        """
        Search the car inventory with optional structured filters.

        Args:
            query: Natural language search query.
            filters: Optional dict of metadata filters
                     (fuel_type, body_type, max_price, location, etc.)
            top_k: Number of results to return.

        Returns:
            Formatted string of matching cars.
        """
        if self.retriever:
            results = self.retriever.search(query, top_k=top_k)
            if not results:
                return "No matching cars found in the inventory."

            output = []
            for r in results:
                output.append(f"[Score: {r['score']:.3f}] {r['text'][:300]}")
            return "\n\n".join(output)

        return "Retriever not available."

    def compare_cars(self, car_ids: list[str]) -> str:
        """
        Generate a side-by-side comparison of specified cars.

        Args:
            car_ids: List of car IDs to compare.

        Returns:
            Formatted comparison table as string.
        """
        if not self.inventory_cache:
            return "Inventory data not available for comparison."

        # This would need async context — return structured format
        return (
            f"Compare cars with IDs: {', '.join(car_ids)}. "
            "Please retrieve details for each car and present a "
            "side-by-side comparison of price, mileage, features, "
            "and key specifications."
        )

    def get_car_details(self, car_id: str) -> str:
        """
        Get full details for a specific car by ID.

        Args:
            car_id: The car's unique identifier.

        Returns:
            Formatted car details string.
        """
        return (
            f"Retrieve complete details for car ID: {car_id}. "
            "Include price, mileage, features, condition, location, "
            "and any special notes."
        )

    def check_market_value(
        self,
        make: str,
        model: str,
        year: int,
    ) -> str:
        """
        Check the estimated market value for a car.

        Args:
            make: Car manufacturer.
            model: Car model name.
            year: Manufacturing year.

        Returns:
            Market value estimate string.
        """
        return (
            f"Market value check for {year} {make} {model}. "
            "Compare the listed price against the ML-estimated fair value "
            "to determine if it's a good deal."
        )

    def book_test_drive(
        self,
        car_id: str,
        preferred_date: Optional[str] = None,
    ) -> str:
        """
        Initiate a test drive booking for a specific car.

        Args:
            car_id: The car to test drive.
            preferred_date: Optional preferred date.

        Returns:
            Booking confirmation or next steps.
        """
        return f"[ACTION: BOOK_TEST_DRIVE {car_id}]"

    def web_search(self, query: str) -> str:
        """
        Perform a web search for information not in the local knowledge base.

        Args:
            query: Search query.

        Returns:
            Web search results as formatted string.
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            if not results:
                return "No web results found."

            output = []
            for r in results:
                output.append(
                    f"**{r.get('title', 'N/A')}**\n"
                    f"{r.get('body', '')}\n"
                    f"Source: {r.get('href', '')}"
                )
            return "\n\n".join(output)

        except ImportError:
            return "Web search not available (install duckduckgo-search)."
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            return f"Web search failed: {e}"

    def get_tool_definitions(self) -> list[dict]:
        """
        Return tool definitions for LangChain/LangGraph binding.

        Returns:
            List of tool definition dicts compatible with LLM tool binding.
        """
        return [
            {
                "name": "search_inventory",
                "description": "Search the car inventory using natural language. "
                               "Use when the user is looking for cars matching certain criteria.",
                "parameters": {
                    "query": "string: search query",
                    "top_k": "integer: number of results (default 5)",
                },
            },
            {
                "name": "compare_cars",
                "description": "Compare two or more cars side by side. "
                               "Use when the user wants to compare specific cars.",
                "parameters": {
                    "car_ids": "list of strings: car IDs to compare",
                },
            },
            {
                "name": "get_car_details",
                "description": "Get full details about a specific car. "
                               "Use when the user asks about a particular car by ID.",
                "parameters": {
                    "car_id": "string: car ID",
                },
            },
            {
                "name": "book_test_drive",
                "description": "Book a test drive for a specific car. "
                               "Use when the user wants to schedule a test drive.",
                "parameters": {
                    "car_id": "string: car ID to test drive",
                    "preferred_date": "string: preferred date (optional)",
                },
            },
            {
                "name": "web_search",
                "description": "Search the web for information not available in inventory. "
                               "Use as a last resort when local knowledge is insufficient.",
                "parameters": {
                    "query": "string: web search query",
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Tool-specific arguments.

        Returns:
            Tool execution result string.
        """
        tool_map = {
            "search_inventory": self.search_inventory,
            "compare_cars": self.compare_cars,
            "get_car_details": self.get_car_details,
            "check_market_value": self.check_market_value,
            "book_test_drive": self.book_test_drive,
            "web_search": self.web_search,
        }

        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return f"Unknown tool: {tool_name}"

        try:
            return tool_func(**kwargs)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            return f"Tool execution failed: {e}"
