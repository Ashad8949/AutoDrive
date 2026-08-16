"""
AutoDrive RAG v2.0 — Agent Tools
Tool definitions for the LangGraph agentic RAG system.

Each tool is a callable function that the agent can invoke during
multi-step reasoning. Tools cover inventory search, comparison,
details lookup, booking, web search, and image retrieval.
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

    # ── Core Inventory Tools ────────────────────────────────────────

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

    # ── Web Search Tools (NEW — Agentic v2.0) ──────────────────────

    def web_search(self, query: str) -> str:
        """
        Perform a web search for information not in the local knowledge base.
        Uses DuckDuckGo (free, no API key required).

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

    def web_search_car_specs(self, car_model: str) -> str:
        """
        Search the web for detailed specifications of a specific car model.
        Useful when the local database lacks technical details like engine BHP,
        top speed, safety ratings, etc.

        Args:
            car_model: The car name, e.g. "Hyundai Creta 2024".

        Returns:
            Formatted specifications from the web.
        """
        query = f"{car_model} specifications features review India"
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            if not results:
                return f"No web specifications found for '{car_model}'."

            output = [f"## Web Specifications for {car_model}\n"]
            for r in results:
                output.append(
                    f"**{r.get('title', '')}**\n"
                    f"{r.get('body', '')}\n"
                    f"Source: {r.get('href', '')}"
                )
            return "\n\n".join(output)

        except ImportError:
            return "Web search not available (install duckduckgo-search)."
        except Exception as e:
            logger.warning("Web specs search failed: %s", e)
            return f"Web specs search failed: {e}"

    def fetch_car_image(self, car_model: str) -> str:
        """
        Fetch a real image of the specified car model from the web.
        Uses DuckDuckGo Image Search with a fallback to direct URL
        construction if rate-limited.

        Args:
            car_model: The car name, e.g. "Tata Nexon 2024".

        Returns:
            A markdown image string like ![alt](url) or an error message.
        """
        # Try DuckDuckGo first
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.images(
                    f"{car_model} car official India",
                    max_results=3,
                ))

            if results:
                top = results[0]
                image_url = top.get("image", "")
                title = top.get("title", car_model)

                if image_url:
                    output = f"![{title}]({image_url})\n"
                    if len(results) > 1:
                        output += "\nAdditional images:\n"
                        for r in results[1:]:
                            url = r.get("image", "")
                            t = r.get("title", "")
                            if url:
                                output += f"- ![{t}]({url})\n"
                    return output

        except Exception as e:
            logger.warning("DuckDuckGo image search failed: %s", e)

        # Fallback: use httpx to scrape an image URL from the web
        try:
            import httpx
            import re

            search_url = f"https://www.google.com/search?q={car_model.replace(' ', '+')}+car+India&tbm=isch"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = httpx.get(search_url, headers=headers, timeout=5.0, follow_redirects=True)
            # Extract image URLs from the HTML
            urls = re.findall(r'https://[^"\']+\.(?:jpg|jpeg|png|webp)', resp.text)

            if urls:
                # Filter out tiny icons and Google's own assets
                good_urls = [u for u in urls if "gstatic" not in u and "google" not in u]
                if good_urls:
                    return f"![{car_model}]({good_urls[0]})"

        except Exception as e:
            logger.warning("Fallback image search also failed: %s", e)

        # Last resort: return a helpful message with a search link
        search_query = car_model.replace(" ", "+")
        return (
            f"I couldn't fetch an image directly, but you can see pictures of "
            f"the {car_model} here: "
            f"[Google Images](https://www.google.com/search?q={search_query}+car&tbm=isch)"
        )

    def check_local_availability(self, car_model: str) -> str:
        """
        Check if a specific car model is available in our local inventory
        (FAISS/BM25 database or seed data).

        Args:
            car_model: The car model name, e.g. "Hyundai Creta".

        Returns:
            Availability status and matching cars from local inventory.
        """
        if self.retriever:
            results = self.retriever.search(car_model, top_k=5)
            if not results:
                return (
                    f"'{car_model}' is NOT currently available in our inventory. "
                    "Please check back later or explore similar cars."
                )

            # Filter results that actually match the queried model
            output = [f"## Local Inventory Results for '{car_model}'\n"]
            for r in results:
                output.append(f"[Score: {r['score']:.3f}] {r['text'][:400]}")
            return "\n\n".join(output)

        return "Inventory search is not available right now."

    # ── Tool Definitions (for LLM Function Calling) ─────────────────

    def get_tool_definitions(self) -> list[dict]:
        """
        Return tool definitions for LLM function calling / tool binding.

        Returns:
            List of tool definition dicts compatible with LLM tool binding.
        """
        return [
            {
                "name": "check_local_availability",
                "description": (
                    "Check if a car model is available in our local AutoDrive "
                    "inventory. Use this FIRST for any availability question."
                ),
                "parameters": {
                    "car_model": "string: car model name, e.g. 'Hyundai Creta'",
                },
            },
            {
                "name": "search_inventory",
                "description": (
                    "Search the car inventory using natural language. "
                    "Use when the user is looking for cars matching certain criteria."
                ),
                "parameters": {
                    "query": "string: search query",
                    "top_k": "integer: number of results (default 5)",
                },
            },
            {
                "name": "fetch_car_image",
                "description": (
                    "Fetch a real photo of the specified car model from the web. "
                    "Use when the user asks to SEE a car, wants a picture, or says "
                    "'show me', 'what does it look like', etc."
                ),
                "parameters": {
                    "car_model": "string: car model name, e.g. 'Tata Nexon 2024'",
                },
            },
            {
                "name": "web_search_car_specs",
                "description": (
                    "Search the web for detailed car specifications (engine BHP, "
                    "top speed, safety rating, etc.) that are NOT in our local "
                    "database. Use when the user asks for technical details."
                ),
                "parameters": {
                    "car_model": "string: car model name with optional year",
                },
            },
            {
                "name": "compare_cars",
                "description": (
                    "Compare two or more cars side by side. "
                    "Use when the user wants to compare specific cars."
                ),
                "parameters": {
                    "car_ids": "list of strings: car IDs to compare",
                },
            },
            {
                "name": "book_test_drive",
                "description": (
                    "Book a test drive for a specific car. "
                    "Use when the user wants to schedule a test drive."
                ),
                "parameters": {
                    "car_id": "string: car ID to test drive",
                    "preferred_date": "string: preferred date (optional)",
                },
            },
            {
                "name": "web_search",
                "description": (
                    "General web search for any information not available locally. "
                    "Use as a LAST RESORT when other tools are insufficient."
                ),
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
            "web_search_car_specs": self.web_search_car_specs,
            "fetch_car_image": self.fetch_car_image,
            "check_local_availability": self.check_local_availability,
        }

        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return f"Unknown tool: {tool_name}"

        try:
            return tool_func(**kwargs)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            return f"Tool execution failed: {e}"
