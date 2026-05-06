"""
AutoDrive Chatbot — RAG Engine
Fetches live inventory from the AutoDrive API with in-memory TTL cache.
Falls back to seed_data.json if the API is unreachable.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time

import httpx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import settings

logger = logging.getLogger("chatbot.rag")

INVENTORY_API_URL: str = os.getenv(
    "INVENTORY_API_URL", "https://autodriveai.duckdns.org/api/cars"
)
# How long (seconds) before the cache is considered stale.
# 300 s = 5 min — fresh enough for live demos, cheap on API calls.
CACHE_TTL: int = int(os.getenv("INVENTORY_CACHE_TTL", "300"))


# ── System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are **AutoDrive AI**, a professional and knowledgeable automotive sales consultant \
for AutoDrive — India's trusted pre-owned car marketplace.

## Your Role
Help customers find the right car, understand pricing, compare options, and book test drives. \
Be warm, consultative, and specific — like a trusted showroom advisor who knows every car in stock.

## Key Guidelines
- **Currency:** Always use Indian Rupees (₹). Express prices as "₹X lakh" or "₹X.XX lakh". \
  Never use $ or any foreign currency.
- **Location matters:** Mention the city where each car is located so customers know where to pick it up.
- **Be specific:** Quote exact prices, mileage (in km), fuel type, and key features from the inventory.
- **Concise answers:** 2–4 short paragraphs or a clean bullet list. Avoid walls of text.
- **Honest limitations:** If a car isn't in the current inventory, say so and suggest the closest match.

## Special Instructions

### Car References (IMPORTANT)
Every time you mention a specific car by name in your response, you MUST immediately follow it \
with its ID tag in this exact format: `[CAR_ID:X]` where X is the car's numeric ID from the inventory.
Example: "The Hyundai Creta [CAR_ID:2] is an excellent choice at ₹14.50 lakh."
This enables one-click deep-links for the customer. Never skip the tag when naming a car.

### Recommendations
When asked for recommendations:
1. Clarify the customer's key criteria (budget, fuel preference, body type, city if possible)
2. Suggest the top 2–3 best-fit cars from the current inventory
3. For each, give: price, mileage, standout features, and location

### Test Drive Booking
When a customer expresses intent to book a test drive:
1. Confirm which specific car they want
2. Once confirmed, include this exact tag at the end of your message: `[ACTION: BOOK_TEST_DRIVE <car_id>]`
3. Follow it with a friendly confirmation message

### Electric Vehicles
Always mention the range prominently for EVs. Highlight home-charger inclusion if present.

## Current Inventory
{context}"""


def _fmt_price(price: int | None) -> str:
    """Format price as ₹X.XX lakh or ₹X.XX Cr."""
    if not price:
        return "Price on request"
    if price >= 10_000_000:
        return f"₹{price / 10_000_000:.2f} Cr"
    return f"₹{price / 100_000:.2f} lakh"


def _car_to_text(car: dict) -> str:
    """Convert a car dict to a dense text line for the LLM context."""
    features = ", ".join(car.get("features") or [])
    engine = f"{car['engine_cc']} cc" if car.get("engine_cc") else "Electric"
    return (
        f"[ID:{car['id']}] {car.get('year')} {car.get('make')} {car.get('model')} | "
        f"Price: {_fmt_price(car.get('price'))} | "
        f"AI-estimated value: {_fmt_price(car.get('ml_price'))} | "
        f"Mileage: {car.get('mileage', 0):,} km | "
        f"Fuel: {car.get('fuel_type')} | "
        f"Transmission: {car.get('transmission')} | "
        f"Body: {car.get('body_type')} | "
        f"Engine: {engine} | "
        f"Seats: {car.get('seating', 5)} | "
        f"Color: {car.get('color')} | "
        f"Owners: {car.get('owners', 1)} | "
        f"Location: {car.get('location')} | "
        f"Rating: {car.get('rating', 0)}/5 ({car.get('reviews', 0)} reviews) | "
        f"Features: {features} | "
        f"Details: {car.get('description', '')}"
    )


# ── Inventory Cache ──────────────────────────────────────────────────
class InventoryCache:
    """
    In-memory TTL cache for the live car inventory.
    - Lazy refresh: fetches on first use, re-fetches when TTL expires
    - Thread-safe: asyncio.Lock prevents thundering-herd on concurrent requests
    - Offline fallback: reads seed_data.json if the API is unreachable
    - Background refresh: optional periodic task to proactively warm the cache
    """

    def __init__(self) -> None:
        self._cars: list[dict] = []
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > CACHE_TTL

    async def _fetch(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(INVENTORY_API_URL)
            resp.raise_for_status()
            data = resp.json()
            return data.get("cars", data) if isinstance(data, dict) else data

    def _load_seed(self) -> list[dict]:
        with open(settings.SEED_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    async def get(self) -> list[dict]:
        """Return cached cars, refreshing if stale."""
        if not self._is_stale():
            return self._cars

        async with self._lock:
            if not self._is_stale():          # re-check after acquiring lock
                return self._cars
            try:
                cars = await self._fetch()
                self._cars = cars
                self._fetched_at = time.monotonic()
                logger.info("Inventory refreshed from API: %d cars", len(cars))
            except Exception as exc:
                logger.warning("API fetch failed (%s) — using %s data",
                               exc, "cached" if self._cars else "seed")
                if not self._cars:
                    self._cars = self._load_seed()
                    self._fetched_at = time.monotonic()
            return self._cars

    async def force_refresh(self) -> int:
        """Bypass TTL and refresh immediately. Returns new car count."""
        self._fetched_at = 0.0          # mark stale
        cars = await self.get()
        return len(cars)

    async def get_context(self, query: str) -> str:
        """
        Build the context string for the LLM.
        ≤ 30 cars  → include all (LLM sees full inventory, most reliable)
        > 30 cars  → TF-IDF top-K to stay within token budget
        """
        cars = await self.get()
        if len(cars) <= 30:
            lines = [_car_to_text(c) for c in cars]
        else:
            docs = [Document(page_content=_car_to_text(c)) for c in cars]
            retriever = TFIDFRetriever(docs, k=settings.RETRIEVER_K)
            lines = [d.page_content for d in retriever.invoke(query)]
        return "\n".join(lines)


# Global cache instance (shared across requests, lives for process lifetime)
_inventory = InventoryCache()


# ── TF-IDF Retriever (free, no API key) ─────────────────────────────
class TFIDFRetriever:
    def __init__(self, documents: list[Document], k: int = 5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.documents = documents
        self.k = k
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [d.page_content for d in documents]
        )
        self._sim = cosine_similarity

    def invoke(self, query: str) -> list[Document]:
        q_vec = self.vectorizer.transform([query])
        scores = self._sim(q_vec, self.tfidf_matrix)[0]
        top_idx = scores.argsort()[-self.k:][::-1]
        return [self.documents[i] for i in top_idx if scores[i] > 0]

    async def ainvoke(self, query: str) -> list[Document]:
        return self.invoke(query)


# ── LLM Factory ─────────────────────────────────────────────────────
def _get_llm():
    provider = settings.llm_provider
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
            streaming=True,
            temperature=settings.LLM_TEMPERATURE,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            api_version="2024-02-01",
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            streaming=True,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
    # Ollama local
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "No LLM configured. Set GROQ_API_KEY in .env or install langchain-ollama."
        )
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


# ── Public API ───────────────────────────────────────────────────────
class RAGEngine:
    """Encapsulates retrieval + LLM streaming. One instance per process."""

    def __init__(self) -> None:
        self.llm = _get_llm()
        self.prompt = _build_prompt()

    async def retrieve_context(self, query: str) -> str:
        return await _inventory.get_context(query)

    async def stream_response(self, user_msg: str, context: str, chat_history: list):
        chain = self.prompt | self.llm
        async for chunk in chain.astream({
            "input": user_msg,
            "context": context,
            "chat_history": chat_history,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield token


async def force_inventory_refresh() -> int:
    """Exposed so the /inventory/refresh endpoint can call it."""
    return await _inventory.force_refresh()
