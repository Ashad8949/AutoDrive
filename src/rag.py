"""
AutoDrive RAG v2.0 — RAG Engine
Integrates all v2.0 subsystems into a unified pipeline:
  - Hybrid Retrieval (Dense + BM25 + RRF)
  - Document Chunking (5 strategies)
  - Cross-Encoder Re-Ranking
  - Query Transformation (HyDE, Multi-Query, Expansion)
  - Intent Classification & Adaptive Routing
  - Corrective RAG (quality grading + retry)
  - Advanced Memory (summary + entity tracking)
  - Knowledge Graph (GraphRAG)
  - Semantic Cache
  - Safety Guardrails
  - Evaluation Metrics (EM, F1, Retrieval)

Backwards compatible: still exposes RAGEngine with same interface.
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
        f"Rating: {car.get('rating') or 'N/A'}/5 ({car.get('reviews') or 0} reviews) | "
        f"Features: {features} | "
        f"Details: {car.get('description', '')}"
    )


def _car_to_metadata(car: dict) -> dict:
    """Extract structured metadata from a car dict for filtering."""
    return {
        "car_id": str(car.get("id", "")),
        "make": car.get("make", ""),
        "model": car.get("model", ""),
        "year": car.get("year", 0),
        "price": car.get("price", 0),
        "fuel_type": car.get("fuel_type", ""),
        "body_type": car.get("body_type", ""),
        "transmission": car.get("transmission", ""),
        "location": car.get("location", ""),
        "mileage": car.get("mileage", 0),
        "owners": car.get("owners", 1),
    }


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


# ── RAG v2.0 Pipeline Components ────────────────────────────────────

def _init_hybrid_retriever():
    """Lazily initialize the hybrid retriever from inventory data."""
    try:
        from .retrieval import HybridRetriever
        return HybridRetriever(
            fusion_strategy=settings.FUSION_STRATEGY,
            alpha=settings.FUSION_ALPHA,
        )
    except ImportError as e:
        logger.warning("Hybrid retriever unavailable (%s) — using TF-IDF fallback", e)
        return None


def _init_reranker():
    """Lazily initialize the cross-encoder re-ranker."""
    try:
        from .retrieval import CrossEncoderReranker
        return CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            top_n=settings.RETRIEVER_K,
        )
    except ImportError as e:
        logger.warning("Re-ranker unavailable (%s)", e)
        return None


def _init_query_transformer(llm):
    """Initialize query transformation pipeline."""
    try:
        from .query import QueryTransformer
        return QueryTransformer(llm=llm)
    except ImportError:
        return None


def _init_adaptive_router():
    """Initialize the adaptive query router."""
    try:
        from .query import AdaptiveRouter, IntentClassifier
        return AdaptiveRouter(intent_classifier=IntentClassifier())
    except ImportError:
        return None


def _init_semantic_cache():
    """Initialize the semantic cache if enabled."""
    if not settings.CACHE_ENABLED:
        return None
    try:
        from .cache import SemanticCache
        return SemanticCache(
            similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
            ttl_seconds=settings.CACHE_TTL,
        )
    except ImportError:
        return None


def _init_guardrails():
    """Initialize safety guardrails."""
    try:
        from .guardrails import SafetyGuardrails
        return SafetyGuardrails()
    except ImportError:
        return None


def _init_knowledge_graph():
    """Initialize the knowledge graph if enabled."""
    if not settings.KG_ENABLED:
        return None
    try:
        from .knowledge import KnowledgeGraphBuilder
        return KnowledgeGraphBuilder()
    except ImportError:
        return None


def _init_corrective_rag(llm):
    """Initialize Corrective RAG."""
    try:
        from .agents import CorrectiveRAG
        return CorrectiveRAG(llm=llm)
    except ImportError:
        return None


def _init_memory():
    """Initialize conversation memory."""
    try:
        from .memory import ConversationMemory
        return ConversationMemory()
    except ImportError:
        return None


# ── Public API ───────────────────────────────────────────────────────
class RAGEngine:
    """
    AutoDrive RAG v2.0 Engine.

    Integrates all v2.0 components while maintaining backwards
    compatibility with the v1 streaming interface.
    """

    def __init__(self) -> None:
        logger.info("Initializing RAG Engine v2.0...")
        self.llm = _get_llm()
        self.prompt = _build_prompt()

        # v2.0 components (lazy init — only loaded when first needed)
        self._hybrid_retriever = None
        self._reranker = None
        self._query_transformer = None
        self._adaptive_router = None
        self._semantic_cache = None
        self._guardrails = None
        self._knowledge_graph = None
        self._corrective_rag = None
        self._memory = None
        self._v2_initialized = False

        logger.info(
            "RAG Engine v2.0 initialized (provider=%s)", settings.llm_provider
        )

    def _ensure_v2_components(self) -> None:
        """Lazy-initialize v2.0 components on first use."""
        if self._v2_initialized:
            return

        logger.info("Loading RAG v2.0 components...")
        self._adaptive_router = _init_adaptive_router()
        self._query_transformer = _init_query_transformer(self.llm)
        self._semantic_cache = _init_semantic_cache()
        self._guardrails = _init_guardrails()
        self._corrective_rag = _init_corrective_rag(self.llm)
        self._memory = _init_memory()
        self._v2_initialized = True
        logger.info("RAG v2.0 components loaded ✓")

    async def _build_hybrid_index(self, cars: list[dict]) -> None:
        """Build hybrid index from car inventory data."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = _init_hybrid_retriever()
        if self._hybrid_retriever is None:
            return

        texts = [_car_to_text(c) for c in cars]
        metadata = [_car_to_metadata(c) for c in cars]

        # Build in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._hybrid_retriever.build_index, texts, metadata
        )

        # Build knowledge graph
        if self._knowledge_graph is None:
            self._knowledge_graph = _init_knowledge_graph()
        if self._knowledge_graph:
            self._knowledge_graph.build_from_inventory(cars)

        # Update guardrails with valid car IDs
        if self._guardrails:
            self._guardrails.update_inventory_ids(cars)

        logger.info("Hybrid index + KG built for %d cars", len(cars))

    async def retrieve_context(self, query: str) -> str:
        """Retrieve context — uses v2 hybrid retrieval if available, falls back to TF-IDF."""
        return await _inventory.get_context(query)

    async def retrieve_context_v2(self, query: str) -> tuple[str, dict]:
        """
        v2.0 retrieval pipeline:
        1. Route query → determine complexity
        2. Transform query if needed
        3. Hybrid search → re-rank
        4. Return context + pipeline metadata
        """
        self._ensure_v2_components()
        pipeline_meta = {"version": "v2.0", "stages": []}

        # 1. Adaptive routing
        route = None
        if self._adaptive_router:
            route = self._adaptive_router.route(query)
            pipeline_meta["route"] = {
                "intent": str(route.get("intent", "")),
                "complexity": str(route.get("complexity", "")),
                "pipeline": route.get("pipeline", "standard"),
            }
            pipeline_meta["stages"].append("routing")

        # 2. Ensure hybrid index is built
        cars = await _inventory.get()
        if self._hybrid_retriever is None or len(self._hybrid_retriever) == 0:
            await self._build_hybrid_index(cars)

        # 3. Query transformation
        search_query = query
        if self._query_transformer and route and route.get("pipeline") != "simple":
            strategies = route.get("strategies", ["expand"])
            transform_result = self._query_transformer.transform(query, strategies)
            search_queries = transform_result.get("all_queries", [query])
            search_query = search_queries[0] if search_queries else query
            pipeline_meta["stages"].append("query_transform")
            pipeline_meta["transformed_queries"] = len(search_queries)

        # 4. Hybrid retrieval
        if self._hybrid_retriever and len(self._hybrid_retriever) > 0:
            retrieve_k = route.get("retrieval_k", settings.RETRIEVER_K) if route else settings.RETRIEVER_K
            results = self._hybrid_retriever.search(search_query, top_k=retrieve_k * 2)
            pipeline_meta["stages"].append("hybrid_retrieval")
            pipeline_meta["retrieved"] = len(results)

            # 5. Re-ranking
            if self._reranker is None:
                self._reranker = _init_reranker()
            if self._reranker and results and (route is None or route.get("use_reranker", True)):
                results = self._reranker.rerank(query, results, top_n=settings.RETRIEVER_K)
                pipeline_meta["stages"].append("reranking")

            # 6. Corrective RAG (grade quality)
            if self._corrective_rag and results:
                eval_result = self._corrective_rag.evaluate_retrieval(query, results)
                if eval_result["action"] == "proceed":
                    results = eval_result.get("relevant_documents", results)
                    pipeline_meta["stages"].append("crag_pass")
                else:
                    pipeline_meta["stages"].append(f"crag_{eval_result['action']}")
                pipeline_meta["relevance_ratio"] = eval_result.get("relevance_ratio", 0)

            context = "\n".join(r["text"][:500] for r in results[:settings.RETRIEVER_K])
        else:
            # Fallback to TF-IDF
            context = await _inventory.get_context(query)
            pipeline_meta["stages"].append("tfidf_fallback")

        return context, pipeline_meta

    async def stream_response(self, user_msg: str, context: str, chat_history: list):
        """Stream LLM response — backwards compatible with v1."""
        chain = self.prompt | self.llm
        async for chunk in chain.astream({
            "input": user_msg,
            "context": context,
            "chat_history": chat_history,
        }):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield token

    async def stream_response_v2(
        self, user_msg: str, chat_history: list, session_id: str = ""
    ):
        """
        v2.0 streaming pipeline with semantic cache, guardrails, and memory.
        """
        self._ensure_v2_components()

        # Input guardrails
        if self._guardrails:
            check = self._guardrails.check_input(user_msg)
            if not check["safe"]:
                yield "I'm sorry, I can't process that request. Please rephrase your question."
                return
            user_msg = check["sanitized"]

        # Semantic cache check
        if self._semantic_cache:
            cached = self._semantic_cache.get(user_msg)
            if cached:
                yield cached
                return

        # Memory: record user message
        if self._memory and session_id:
            self._memory.add_user_message(session_id, user_msg)

        # v2.0 retrieval
        context, pipeline_meta = await self.retrieve_context_v2(user_msg)

        # Stream LLM response
        full_response = ""
        async for token in self.stream_response(user_msg, context, chat_history):
            full_response += token
            yield token

        # Post-generation: cache + memory
        if self._semantic_cache and full_response:
            self._semantic_cache.put(user_msg, full_response)

        if self._memory and session_id and full_response:
            self._memory.add_ai_message(session_id, full_response)

    async def stream_response_agent(
        self, user_msg: str, chat_history: list, session_id: str = ""
    ):
        """
        Agentic streaming pipeline with ReAct tool calling.
        Routes to the ReAct agent when the query needs tools
        (images, web specs, availability checks), otherwise
        falls back to the standard v2 pipeline.
        """
        from .agents import ReActAgent, AgentTools, should_use_agent

        self._ensure_v2_components()

        # Input guardrails
        if self._guardrails:
            check = self._guardrails.check_input(user_msg)
            if not check["safe"]:
                yield "I'm sorry, I can't process that request. Please rephrase your question."
                return
            user_msg = check["sanitized"]

        # Semantic cache check
        if self._semantic_cache:
            cached = self._semantic_cache.get(user_msg)
            if cached:
                yield cached
                return

        # Decide: agent or standard pipeline?
        if should_use_agent(user_msg):
            # Build inventory context for the agent
            context = await _inventory.get_context(user_msg)

            # Ensure hybrid index is built before passing to tools
            cars = await _inventory.get()
            if self._hybrid_retriever is None or len(self._hybrid_retriever) == 0:
                await self._build_hybrid_index(cars)

            # Initialize tools with retriever
            tools = AgentTools(
                retriever=self._hybrid_retriever,
                inventory_cache=_inventory,
            )

            # Create and run the ReAct agent
            agent = ReActAgent(
                llm=self.llm,
                tools=tools,
                inventory_context=context,
            )

            full_response = ""
            async for token in agent.run(user_msg, chat_history):
                full_response += token
                yield token

            # Cache the response
            if self._semantic_cache and full_response:
                self._semantic_cache.put(user_msg, full_response)
            if self._memory and session_id and full_response:
                self._memory.add_ai_message(session_id, full_response)
        else:
            # Standard v2 pipeline (no tool calling needed)
            async for token in self.stream_response_v2(
                user_msg, chat_history, session_id
            ):
                yield token

    def get_pipeline_info(self) -> dict:
        """Return info about which v2.0 components are active."""
        self._ensure_v2_components()
        return {
            "version": "2.0",
            "llm_provider": settings.llm_provider,
            "components": {
                "hybrid_retriever": self._hybrid_retriever is not None,
                "reranker": self._reranker is not None,
                "query_transformer": self._query_transformer is not None,
                "adaptive_router": self._adaptive_router is not None,
                "semantic_cache": self._semantic_cache is not None,
                "guardrails": self._guardrails is not None,
                "knowledge_graph": self._knowledge_graph is not None,
                "corrective_rag": self._corrective_rag is not None,
                "memory": self._memory is not None,
            },
            "config": {
                "dense_model": settings.DENSE_MODEL,
                "reranker_model": settings.RERANKER_MODEL,
                "fusion_strategy": settings.FUSION_STRATEGY,
                "chunk_strategy": settings.CHUNK_STRATEGY,
                "cache_enabled": settings.CACHE_ENABLED,
                "kg_enabled": settings.KG_ENABLED,
            },
        }


async def force_inventory_refresh() -> int:
    """Exposed so the /inventory/refresh endpoint can call it."""
    return await _inventory.force_refresh()
