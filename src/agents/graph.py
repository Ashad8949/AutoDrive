"""
AutoDrive RAG v2.0 — LangGraph Agentic RAG
Implements a full agentic RAG pipeline using LangGraph's StateGraph.

The agent follows a sophisticated flow:
  1. Analyze query → classify intent and complexity
  2. Route to appropriate pipeline (simple/standard/agentic)
  3. Transform query (HyDE, multi-query, etc.)
  4. Retrieve relevant documents
  5. Grade document relevance (CRAG)
  6. Generate answer
  7. Self-reflect on answer quality (Self-RAG)
  8. Return or retry based on reflection

This is the orchestration layer that ties together all the individual
RAG components into a coherent, self-correcting system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("chatbot.agents.graph")


@dataclass
class RAGState:
    """
    State object that flows through the LangGraph pipeline.

    Contains all intermediate data from query analysis through
    final answer generation, enabling inspection and debugging.
    """

    # Input
    query: str = ""
    chat_history: list = field(default_factory=list)
    session_id: str = ""

    # Query understanding
    intent: str = ""
    complexity: str = ""
    pipeline: str = "standard"

    # Query transformation
    transformed_queries: list[str] = field(default_factory=list)
    hyde_hypothesis: str = ""

    # Retrieval
    retrieved_documents: list[dict] = field(default_factory=list)
    graded_documents: list[dict] = field(default_factory=list)
    relevant_documents: list[dict] = field(default_factory=list)
    retrieval_source: str = "hybrid"  # hybrid, dense, sparse, web

    # Generation
    context: str = ""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)

    # Reflection
    reflection_results: dict = field(default_factory=dict)
    is_grounded: bool = True

    # Control flow
    retry_count: int = 0
    max_retries: int = 2
    error: Optional[str] = None
    should_retrieve: bool = True

    def to_dict(self) -> dict:
        """Serialize state for logging/debugging."""
        return {
            "query": self.query,
            "intent": self.intent,
            "complexity": self.complexity,
            "pipeline": self.pipeline,
            "num_retrieved": len(self.retrieved_documents),
            "num_relevant": len(self.relevant_documents),
            "answer_length": len(self.answer),
            "is_grounded": self.is_grounded,
            "retry_count": self.retry_count,
        }


class RAGGraph:
    """
    LangGraph-style agentic RAG pipeline.

    Orchestrates the full RAG flow with conditional routing,
    self-correction, and multiple retrieval strategies.

    This class implements the graph logic directly (without requiring
    the langgraph library) to minimize dependencies, but follows the
    same StateGraph pattern.

    Attributes:
        retriever: HybridRetriever for document retrieval.
        reranker: CrossEncoderReranker for result re-ranking.
        query_transformer: QueryTransformer for query optimization.
        adaptive_router: AdaptiveRouter for pipeline selection.
        corrective_rag: CorrectiveRAG for retrieval evaluation.
        self_rag: SelfRAG for answer reflection.
        llm: LLM for generation.
    """

    def __init__(
        self,
        retriever=None,
        reranker=None,
        query_transformer=None,
        adaptive_router=None,
        corrective_rag=None,
        self_rag=None,
        llm=None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.adaptive_router = adaptive_router
        self.corrective_rag = corrective_rag
        self.self_rag = self_rag
        self.llm = llm

    async def invoke(self, query: str, chat_history: list = None, **kwargs) -> RAGState:
        """
        Run the full agentic RAG pipeline.

        Args:
            query: User's query.
            chat_history: Previous conversation messages.
            **kwargs: Additional state overrides.

        Returns:
            Final RAGState with answer and all intermediate results.
        """
        state = RAGState(
            query=query,
            chat_history=chat_history or [],
            **kwargs,
        )

        try:
            # Step 1: Analyze and route query
            state = self._analyze_query(state)
            logger.info(
                "Graph: query='%s' → intent=%s, complexity=%s, pipeline=%s",
                query[:50],
                state.intent,
                state.complexity,
                state.pipeline,
            )

            # Step 2: Check if retrieval is needed
            if self.self_rag:
                retrieval_check = self.self_rag.should_retrieve(query)
                state.should_retrieve = retrieval_check.passed
                if not state.should_retrieve:
                    logger.info("Graph: Skipping retrieval (not needed)")
                    state = await self._generate_direct(state)
                    return state

            # Step 3: Transform query
            if state.pipeline != "simple":
                state = self._transform_query(state)

            # Step 4: Retrieve with retry loop
            while state.retry_count <= state.max_retries:
                state = self._retrieve(state)
                state = self._grade_documents(state)

                # Check if we have enough relevant documents
                if state.relevant_documents or state.retry_count >= state.max_retries:
                    break

                # Retry with rewritten query
                logger.info(
                    "Graph: Retry %d — rewriting query", state.retry_count + 1
                )
                if self.corrective_rag:
                    rewritten = self.corrective_rag.rewrite_query(state.query)
                    state.transformed_queries = [rewritten]
                state.retry_count += 1

            # Step 5: If still no good docs, try web search
            if not state.relevant_documents and self.corrective_rag:
                logger.info("Graph: Falling back to web search")
                web_results = self.corrective_rag.web_search_fallback(state.query)
                if web_results:
                    state.relevant_documents = web_results
                    state.retrieval_source = "web"

            # Step 6: Build context and generate
            state = self._build_context(state)
            state = await self._generate(state)

            # Step 7: Self-reflect on answer
            if self.self_rag and state.answer:
                state = self._reflect(state)

                # Regenerate if answer is not grounded
                if not state.is_grounded and state.retry_count < state.max_retries:
                    logger.info("Graph: Regenerating (answer not grounded)")
                    state.retry_count += 1
                    state = await self._generate(state)

        except Exception as e:
            logger.error("Graph execution error: %s", e)
            state.error = str(e)
            state.answer = (
                "I'm sorry, I encountered an error processing your request. "
                "Please try rephrasing your question."
            )

        return state

    def _analyze_query(self, state: RAGState) -> RAGState:
        """Step 1: Analyze query intent and complexity."""
        if self.adaptive_router:
            route = self.adaptive_router.route(state.query)
            state.intent = route["intent"].value if hasattr(route["intent"], "value") else str(route["intent"])
            state.complexity = route["complexity"].value if hasattr(route["complexity"], "value") else str(route["complexity"])
            state.pipeline = route["pipeline"]
        else:
            state.intent = "inventory_search"
            state.complexity = "standard"
            state.pipeline = "standard"
        return state

    def _transform_query(self, state: RAGState) -> RAGState:
        """Step 3: Transform query for better retrieval."""
        if not self.query_transformer:
            state.transformed_queries = [state.query]
            return state

        strategies = ["expand"]
        if state.pipeline == "agentic":
            strategies.extend(["multi_query", "hyde"])

        result = self.query_transformer.transform(state.query, strategies)
        state.transformed_queries = result.get("all_queries", [state.query])
        state.hyde_hypothesis = result.get("hyde", "")

        logger.debug(
            "Graph: %d transformed queries generated",
            len(state.transformed_queries),
        )
        return state

    def _retrieve(self, state: RAGState) -> RAGState:
        """Step 4: Retrieve documents using all transformed queries."""
        if not self.retriever:
            return state

        all_results = []
        seen_texts = set()

        queries = state.transformed_queries or [state.query]
        for q in queries[:5]:  # Cap at 5 queries
            results = self.retriever.search(q, top_k=10)
            for r in results:
                if r["text"] not in seen_texts:
                    seen_texts.add(r["text"])
                    all_results.append(r)

        # Re-rank if available
        if self.reranker and all_results:
            all_results = self.reranker.rerank(
                state.query, all_results, top_n=10
            )

        state.retrieved_documents = all_results
        logger.debug("Graph: Retrieved %d unique documents", len(all_results))
        return state

    def _grade_documents(self, state: RAGState) -> RAGState:
        """Step 5: Grade retrieved documents for relevance."""
        if not self.corrective_rag or not state.retrieved_documents:
            state.relevant_documents = state.retrieved_documents
            return state

        eval_result = self.corrective_rag.evaluate_retrieval(
            state.query, state.retrieved_documents
        )
        state.graded_documents = eval_result["graded_documents"]
        state.relevant_documents = eval_result["relevant_documents"]
        return state

    def _build_context(self, state: RAGState) -> RAGState:
        """Step 6: Build context string from relevant documents."""
        docs = state.relevant_documents or state.retrieved_documents
        if docs:
            state.context = "\n\n---\n\n".join(
                d["text"][:500] for d in docs[:5]
            )
        else:
            state.context = "No relevant documents found."
        return state

    async def _generate(self, state: RAGState) -> RAGState:
        """Step 7: Generate answer using LLM."""
        if not self.llm:
            state.answer = state.context
            return state

        prompt = (
            f"Based on the following context, answer the user's question.\n\n"
            f"Context:\n{state.context}\n\n"
            f"Question: {state.query}\n\n"
            f"Answer concisely and specifically. If the context doesn't contain "
            f"enough information, say so honestly."
        )

        try:
            response = self.llm.invoke(prompt)
            state.answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as e:
            logger.error("Generation failed: %s", e)
            state.answer = "I'm sorry, I couldn't generate a response."
            state.error = str(e)

        return state

    async def _generate_direct(self, state: RAGState) -> RAGState:
        """Generate answer without retrieval (for simple queries)."""
        if not self.llm:
            state.answer = "I'm an AutoDrive assistant. How can I help you?"
            return state

        try:
            response = self.llm.invoke(state.query)
            state.answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as e:
            state.answer = "Hello! I'm AutoDrive AI. How can I help you find your perfect car?"
            state.error = str(e)

        return state

    def _reflect(self, state: RAGState) -> RAGState:
        """Step 8: Self-reflect on answer quality."""
        if not self.self_rag:
            return state

        reflection = self.self_rag.full_reflection(
            query=state.query,
            documents=state.relevant_documents or state.retrieved_documents,
            answer=state.answer,
        )

        state.reflection_results = reflection
        state.is_grounded = reflection["overall_passed"]

        if not state.is_grounded:
            logger.warning(
                "Graph: Self-reflection failed (score=%.2f, action=%s)",
                reflection["overall_score"],
                reflection["action"],
            )

        return state

    def __repr__(self) -> str:
        components = []
        if self.retriever:
            components.append("retriever")
        if self.reranker:
            components.append("reranker")
        if self.query_transformer:
            components.append("query_transform")
        if self.corrective_rag:
            components.append("crag")
        if self.self_rag:
            components.append("self_rag")
        return f"RAGGraph(components=[{', '.join(components)}])"
