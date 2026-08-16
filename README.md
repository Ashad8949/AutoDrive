# AutoDrive Chatbot Service 🚗🤖 (v2.0)

An intelligent, LLM-powered Retrieval-Augmented Generation (RAG) chatbot for the AutoDrive car dealership.

This service allows users to ask questions about the current car inventory, get personalized recommendations, and even book test drives directly through a conversational interface.

---

## 🏗️ Architecture (v2.0)

The chatbot has been upgraded to a **state-of-the-art Agentic RAG v2.0** architecture, featuring hybrid search, re-ranking, and a corrective feedback loop (CRAG) to guarantee zero hallucination.

### Key v2.0 Upgrades
- **Hybrid Retrieval**: Combines Dense Vector Search (FAISS + sentence-transformers) with Sparse Keyword Search (BM25) using **Reciprocal Rank Fusion (RRF)** for maximum recall.
- **Cross-Encoder Re-Ranking**: Uses `ms-marco-MiniLM` to re-rank the retrieved documents for absolute precision.
- **Agentic CRAG**: Corrective RAG state machine (built with LangGraph concepts) that grades document relevance and automatically rewrites/expands queries if the initial retrieval is poor.
- **Advanced Chunking**: 5 different chunking strategies including contextual and parent-document retrieval.
- **Knowledge Graph (GraphRAG)**: `NetworkX`-powered entity extraction connecting cars, brands, and features for complex multi-hop queries.
- **Production Hardening**: Semantic response caching to reduce LLM calls, and safety guardrails against prompt injection and PII leaks.
- **Evaluation Pipeline**: Built-in SQuAD-style metrics (Exact Match, F1, NDCG, MRR) with statistical significance testing.

### Core Stack
- **Framework**: `FastAPI` (Python)
- **RAG Orchestration**: `LangChain`
- **Models**: `LLaMA 3.3-70B` (via Groq API)
- **Vector DB / Graph**: `FAISS` / `NetworkX`
- **Deployment**: `Docker` + `Terraform` on Azure

---

## ✨ Features

- **Semantic Inventory Search**: Ask questions like "Show me SUVs under 40k" and get exact matches based on context.
- **Zero Hallucination Guarantee**: Grounding checkers verify all LLM output against the retrieved inventory data before responding.
- **Test Drive Booking Detection**: The LLM intercepts booking intents via `[ACTION: BOOK_TEST_DRIVE <car_id>]` and dynamically renders a calendar widget.
- **Real-time Streaming**: Responses stream token-by-token directly to the UI using Server-Sent Events (`/chat/v2/stream`).
- **Conversation Memory**: Remembers context, user preferences, and mentioned entities across the session.

---

## 🚀 Local Development

You can run this entire service on your laptop.

### 1. Prerequisites
- Python 3.11 or 3.12 (Conda recommended — *Note: Python 3.14 is not fully supported by `scikit-learn` binaries yet*)
- A Groq API Key

### 2. Setup
Clone the repo and install dependencies:
```bash
conda create -n autodrive python=3.12 -y
conda activate autodrive
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Ensure your `.env` looks like this:
```env
GROQ_API_KEY=gsk_your_actual_key_here
LLM_PROVIDER=groq
PORT=8002
```

### 4. Run the Server
```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8002 --reload
```

### 5. Access the App
- **Chat UI**: `http://localhost:8002/`
- **API Swagger Docs**: `http://localhost:8002/docs`
- **Pipeline Info**: `http://localhost:8002/pipeline/info`

---

## 📁 Project Structure (v2.0)

```text
AutoDrive/
├── src/
│   ├── main.py                 # FastAPI application and v2 streaming endpoints
│   ├── rag.py                  # Main RAGEngine v2.0 integrator
│   ├── config.py               # Centralized settings and feature flags
│   ├── agents/                 # CRAG, Self-RAG, and LangGraph definitions
│   ├── cache/                  # Semantic caching
│   ├── chunking/               # Late chunking, contextual, and 5 standard strategies
│   ├── evaluation/             # Evaluation metrics and statistical tests
│   ├── generation/             # Citations and grounding checkers
│   ├── guardrails/             # Security and PII detection
│   ├── knowledge/              # Knowledge Graph (NetworkX) builder and GraphRAG
│   ├── memory/                 # Advanced session memory and context compression
│   ├── query/                  # Intent classification and adaptive routing
│   └── retrieval/              # Hybrid retrieval, FAISS, BM25, and Rerankers
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies (includes FAISS, Sentence-Transformers)
└── verify_v2.py                # Verification script to test backend endpoints
```
