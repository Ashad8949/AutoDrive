# AutoDrive Chatbot Service 🚗🤖

An intelligent, LLM-powered Retrieval-Augmented Generation (RAG) chatbot for the AutoDrive car dealership.

This service allows users to ask questions about the current car inventory, get personalized recommendations, and even book test drives directly through a conversational interface.

---

## 🏗️ Architecture

The chatbot is built on a **Dual-Mode Architecture** designed to be 100% free for local student development, while seamlessly scaling to Azure Enterprise services for production.

### System Architecture Diagram


```mermaid
architecture-beta
    group user_layer(cloud)[User Layer]
    group frontend(server)[Frontend]
    group chatbot_service(server)[Chatbot Microservice]
    group data_layer(database)[Data Layer]
    group ai_providers(cloud)[AI Providers]

    service browser(internet)[Web Browser] in user_layer
    service nextjs(server)[Next.js Web App] in frontend
    service fastapi(server)[FastAPI Server] in chatbot_service
    service history(database)[Redis Chat History] in chatbot_service
    
    service pg(database)[PostgreSQL] in data_layer
    service vector(database)[Azure AI Search / FAISS] in data_layer
    
    service openai(cloud)[Azure OpenAI / Ollama] in ai_providers

    browser:R --> L:nextjs
    nextjs:B --> T:fastapi
    
    fastapi:R --> L:history
    fastapi:R --> L:vector
    fastapi:B --> T:openai
    
    vector:B -.-> T:pg
```


### Core Stack
- **Framework**: `FastAPI` (Python)
- **RAG Orchestration**: `LangChain`
- **UI**: Standalone Vanilla HTML/JS/CSS widget with Server-Sent Events (SSE) streaming.
- **Containerization**: `Docker`

### Dual-Mode Resources
| Component | 🟢 Local Dev Mode (FREE) | 🔵 Production Azure Mode |
| :--- | :--- | :--- |
| **LLM Provider** | **Ollama** (Llama 3 locally) | Azure OpenAI (`gpt-4o`) |
| **Embeddings** | **TF-IDF** (sklearn, local) | Azure OpenAI (`text-embedding-`)|
| **Vector DB** | **FAISS** / In-Memory Array | Azure AI Search |
| **Chat History**| **In-Memory** Dictionary | Redis |
| **Data Source** | `seed_data.json` | PostgreSQL (Main App) |

---

## ✨ Features

- **Semantic Inventory Search**: Ask questions like "Show me SUVs under $40k" and get exact matches based on context.
- **Test Drive Booking Detection**: The LLM detects when a user wants to book a test drive, intercepts the intent via `[ACTION: BOOK_TEST_DRIVE <car_id>]`, and dynamically renders a booking calendar widget in the UI.
- **Real-time Streaming**: Responses stream token-by-token directly to the UI using Server-Sent Events (`/chat/stream`).
- **Conversation Memory**: Remembers the last 10 messages of context per session ID so follow-up questions work smoothly.

---

## 🚀 Local Development (100% Free)

You can run this entire service on your laptop without any API keys or paid cloud accounts.

### 1. Prerequisites
- Python 3.12+ (Conda recommended)
- [Ollama](https://ollama.com/download) installed locally.

### 2. Setup
Clone the repo and install dependencies:
```bash
conda create -n autodrive python=3.12 -y
conda activate autodrive
pip install -r requirements.txt
```

Download the local LLM model (Llama 3, ~4.7GB):
```bash
ollama pull llama3
```

### 3. Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Ensure your `.env` looks like this for free local mode:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
LLM_TEMPERATURE=0.3
RETRIEVER_K=5
PORT=8002
```
*Note: Do NOT set `OPENAI_API_KEY` or `AZURE_*` keys if you want to use the free local mode.*

### 4. Run the Server
```bash
python main.py
```

### 5. Access the App
- **Chat UI**: `http://localhost:8002/`
- **API Swagger Docs**: `http://localhost:8002/docs`
- **Health Check**: `http://localhost:8002/health`

---

## ☁️ Production Deployment (Azure)

When you are ready to deploy this to production, you do **not** need to change any code. The app automatically detects Azure environment variables and switches resources.

### 1. Required Azure Variables
In your production `.env` or Azure Container Apps secrets, supply:
```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT="https://..."
AZURE_OPENAI_KEY="sk-..."
AZURE_SEARCH_ENDPOINT="https://..."
AZURE_SEARCH_KEY="..."
AZURE_SEARCH_INDEX="cars-index"
REDIS_URL="redis://..."
DATABASE_URL="postgresql://..." # For data ingestion
```

### 2. Deployment Steps
Please refer to the `azure_deployment_guide.md` file in this directory for a step-by-step tutorial on:
1. Creating Azure OpenAI & AI Search resources.
2. Building and pushing the Docker image to Azure Container Registry.
3. Deploying the image to Azure Container Apps.

---

## 📁 Project Structure

```text
chatbot/
├── main.py                 # FastAPI application, UI routes, and Streaming endpoints
├── rag.py                  # LangChain logic, System Prompt, and Embeddings/LLM factory
├── history.py              # In-memory and Redis chat history management
├── ingest.py               # Script to load data into FAISS or Azure AI Search
├── config.py               # Auto-detects Local vs Azure mode based on .env
├── seed_data.json          # 20 realistic cars for local testing
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage build for Azure Container Apps
├── azure_deployment_guide.md # Deployment guide
└── static/
    └── index.html          # Beautiful Vanilla JS/HTML Chat UI widget
```

---

## 🧑‍💻 Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the HTML Chat UI |
| `POST` | `/chat/stream` | Main endpoint. Streams SSE tokens for real-time chat. |
| `POST` | `/chat` | Non-streaming endpoint for simple testing. |
| `GET` | `/health` | Liveness probe for Kubernetes/Azure Container Apps. |
| `GET` | `/ready` | Readiness probe (verifies RAG engine initialization). |
