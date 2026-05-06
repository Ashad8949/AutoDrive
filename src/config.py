"""
AutoDrive Chatbot — Configuration Module
Auto-detects LOCAL vs AZURE mode based on environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)  # repo root (one level above src/)


class Settings:
    """Central configuration. Reads from .env or system environment."""

    # ── Azure OpenAI ────────────────────────────────────────────────
    AZURE_OPENAI_KEY: str | None = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

    # ── Azure AI Search ─────────────────────────────────────────────
    AZURE_SEARCH_ENDPOINT: str | None = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY: str | None = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "car-dealership")

    # ── Groq (free, fast — fallback when Azure quota not available) ───
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Local / OpenAI fallback ─────────────────────────────────────
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # ── Database / Cache ────────────────────────────────────────────
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    REDIS_URL: str | None = os.getenv("REDIS_URL")

    # ── Backend service URL (for fetching live car data) ────────────
    CARS_API_URL: str = os.getenv("CARS_API_URL", "http://localhost:8000")

    # ── CORS ────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g.:
    # ALLOWED_ORIGINS=https://autodrive-frontend.azurestaticapps.net,http://localhost:3000
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

    # ── General ─────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "5"))
    PORT: int = int(os.getenv("PORT", "8002"))

    # paths resolve relative to repo root, not src/
    SEED_DATA_PATH: str = os.getenv(
        "SEED_DATA_PATH",
        os.path.join(_ROOT, "seed_data.json"),
    )
    FAISS_INDEX_PATH: str = os.getenv(
        "FAISS_INDEX_PATH",
        os.path.join(_ROOT, "faiss_index"),
    )

    @property
    def is_azure(self) -> bool:
        return bool(self.AZURE_OPENAI_KEY and self.AZURE_OPENAI_ENDPOINT)

    @property
    def has_redis(self) -> bool:
        return bool(self.REDIS_URL)

    @property
    def has_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def has_groq(self) -> bool:
        return bool(self.GROQ_API_KEY)

    @property
    def llm_provider(self) -> str:
        if self.has_groq:
            return "groq"
        if self.is_azure:
            return "azure"
        if self.has_openai:
            return "openai"
        return "ollama"


settings = Settings()
