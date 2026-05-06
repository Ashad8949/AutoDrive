"""
AutoDrive Chatbot — Data Ingestion Script
Populates the vector store with car data.

Usage:
  LOCAL  → Reads seed_data.json → FAISS index
  AZURE  → Reads PostgreSQL    → Azure AI Search

Run:
  python ingest.py
"""

from __future__ import annotations
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

from .config import settings
from langchain_core.documents import Document


def load_seed_cars() -> list[Document]:
    """Load car documents from the local JSON seed file."""
    with open(settings.SEED_DATA_PATH, "r", encoding="utf-8") as f:
        cars = json.load(f)

    docs = []
    for car in cars:
        content = (
            f"{car['year']} {car['make']} {car['model']} | "
            f"Price: ${car['price']:,} | "
            f"Mileage: {car['mileage']:,} mi | "
            f"Fuel: {car['fuel_type']} | "
            f"Location: {car['location']} | "
            f"{car['description']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"car_id": car["id"], "make": car["make"], "price": car["price"]},
        ))
    logger.info(f"Loaded {len(docs)} cars from seed_data.json")
    return docs


def load_backend_cars() -> list[Document]:
    """Fetch live car data from the backend service (production mode)."""
    import httpx

    logger.info(f"Fetching cars from backend: {settings.CARS_API_URL}")
    response = httpx.get(f"{settings.CARS_API_URL}/cars?limit=1000", timeout=30)
    response.raise_for_status()
    cars = response.json()["cars"]

    docs = []
    for car in cars:
        content = (
            f"{car['year']} {car['make']} {car['model']} | "
            f"Price: ${car['price']:,} | "
            f"Mileage: {car['mileage']:,} mi | "
            f"Fuel: {car['fuel_type']} | "
            f"Location: {car.get('location', 'N/A')} | "
            f"{car.get('description', '')}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"car_id": car["id"], "make": car["make"], "price": float(car["price"])},
        ))
    logger.info(f"Loaded {len(docs)} cars from backend API")
    return docs


def get_embeddings():
    if settings.is_azure:
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )


def ingest_to_faiss(docs: list[Document]) -> None:
    """Create FAISS index from documents and save locally."""
    from langchain_community.vectorstores import FAISS

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(settings.FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {settings.FAISS_INDEX_PATH}")


def ingest_to_azure_search(docs: list[Document]) -> None:
    """Upsert documents into Azure AI Search vector index."""
    from langchain_community.vectorstores.azuresearch import AzureSearch

    embeddings = get_embeddings()
    AzureSearch.from_documents(
        docs,
        embeddings,
        azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
        azure_search_key=settings.AZURE_SEARCH_KEY,
        index_name=settings.AZURE_SEARCH_INDEX,
    )
    logger.info(
        f"Ingested {len(docs)} cars into Azure AI Search index "
        f"'{settings.AZURE_SEARCH_INDEX}'"
    )


def main():
    logger.info("=" * 50)
    logger.info("  AutoDrive — Car Data Ingestion")
    logger.info("=" * 50)

    # Choose data source: backend API > seed file
    try:
        docs = load_backend_cars()
    except Exception as e:
        logger.warning(f"Backend fetch failed ({e}), falling back to seed_data.json")
        docs = load_seed_cars()

    # Choose vector store target
    if settings.is_azure and settings.AZURE_SEARCH_ENDPOINT:
        ingest_to_azure_search(docs)
    elif settings.has_openai:
        ingest_to_faiss(docs)
    else:
        logger.info(
            "No OpenAI/Azure keys found → using FREE TF-IDF mode.\n"
            "  ✓ No ingestion step needed!\n"
            "  ✓ Cars will be loaded automatically when the server starts.\n"
            "  ✓ Just run: python main.py"
        )

    logger.info("Ingestion complete ✓")


if __name__ == "__main__":
    main()
