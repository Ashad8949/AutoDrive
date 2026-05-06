

100% Azure Service Map
AutoDrive Cloud-Native Car Dealership
Complete Azure Resource Setup Guide — All Services Native to Microsoft Azure


What It Is
Azure OpenAI is Microsoft's hosted version of OpenAI's models — same GPT-4o, same API, but running inside Azure's datacenters. Your data never leaves Azure. It's better for compliance, offers an SLA, and integrates natively with Azure AD and Key Vault.

Step-by-Step Setup

LangChain Code — Azure OpenAI (chatbot service)


What It Is
Azure AI Search is Microsoft's cloud search platform. It now supports native vector search (replaces Pinecone) AND full-text keyword search (replaces Elasticsearch) in a single service. For this project it powers two things: semantic RAG retrieval for the chatbot, and filter-based car listings search.

Step-by-Step Setup

Create Vector Index (Python script — run once)

LangChain Integration (chatbot/ingest.py)


What It Is
Azure Event Hubs exposes a Kafka-compatible endpoint. Your existing KafkaJS (Node.js) or confluent-kafka-python code works with zero code changes. You only swap the broker URL and add SASL credentials. Event Hubs auto-scales, is fully managed, and integrates with Azure Monitor out of the box.

Step-by-Step Setup

Node.js (KafkaJS) — Zero Code Change

Python (Azure Functions consumer — Sentiment Service)
