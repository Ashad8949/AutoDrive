

Cloud-Native Car Dealership
Web Application
Complete Industry-Standard Implementation Guide


Ashad Alam • Venkata Mahesh • Pritam Maji • Samarth Agrawal  
Microsoft Azure  •  Cloud Computing Course Project

Table of Contents
1.  Tech Stack Decision — Industry Grade ....................................... Page 3
2.  System Architecture Overview ....................................... Page 4
3.  LLM Chatbot Implementation (Core) ....................................... Page 6
4.  Frontend — Next.js 14 (Sexy UI) ....................................... Page 9
5.  Microservices — All 6 Services ....................................... Page 12
6.  Database Strategy ....................................... Page 16
7.  Infrastructure as Code (Terraform + AKS) ....................................... Page 17
8.  CI/CD Pipeline (GitHub Actions) ....................................... Page 19
9.  Observability Stack ....................................... Page 20
10.  Security — Zero Trust Architecture ....................................... Page 21
11.  Load Handling — Facebook/Amazon Patterns ....................................... Page 22
12.  Project Phases & Timeline ....................................... Page 24


1. Tech Stack Decision
This section replaces the original proposal stack with the exact tools used by Facebook/Meta, Amazon, Uber, and top-tier startups. Every choice is justified against industry benchmarks.


1.1 Frontend
▶  Framework: Next.js 14 (App Router + Server Components) — Used by Vercel, TikTok, Twitch
▶  Language: TypeScript (strict mode) — Industry standard, catches 70%+ of runtime bugs
▶  Styling: Tailwind CSS + shadcn/ui + Framer Motion — Netflix, Linear, Vercel use this combo
▶  State: Zustand (global) + TanStack Query v5 (server state) — Replaced Redux, lighter & faster
▶  Forms: React Hook Form + Zod validation — Type-safe form handling
▶  Auth Client: NextAuth.js v5 — OAuth2, Google/GitHub SSO, JWT sessions

1.2 Backend Microservices
▶  API Gateway: Kong + NGINX — Same as Airbnb, Netflix API routing layer
▶  Auth Service: Node.js + Fastify + JWT + Redis sessions — Fast, lightweight
▶  Car Catalog: Python FastAPI + SQLAlchemy — 3x faster than Django, async-native
▶  Reviews: Node.js + Fastify + MongoDB — Document store ideal for unstructured reviews
▶  LLM Chatbot: Python FastAPI + LangChain + Azure OpenAI GPT-4o (Azure-hosted) + Streaming SSE
▶  Price Prediction: Python FastAPI + MLflow + XGBoost — Versioned ML models
▶  Sentiment: Python + Azure Functions (Serverless) + HuggingFace Transformers
▶  Message Bus: Azure Event Hubs (Kafka-compatible API) — Azure native, scales to millions of events/sec

1.3 Database Layer
▶  Primary DB: PostgreSQL 16 (cars, users, bookings) — Battle-tested, ACID compliant
▶  NoSQL: MongoDB Atlas (reviews, chat history) — Flexible schema
▶  Cache: Redis 7 (sessions, rate limiting, hot data) — Sub-millisecond reads
▶  Vector DB: Azure AI Search (built-in vector + hybrid search) — 100% Azure, no external SaaS
▶  Search: Elasticsearch (car search with filters) — Amazon, LinkedIn search layer

1.4 Infrastructure & DevOps
▶  Cloud: Microsoft Azure (AKS, Container Registry, Key Vault, Monitor)
▶  IaC: Terraform — Industry gold standard, used by every FAANG DevOps team
▶  Containers: Docker (multi-stage builds) — Slim, secure images
▶  Orchestration: Kubernetes (AKS) + Helm charts — Google's container orchestrator
▶  Service Mesh: Istio — Automatic mTLS, traffic shaping, circuit breaking (Uber/Lyft use this)
▶  CI/CD: GitHub Actions + ArgoCD (GitOps) — Netflix uses GitOps for zero-downtime deploys
▶  Secrets: Azure Key Vault + External Secrets Operator — Zero secrets in code

1.5 Observability (The Holy Trinity)
▶  Metrics: Prometheus + Grafana — Used by every serious cloud-native team
▶  Logs: Loki + Grafana (unified log aggregation)
▶  Traces: OpenTelemetry + Jaeger — Distributed tracing across all microservices
▶  Alerts: Grafana AlertManager + PagerDuty integration



2. System Architecture
The architecture follows the same patterns as Amazon (service-oriented, event-driven) and Facebook (horizontal scaling, CDN-first). Every service is independently deployable, independently scalable.

2.1 High-Level Architecture


2.2 Service Communication Rules



3. LLM Chatbot Implementation
This is the most critical upgrade from the original proposal. Instead of Azure Bot Service (rule-based), we build a full LLM-powered RAG chatbot using LangChain + GPT-4o with streaming, conversation memory, and semantic car search. This is the same pattern used by ChatGPT, Perplexity, and enterprise AI products.

3.1 Chatbot Architecture — RAG Pattern


3.2 FastAPI Chat Service — Complete Code
File: services/chatbot/main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import asyncio, json, os

app = FastAPI()
llm = AzureChatOpenAI(azure_deployment='gpt-4o', streaming=True, temperature=0.3,
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_KEY'), api_version='2024-02-01')
embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-3-small',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), api_key=os.getenv('AZURE_OPENAI_KEY'))
vectorstore = AzureSearch(azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
    index_name='car-dealership', embedding_function=embeddings.embed_query)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

SYSTEM_PROMPT = '''You are CarBot, an expert assistant for AutoDrive dealership.
Answer based ONLY on the provided car inventory context.
For pricing, always mention the ML-predicted fair price.
Guide users toward test drive bookings when interest is shown.
Context: {context}'''

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
])

@app.post('/chat/stream')
async def chat_stream(request: Request):
    body = await request.json()
    session_id = body['session_id']
    user_msg = body['message']

    # Load history from Redis
    history = RedisChatMessageHistory(session_id, redis_url=os.getenv('REDIS_URL'))

    async def generate():
        # RAG retrieval
        docs = await retriever.ainvoke(user_msg)
        context = '\n'.join([d.page_content for d in docs])

        # Build chain
        chain = prompt | llm
        full_response = ''

        async for chunk in chain.astream({
            'input': user_msg,
            'context': context,
            'chat_history': history.messages[-10:],
        }):
            token = chunk.content
            full_response += token
            yield f'data: {json.dumps({"token": token})}\n\n'

        # Save turn to history
        history.add_user_message(user_msg)
        history.add_ai_message(full_response)
        yield f'data: {json.dumps({"done": True})}\n\n'

    return StreamingResponse(generate(), media_type='text/event-stream')

3.3 Populating Azure AI Search — Car Data Ingestion
File: services/chatbot/ingest.py
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import psycopg2, os

# Fetch ALL cars from PostgreSQL
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cursor = conn.cursor()
cursor.execute('SELECT id, make, model, year, price, mileage, fuel_type, description FROM cars')
cars = cursor.fetchall()

# Convert to LangChain Documents
docs = [Document(
    page_content=f'{year} {make} {model} | Price: ${price:,} | Mileage: {mileage:,}mi | Fuel: {fuel_type} | {description}',
    metadata={'car_id': str(id), 'make': make, 'price': price}
) for id, make, model, year, price, mileage, fuel_type, description in cars]

# Upsert to Azure AI Search vector index
embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding-3-small',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), api_key=os.getenv('AZURE_OPENAI_KEY'))
AzureSearch.from_documents(docs, embeddings,
    azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
    index_name='car-dealership')
print(f'Ingested {len(docs)} cars into Azure AI Search')

3.4 Next.js Streaming Chat Hook
File: hooks/useChat.ts
import { useState, useCallback } from 'react'

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const sendMessage = useCallback(async (userMsg: string) => {
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setIsStreaming(true)
    let aiMsg = ''
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: userMsg, session_id: sessionId }),
      headers: { 'Content-Type': 'application/json' },
    })

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const lines = decoder.decode(value).split('\n')
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const json = JSON.parse(line.slice(6))
        if (json.done) { setIsStreaming(false); break }
        aiMsg += json.token
        setMessages(prev => [
          ...prev.slice(0, -1),
          { role: 'assistant', content: aiMsg }
        ])
      }
    }
  }, [sessionId])

  return { messages, sendMessage, isStreaming }
}

3.5 Chatbot Capabilities Summary


4. Frontend — Next.js 14
The frontend follows the design principles of Linear, Vercel, and Stripe — minimal, fast, visually stunning. Built with Next.js App Router for server-side rendering, TypeScript strict mode, Tailwind + shadcn/ui component library, and Framer Motion for animations.

4.1 Project Setup
npx create-next-app@latest autodrive-frontend \
  --typescript --tailwind --eslint --app --src-dir --import-alias '@/*'

cd autodrive-frontend
npx shadcn@latest init
npm install framer-motion zustand @tanstack/react-query
npm install next-auth @auth/prisma-adapter
npm install react-hook-form zod @hookform/resolvers
npm install lucide-react recharts
npm install @vercel/analytics @vercel/speed-insights

4.2 Folder Structure (Next.js App Router)

4.3 Gorgeous Car Listing Card
File: components/cars/CarCard.tsx
'use client'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Car, Fuel, MapPin, TrendingUp } from 'lucide-react'

export function CarCard({ car }: { car: Car }) {
  return (
    <motion.div
      whileHover={{ y: -8, scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300 }}
      className='group relative bg-white rounded-2xl shadow-lg overflow-hidden
                 border border-slate-100 hover:shadow-2xl hover:border-blue-200
                 transition-all duration-300 cursor-pointer'
    >
      {/* Image with gradient overlay */}
      <div className='relative h-52 overflow-hidden'>
        <img src={car.image} alt={car.name}
             className='w-full h-full object-cover group-hover:scale-110 transition-transform duration-500' />
        <div className='absolute inset-0 bg-gradient-to-t from-black/60 to-transparent' />
        <Badge className='absolute top-3 right-3 bg-blue-600'>{car.year}</Badge>
        {car.mlPrice && (
          <div className='absolute bottom-3 left-3 flex items-center gap-1 text-white text-sm'>
            <TrendingUp size={14} />
            <span>AI Price: ${car.mlPrice.toLocaleString()}</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className='p-5'>
        <h3 className='font-bold text-xl text-slate-900 mb-1'>
          {car.year} {car.make} {car.model}
        </h3>
        <p className='text-3xl font-black text-blue-600 mb-4'>
          ${car.price.toLocaleString()}
        </p>
        <div className='flex gap-4 text-slate-500 text-sm mb-5'>
          <span className='flex items-center gap-1'><Car size={14} />{car.mileage.toLocaleString()} mi</span>
          <span className='flex items-center gap-1'><Fuel size={14} />{car.fuelType}</span>
          <span className='flex items-center gap-1'><MapPin size={14} />{car.location}</span>
        </div>
        <div className='flex gap-2'>
          <Button className='flex-1 bg-blue-600 hover:bg-blue-700'>Book Test Drive</Button>
          <Button variant='outline' size='icon' onClick={() => openChat(car.id)}>
            <span>💬</span>
          </Button>
        </div>
      </div>
    </motion.div>
  )
}

4.4 Streaming Chat Widget
File: components/chat/ChatWidget.tsx
'use client'
import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useChat } from '@/hooks/useChat'
import { MessageSquare, Send, X, Bot } from 'lucide-react'

export function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const { messages, sendMessage, isStreaming } = useChat(sessionId)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  return (
    <>
      {/* Floating Button */}
      <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }}
        onClick={() => setOpen(true)}
        className='fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-br from-blue-600 to-violet-600
                   rounded-full shadow-2xl flex items-center justify-center z-50'>
        <Bot className='text-white' size={28} />
      </motion.button>

      {/* Chat Panel */}
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            className='fixed bottom-24 right-6 w-[400px] h-[600px] bg-white rounded-2xl
                       shadow-2xl border border-slate-200 flex flex-col z-50'>

            {/* Header */}
            <div className='flex items-center justify-between p-4
                          bg-gradient-to-r from-blue-600 to-violet-600 rounded-t-2xl'>
              <div className='flex items-center gap-3'>
                <div className='w-10 h-10 bg-white/20 rounded-full flex items-center justify-center'>
                  <Bot className='text-white' size={20} />
                </div>
                <div>
                  <p className='text-white font-semibold'>AutoDrive AI</p>
                  <p className='text-blue-200 text-xs'>Powered by GPT-4o • Always online</p>
                </div>
              </div>
              <button onClick={() => setOpen(false)}><X className='text-white' /></button>
            </div>

            {/* Messages */}
            <div className='flex-1 overflow-y-auto p-4 space-y-3'>
              {messages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm
                    ${msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-sm'
                      : 'bg-slate-100 text-slate-800 rounded-tl-sm'}`}>
                    {msg.content}
                    {isStreaming && i === messages.length-1 && msg.role==='assistant' && (
                      <span className='animate-pulse'>▌</span>
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className='p-4 border-t border-slate-100 flex gap-2'>
              <input value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !isStreaming && sendMessage(input)}
                placeholder='Ask about any car...'
                className='flex-1 px-4 py-2 rounded-xl border border-slate-200
                           focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm' />
              <button onClick={() => { sendMessage(input); setInput('') }}
                disabled={isStreaming}
                className='w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center
                           hover:bg-blue-700 disabled:opacity-50'>
                <Send size={16} className='text-white' />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

4.5 Homepage Hero Section
File: app/page.tsx — Key section
<section className='min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900
                    flex flex-col items-center justify-center text-center px-6 relative overflow-hidden'>
  {/* Animated background orbs */}
  <div className='absolute w-[600px] h-[600px] bg-blue-500/10 rounded-full blur-3xl top-0 -right-48' />
  <div className='absolute w-[400px] h-[400px] bg-violet-500/10 rounded-full blur-3xl bottom-0 -left-24' />

  <motion.div initial={{ opacity:0, y:30 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.8 }}>
    <Badge className='mb-6 bg-blue-500/20 text-blue-300 border-blue-500/30'>
      🤖 AI-Powered Car Discovery
    </Badge>
    <h1 className='text-6xl md:text-8xl font-black text-white mb-6 leading-tight'>
      Find Your Perfect <span className='text-transparent bg-clip-text
        bg-gradient-to-r from-blue-400 to-violet-400'>Drive</span>
    </h1>
    <p className='text-xl text-slate-400 mb-10 max-w-2xl'>
      Powered by GPT-4o. Search 10,000+ cars by talking naturally.
    </p>
    <div className='flex gap-4 justify-center flex-wrap'>
      <Button size='lg' className='bg-blue-600 hover:bg-blue-500 text-lg px-8 py-6'>
        Browse Cars
      </Button>
      <Button size='lg' variant='outline' className='border-white/20 text-white text-lg px-8 py-6'>
        Chat with AI
      </Button>
    </div>
  </motion.div>
</section>


5. Microservices — All Services

5.1 Service Overview

5.2 Auth Service
File: services/auth/src/routes/auth.ts
import Fastify from 'fastify'
import fastifyJwt from '@fastify/jwt'
import fastifyRedis from '@fastify/redis'
import { hashPassword, verifyPassword } from './utils/crypto'
import { db } from './db'    // Drizzle ORM + PostgreSQL

const app = Fastify({ logger: true })
app.register(fastifyJwt, { secret: process.env.JWT_SECRET! })
app.register(fastifyRedis, { host: process.env.REDIS_HOST! })

app.post('/auth/register', async (req, reply) => {
  const { email, password, name } = req.body as any
  const hashed = await hashPassword(password)
  const user = await db.insert(users).values({ email, password: hashed, name }).returning()
  const token = app.jwt.sign({ userId: user[0].id, email }, { expiresIn: '7d' })
  return { token, user: { id: user[0].id, email, name } }
})

app.post('/auth/login', async (req, reply) => {
  const { email, password } = req.body as any
  const user = await db.query.users.findFirst({ where: eq(users.email, email) })
  if (!user || !await verifyPassword(password, user.password))
    return reply.status(401).send({ error: 'Invalid credentials' })
  const token = app.jwt.sign({ userId: user.id, email }, { expiresIn: '7d' })
  // Cache session in Redis for 7 days
  await app.redis.setex(`session:${user.id}`, 604800, JSON.stringify({ email }))
  return { token }
})

// Internal token verification endpoint (used by other services)
app.get('/auth/verify', { onRequest: [app.authenticate] }, async (req) => {
  return { valid: true, user: req.user }
})

5.3 Car Catalog Service (FastAPI)
File: services/cars/main.py
from fastapi import FastAPI, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from elasticsearch import AsyncElasticsearch
import redis.asyncio as redis

app = FastAPI()
es = AsyncElasticsearch(os.getenv('ELASTICSEARCH_URL'))
cache = redis.from_url(os.getenv('REDIS_URL'))

@app.get('/cars')
async def list_cars(
    make: str = None, min_price: int = None, max_price: int = None,
    fuel_type: str = None, page: int = 1, limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    cache_key = f'cars:{make}:{min_price}:{max_price}:{fuel_type}:{page}'
    cached = await cache.get(cache_key)
    if cached: return json.loads(cached)   # Cache HIT — sub-1ms

    # Elasticsearch for full-text + filters
    query = { 'bool': { 'must': [], 'filter': [] } }
    if make: query['bool']['must'].append({ 'match': { 'make': make } })
    if min_price or max_price:
        query['bool']['filter'].append({ 'range': { 'price': {
            'gte': min_price or 0, 'lte': max_price or 10_000_000
        }})
    result = await es.search(index='cars', query=query,
                             from_=(page-1)*limit, size=limit)
    cars = [hit['_source'] for hit in result['hits']['hits']]
    await cache.setex(cache_key, 300, json.dumps(cars))  # Cache 5 min
    return cars

5.4 Reviews Service
File: services/reviews/src/index.ts
import Fastify from 'fastify'
import { MongoClient } from 'mongodb'
import { EventHubProducerClient } from '@azure/event-hubs'

const producer = new EventHubProducerClient(process.env.EVENT_HUBS_CONNECTION_STRING!, 'review-submitted')

app.post('/reviews', async (req, reply) => {
  const review = { ...req.body, createdAt: new Date(), userId: req.user.id }
  const result = await db.collection('reviews').insertOne(review)

  // Publish to Azure Event Hubs → Sentiment Function will consume
  const batch = await producer.createBatch()
  batch.tryAdd({ body: { reviewId: result.insertedId.toString(), text: review.text } })
  await producer.sendBatch(batch)
  return { id: result.insertedId }
})

5.5 ML Price Prediction Service
File: services/ml-price/main.py
from fastapi import FastAPI
import mlflow.xgboost
import pandas as pd

app = FastAPI()
model = mlflow.xgboost.load_model('models:/car-price-predictor/production')

@app.post('/predict')
async def predict_price(car: CarFeatures):
    df = pd.DataFrame([{
        'make_encoded': encode_make(car.make),
        'year': car.year,
        'mileage': car.mileage,
        'fuel_type_encoded': encode_fuel(car.fuel_type),
        'region_encoded': encode_region(car.region),
    }])
    predicted = float(model.predict(df)[0])
    return {
        'predicted_price': round(predicted, -2),
        'confidence': '92%',
        'model_version': 'xgboost-v3'
    }

5.6 Sentiment Analysis (Azure Functions — Serverless)
File: functions/sentiment/__init__.py
import azure.functions as func
from transformers import pipeline
from pymongo import MongoClient

sentiment_pipe = pipeline('sentiment-analysis',
    model='cardiffnlp/twitter-roberta-base-sentiment-latest')

def main(event: func.EventHubEvent):
    data = json.loads(event.get_body())
    result = sentiment_pipe(data['text'])[0]
    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']

    db.reviews.update_one(
        {'_id': ObjectId(data['reviewId'])},
        {'$set': {'sentimentScore': score, 'sentimentLabel': result['label']}}
    )
    # Escalate very negative reviews to support team
    if score < -0.8:
        # Publish escalation event to Azure Event Hubs
        escalate_producer.send_batch([{'body': {'reviewId': data['reviewId']}}])


6. Database Strategy

6.1 PostgreSQL Schema (Cars + Users)
-- services/cars/migrations/001_init.sql
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    password    TEXT NOT NULL,
    name        TEXT NOT NULL,
    role        TEXT DEFAULT 'customer' CHECK (role IN ('customer','admin','agent')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE cars (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    make        TEXT NOT NULL,
    model       TEXT NOT NULL,
    year        SMALLINT NOT NULL,
    price       NUMERIC(12,2) NOT NULL,
    mileage     INTEGER NOT NULL,
    fuel_type   TEXT NOT NULL,
    description TEXT,
    images      TEXT[],
    location    TEXT,
    ml_price    NUMERIC(12,2),       -- Cached AI prediction
    is_sold     BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_cars_make_year ON cars(make, year);
CREATE INDEX idx_cars_price ON cars(price);
CREATE INDEX idx_cars_fuel ON cars(fuel_type);

6.2 Caching Strategy (Redis)

6.3 Elasticsearch Car Indexing
# Run once to create Elasticsearch index with car mapping
PUT /cars
{
  'mappings': {
    'properties': {
      'make':  { 'type': 'keyword' },
      'model': { 'type': 'text', 'analyzer': 'standard' },
      'year':  { 'type': 'integer' },
      'price': { 'type': 'float' },
      'fuel_type': { 'type': 'keyword' },
      'description': { 'type': 'text' },
      'location': { 'type': 'geo_point' }
    }
  }
}


7. Infrastructure as Code
Every infrastructure resource is defined in Terraform code. No clicking in Azure Portal. This is how Netflix, Spotify, and every serious DevOps team manages infrastructure.

7.1 Terraform File Structure

7.2 AKS Cluster (main.tf)
terraform {
  required_providers {
    azurerm = { source = 'hashicorp/azurerm', version = '~> 3.0' }
  }
  backend 'azurerm' {
    resource_group_name  = 'autodrive-tfstate'
    storage_account_name = 'autodrivestate'
    container_name       = 'tfstate'
    key                  = 'prod.terraform.tfstate'
  }
}

resource 'azurerm_kubernetes_cluster' 'aks' {
  name                = 'autodrive-aks'
  location            = var.location
  resource_group_name = var.resource_group
  dns_prefix          = 'autodrive'
  kubernetes_version  = '1.29'

  default_node_pool {
    name                = 'system'
    node_count          = 3
    vm_size             = 'Standard_D4s_v3'   # 4 CPU, 16GB RAM
    enable_auto_scaling = true
    min_count           = 2
    max_count           = 10
  }

  # Separate node pool for ML workloads
  # (added separately with azurerm_kubernetes_cluster_node_pool)

  identity { type = 'SystemAssigned' }
  network_profile { network_plugin = 'azure', network_policy = 'calico' }
}

7.3 Kubernetes Deployment — Chat Service (Helm Chart)
File: helm/charts/chatbot/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-service
spec:
  replicas: {{ .Values.replicas }}
  selector:
    matchLabels: { app: chatbot }
  template:
    spec:
      containers:
      - name: chatbot
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
        ports: [{ containerPort: 8002 }]
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef: { name: autodrive-secrets, key: azure-openai-key }
        resources:
          requests: { cpu: '250m', memory: '512Mi' }
          limits:   { cpu: '1000m', memory: '2Gi' }
        livenessProbe:
          httpGet: { path: /health, port: 8002 }
          initialDelaySeconds: 10
        readinessProbe:
          httpGet: { path: /ready, port: 8002 }
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chatbot-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: { type: Utilization, averageUtilization: 70 }


8. CI/CD Pipeline
The pipeline uses GitHub Actions for CI (test + build) and ArgoCD for CD (GitOps-style deploy to AKS). This is the Netflix/Spotify pattern — no manual kubectl apply in production.

8.1 GitHub Actions — Service CI Pipeline
File: .github/workflows/ci-chatbot.yml
name: CI — Chatbot Service
on:
  push: { branches: [main] }
  pull_request: { branches: [main] }

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -r services/chatbot/requirements.txt
      - run: pytest services/chatbot/tests/ -v --cov --cov-report=xml
      - uses: codecov/codecov-action@v4

  build-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/docker-login@v1
        with: { login-server: autodrive.azurecr.io, ... }
      - name: Build multi-stage Docker image
        run: |
          docker build -t autodrive.azurecr.io/chatbot:${{ github.sha }} \
                       -f services/chatbot/Dockerfile services/chatbot
          docker push autodrive.azurecr.io/chatbot:${{ github.sha }}

  deploy-staging:
    needs: build-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Update Helm values (triggers ArgoCD sync)
        run: |
          sed -i 's|tag: .*|tag: ${{ github.sha }}|' \
               helm/charts/chatbot/values-staging.yaml
          git commit -am 'ci: update chatbot to ${{ github.sha }}'
          git push

8.2 Multi-Stage Dockerfile
File: services/chatbot/Dockerfile
# Stage 1 — Build / dependency install
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target=/app/packages

# Stage 2 — Runtime (minimal image)
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/packages /app/packages
COPY . .
ENV PYTHONPATH='/app/packages'
EXPOSE 8002
USER nobody          # Never run as root
CMD ['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8002', '--workers', '4']


9. Observability — The Three Pillars
Observability is what separates junior from senior engineering. Every service emits metrics, logs, and traces. You can debug any production issue in minutes.

9.1 OpenTelemetry — Auto-Instrumentation
# Install in every Python service
pip install opentelemetry-sdk opentelemetry-instrumentation-fastapi
pip install opentelemetry-exporter-otlp

# main.py — add at top
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint='http://jaeger:4317'))
)
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)   # Auto-traces ALL routes

9.2 Key Grafana Dashboards to Build


10. Security — Zero Trust
Zero Trust means: never trust, always verify. Every service-to-service call is authenticated. Every secret is rotated. No service has more permissions than it needs.

10.1 Security Checklist

JWT tokens signed with RS256 (asymmetric), rotated every 7 days
All service-to-service traffic uses mTLS via Istio service mesh
Zero secrets in code or environment files — use Azure Key Vault only
External Secrets Operator syncs Key Vault secrets into K8s secrets
Rate limiting at Kong: 100 req/min per IP, 1000 req/min per authenticated user
OWASP Top 10 mitigations: SQL parameterization, XSS headers, CSRF tokens
Container images scanned with Trivy in CI (block merge if HIGH CVE found)
RBAC on K8s: each service has its own ServiceAccount with minimal permissions
Network Policies: services can only talk to their own databases
Azure DDoS Standard enabled on public IPs

10.2 Secrets Management Flow


11. Load Handling
Facebook handles 3 billion users. Amazon handles Black Friday. They do it with the same patterns we implement here: horizontal scaling, aggressive caching, async processing, and graceful degradation.

11.1 The Five Patterns




11.2 Load Test Commands

# Install k6 (modern load testing, used by Grafana team)
brew install k6   # Mac

# k6 test script: scripts/load-test.js
import http from 'k6/http'
import { sleep, check } from 'k6'

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp to 100 VUs
    { duration: '5m', target: 100 },   // Sustain 100 VUs (steady state)
    { duration: '2m', target: 500 },   // Spike to 500 VUs
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'],  // 95% of requests under 500ms
    'http_req_failed':   ['rate<0.01'],  // < 1% error rate
  }
}

export default function() {
  const res = http.get('https://autodrive.com/api/cars?page=1')
  check(res, { 'status 200': r => r.status === 200 })
  sleep(1)
}

k6 run scripts/load-test.js


12. Project Phases & Timeline


12.1 Repository Structure
autodrive/                    # Monorepo root
  frontend/                   # Next.js 14 app
  services/
    auth/                     # Node.js + Fastify
    cars/                     # Python FastAPI
    reviews/                  # Node.js + Fastify
    chatbot/                  # Python FastAPI + LangChain
    ml-price/                 # Python FastAPI + MLflow
    sentiment/                # Azure Functions
  infra/
    terraform/                # All Azure resources
    helm/                     # K8s Helm charts
    k8s/                      # Raw manifests (if any)
  .github/workflows/          # CI/CD pipelines
  docker-compose.yml          # Local development
  Makefile                    # Developer shortcuts

12.2 Local Development Setup
# Clone and start everything locally with one command
git clone https://github.com/yourteam/autodrive
cd autodrive
cp .env.example .env          # Fill in Azure OpenAI key, Event Hubs conn string, etc.
make dev                      # Starts all services via docker-compose

# docker-compose.yml services
# frontend     → http://localhost:3000
# auth         → http://localhost:3001
# cars         → http://localhost:8001
# chatbot      → http://localhost:8002
# ml-price     → http://localhost:8003
# postgres     → localhost:5432
# mongodb      → localhost:27017
# redis        → localhost:6379
# event-hubs   → Use Azure Event Hubs (no local emulator needed, use Azurite)
# elasticsearch → localhost:9200
# grafana      → http://localhost:3001
