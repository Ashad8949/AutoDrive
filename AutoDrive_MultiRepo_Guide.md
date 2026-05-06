# AutoDrive — Complete Multi-Repo Microservices Implementation Guide

> **Team**: Ashad (Chatbot) · Venkata Mahesh (Frontend) · Pritam Maji (Backend) · Samarth Agrawal (ML)  
> **Project**: Cloud-Native Car Dealership Web Application  
> **Architecture**: 5 GitHub repos, each fully isolated, connected only through HTTP APIs

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Week 1 — API Contracts (Do This First, Together)](#2-week-1--api-contracts-do-this-first-together)
3. [Repo 1: autodrive-infra](#3-repo-1-autodrive-infra)
4. [Repo 2: autodrive-chatbot (Ashad)](#4-repo-2-autodrive-chatbot-ashad)
5. [Repo 3: autodrive-backend (Pritam)](#5-repo-3-autodrive-backend-pritam)
6. [Repo 4: autodrive-ml (Samarth)](#6-repo-4-autodrive-ml-samarth)
7. [Repo 5: autodrive-frontend (Venkata Mahesh)](#7-repo-5-autodrive-frontend-venkata-mahesh)
8. [CI/CD — GitHub Actions for Every Repo](#8-cicd--github-actions-for-every-repo)
9. [Azure Deployment — End-to-End](#9-azure-deployment--end-to-end)
10. [Integration Week — Connecting Everything](#10-integration-week--connecting-everything)

---

## 1. The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERNET / USER                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  autodrive-     │
                    │  frontend       │  Next.js 14
                    │  (Vercel/Azure) │  Port: 3000
                    └────┬───┬───┬───┘
                         │   │   │
          ┌──────────────┘   │   └──────────────────┐
          │                  │                       │
 ┌────────▼───────┐  ┌───────▼──────┐  ┌────────────▼──────┐
 │ autodrive-     │  │ autodrive-   │  │ autodrive-ml       │
 │ chatbot        │  │ backend      │  │                    │
 │ (Ashad)        │  │ (Pritam)     │  │ (Samarth)          │
 │ FastAPI :8002  │  │ Node :8000   │  │ FastAPI :8003      │
 │ RAG + LLM      │  │ Auth+Cars+   │  │ Price Prediction   │
 │ Azure OpenAI   │  │ Reviews      │  │ Sentiment Analysis │
 └────────────────┘  └──────────────┘  └────────────────────┘
          │                  │                       │
          └──────────────────┼───────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  autodrive-infra   │
                   │  (shared config,   │
                   │  docker-compose,   │
                   │  K8s, Terraform)   │
                   └────────────────────┘
```

### Each repo is completely independent:
- Its own `Dockerfile`
- Its own `.github/workflows/` (CI/CD)
- Its own `helm/` chart (Kubernetes scaling)
- Its own environment variables
- Communicates with others **only via HTTP URLs**

---

## 2. Week 1 — API Contracts (Do This First, Together)

> **CRITICAL**: All 4 people must agree on these contracts BEFORE anyone writes a single line of service code.
> The frontend and chatbot depend on the backend being predictable.

Create `autodrive-infra` repo and add this file: `api-contracts/README.md`

---

### 2.1 Backend Service Contracts (Pritam's service)

**Base URL (local)**: `http://localhost:8000`  
**Base URL (production)**: `https://backend.autodrive.io`

#### Auth Endpoints

```
POST   /auth/register
POST   /auth/login
POST   /auth/logout
POST   /auth/refresh
GET    /auth/me
GET    /auth/google          ← OAuth redirect
GET    /auth/google/callback ← OAuth callback
GET    /auth/github
GET    /auth/github/callback
```

**POST /auth/register** — Request:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```
Response `201`:
```json
{
  "user": { "id": "uuid", "name": "John Doe", "email": "john@example.com" },
  "token": "eyJhbGciOiJIUzI1NiJ9..."
}
```

**POST /auth/login** — Request:
```json
{ "email": "john@example.com", "password": "SecurePass123!" }
```
Response `200`:
```json
{
  "user": { "id": "uuid", "name": "John Doe", "email": "john@example.com", "role": "user" },
  "token": "eyJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJhbGci..."
}
```

**GET /auth/me** — Headers: `Authorization: Bearer <token>`  
Response `200`:
```json
{ "id": "uuid", "name": "John Doe", "email": "john@example.com", "role": "user", "created_at": "2025-01-01T00:00:00Z" }
```

---

#### Car Catalog Endpoints

```
GET    /cars                 ← list with filters
GET    /cars/:id             ← single car
POST   /cars                 ← create (admin only)
PUT    /cars/:id             ← update (admin only)
DELETE /cars/:id             ← delete (admin only)
POST   /cars/:id/book        ← book test drive
GET    /cars/:id/bookings    ← get bookings for car
GET    /bookings/me          ← user's own bookings
```

**GET /cars** — Query params: `?make=Toyota&fuel_type=Electric&min_price=20000&max_price=60000&page=1&limit=12&sort=price_asc`  
Response `200`:
```json
{
  "cars": [
    {
      "id": "uuid",
      "make": "Toyota",
      "model": "Camry",
      "year": 2023,
      "price": 28000,
      "mileage": 15000,
      "fuel_type": "Hybrid",
      "location": "Mumbai",
      "description": "Well maintained...",
      "images": ["https://..."],
      "available": true
    }
  ],
  "total": 150,
  "page": 1,
  "pages": 13
}
```

**POST /cars/:id/book** — Headers: `Authorization: Bearer <token>`  
Request:
```json
{
  "date": "2025-03-15",
  "time_slot": "10:00",
  "name": "John Doe",
  "phone": "+91-9876543210"
}
```
Response `201`:
```json
{ "booking_id": "uuid", "car_id": "uuid", "status": "confirmed", "date": "2025-03-15", "time_slot": "10:00" }
```

---

#### Reviews Endpoints

```
GET    /reviews/:car_id      ← get reviews for a car
POST   /reviews/:car_id      ← add review (auth required)
DELETE /reviews/:review_id   ← delete review (own or admin)
```

**GET /reviews/:car_id** — Response `200`:
```json
{
  "reviews": [
    {
      "id": "mongo_object_id",
      "car_id": "uuid",
      "user": { "id": "uuid", "name": "John Doe" },
      "rating": 4,
      "comment": "Great car, smooth ride!",
      "sentiment": "positive",
      "sentiment_score": 0.92,
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "average_rating": 4.2,
  "total": 23
}
```

**POST /reviews/:car_id** — Headers: `Authorization: Bearer <token>`  
Request:
```json
{ "rating": 4, "comment": "Great car, smooth ride!" }
```
Response `201`:
```json
{ "id": "mongo_object_id", "status": "pending_sentiment", "message": "Review submitted, sentiment analysis in progress" }
```

---

### 2.2 Chatbot Service Contracts (Ashad's service)

**Base URL (local)**: `http://localhost:8002`  
**Base URL (production)**: `https://chatbot.autodrive.io`

```
POST   /chat/stream          ← SSE streaming (main endpoint)
POST   /chat                 ← non-streaming JSON
GET    /health
GET    /ready
```

**POST /chat/stream** — Request:
```json
{ "message": "Show me electric cars under 50 lakhs", "session_id": "uuid" }
```
Response: `text/event-stream`
```
data: {"token": "I"}
data: {"token": " found"}
data: {"token": " 3"}
data: {"action": "BOOK_TEST_DRIVE", "car_id": "uuid-123"}
data: [DONE]
```

**POST /chat** — Same request, Response `200`:
```json
{
  "response": "I found 3 electric cars under 50 lakhs...",
  "session_id": "uuid",
  "actions": [{ "type": "BOOK_TEST_DRIVE", "car_id": "uuid-123" }]
}
```

---

### 2.3 ML Service Contracts (Samarth's service)

**Base URL (local)**: `http://localhost:8003`  
**Base URL (production)**: `https://ml.autodrive.io`

```
POST   /predict/price        ← predict price for a car
POST   /sentiment            ← analyze sentiment of text
GET    /health
GET    /ready
```

**POST /predict/price** — Request:
```json
{
  "make": "Toyota",
  "model": "Camry",
  "year": 2020,
  "mileage": 45000,
  "fuel_type": "Hybrid",
  "location": "Mumbai"
}
```
Response `200`:
```json
{
  "predicted_price": 24500,
  "confidence_interval": { "low": 22000, "high": 27000 },
  "model_version": "v1.2.0"
}
```

**POST /sentiment** — Request:
```json
{ "text": "Great car, smooth ride and very fuel efficient!" }
```
Response `200`:
```json
{
  "label": "positive",
  "score": 0.94,
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

---

### 2.4 Frontend Routes (Venkata Mahesh)

These are **page routes**, not API endpoints, but the team should agree:

```
/                            ← Homepage (hero, featured cars)
/cars                        ← Car listing with filters
/cars/[id]                   ← Car detail page
/chat                        ← Chatbot page (embeds chatbot)
/login                       ← Login page
/register                    ← Register page
/dashboard                   ← User dashboard (bookings)
/admin                       ← Admin panel (CRUD cars)
```

---

## 3. Repo 1: autodrive-infra

> **Purpose**: Shared configuration, local development docker-compose, Kubernetes charts, Terraform.  
> **Owner**: Everyone contributes, but designate one person (suggest Pritam since he owns the most services).

### 3.1 Create the repo

```bash
# On GitHub: create repo named "autodrive-infra"
git clone https://github.com/YOUR_ORG/autodrive-infra.git
cd autodrive-infra
```

### 3.2 Full directory structure

```
autodrive-infra/
├── api-contracts/
│   ├── README.md            ← the contracts from Section 2
│   ├── backend.md
│   ├── chatbot.md
│   └── ml.md
├── docker-compose.yml       ← runs ALL services locally
├── docker-compose.dev.yml   ← override for hot reload
├── .env.example             ← all env vars for local dev
├── k8s/
│   ├── namespace.yaml
│   ├── chatbot/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml         ← horizontal pod autoscaler
│   ├── backend/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   ├── ml/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   ├── frontend/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── databases/
│       ├── postgres.yaml
│       ├── mongo.yaml
│       └── redis.yaml
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
│       ├── aks/
│       ├── acr/
│       └── databases/
└── scripts/
    ├── setup-local.sh       ← one-command local setup
    └── deploy-all.sh        ← deploy everything to Azure
```

### 3.3 docker-compose.yml (runs everything locally)

```yaml
version: '3.9'

services:
  # ─── Databases ───────────────────────────────────────────────
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: autodrive
      POSTGRES_USER: autodrive
      POSTGRES_PASSWORD: localpass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U autodrive"]
      interval: 5s
      timeout: 5s
      retries: 5

  mongo:
    image: mongo:7
    environment:
      MONGO_INITDB_DATABASE: autodrive_reviews
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  # ─── Services ────────────────────────────────────────────────
  backend:
    build:
      context: ../autodrive-backend   # ← sibling repo folder
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://autodrive:localpass@postgres:5432/autodrive
      MONGODB_URL: mongodb://mongo:27017/autodrive_reviews
      REDIS_URL: redis://redis:6379
      JWT_SECRET: localsecret-change-in-production
      ML_SERVICE_URL: http://ml:8003
      NODE_ENV: development
    depends_on:
      postgres:
        condition: service_healthy
      mongo:
        condition: service_healthy
      redis:
        condition: service_healthy

  chatbot:
    build:
      context: ../autodrive-chatbot   # ← sibling repo folder
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    environment:
      MODE: local
      REDIS_URL: redis://redis:6379
      CARS_API_URL: http://backend:8000
    depends_on:
      redis:
        condition: service_healthy

  ml:
    build:
      context: ../autodrive-ml        # ← sibling repo folder
      dockerfile: Dockerfile
    ports:
      - "8003:8003"
    environment:
      MODE: local

  frontend:
    build:
      context: ../autodrive-frontend  # ← sibling repo folder
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      NEXT_PUBLIC_BACKEND_URL: http://localhost:8000
      NEXT_PUBLIC_CHATBOT_URL: http://localhost:8002
      NEXT_PUBLIC_ML_URL: http://localhost:8003
    depends_on:
      - backend
      - chatbot
      - ml

volumes:
  postgres_data:
  mongo_data:
```

### 3.4 Local setup script

```bash
# scripts/setup-local.sh
#!/bin/bash
set -e

echo "=== AutoDrive Local Setup ==="

# Check all sibling repos exist
REPOS=("autodrive-chatbot" "autodrive-backend" "autodrive-ml" "autodrive-frontend")
PARENT=$(dirname $(pwd))

for repo in "${REPOS[@]}"; do
  if [ ! -d "$PARENT/$repo" ]; then
    echo "ERROR: $PARENT/$repo not found. Clone all repos as siblings."
    exit 1
  fi
done

# Start all services
docker compose up --build -d

echo ""
echo "=== Services Running ==="
echo "Frontend:  http://localhost:3000"
echo "Backend:   http://localhost:8000"
echo "Chatbot:   http://localhost:8002"
echo "ML:        http://localhost:8003"
echo ""
echo "Logs: docker compose logs -f <service>"
```

---

## 4. Repo 2: autodrive-chatbot (Ashad)

> **Good news**: Your chatbot is already implemented. You just need to restructure it into its own repo and add CI/CD.

### 4.1 Repo structure (final)

```
autodrive-chatbot/
├── src/
│   ├── main.py
│   ├── rag.py
│   ├── history.py
│   ├── ingest.py
│   └── config.py
├── static/
│   └── index.html
├── tests/
│   └── test_main.py
├── seed_data.json
├── requirements.txt
├── Dockerfile
├── .env.example
├── .env               ← gitignored
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        ├── ci.yml     ← test on every PR
        └── deploy.yml ← deploy on merge to main
```

### 4.2 Enhancements needed

The chatbot currently gets car data from `seed_data.json`. In production, it should fetch from the **backend service**. Add this to `ingest.py`:

```python
# ingest.py — add this function
import httpx
import os

async def fetch_cars_from_backend():
    """Fetch live car data from the backend service for RAG ingestion."""
    backend_url = os.getenv("CARS_API_URL", "http://localhost:8000")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{backend_url}/cars?limit=1000")
        response.raise_for_status()
        data = response.json()
        return data["cars"]
```

### 4.3 .env.example

```env
# Mode: "local" (Ollama, no API keys) or "azure" (Azure OpenAI)
MODE=local

# Azure OpenAI (only needed if MODE=azure)
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# OpenAI fallback (only needed if MODE=openai)
OPENAI_API_KEY=

# Azure AI Search (production vector store)
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=autodrive-cars

# Redis (optional — uses in-memory if not set)
REDIS_URL=redis://localhost:6379

# Backend service URL (to fetch live car data)
CARS_API_URL=http://localhost:8000

# Server
PORT=8002
```

### 4.4 Dockerfile (already good — minor update)

```dockerfile
# Build stage
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY static/ ./static/
COPY seed_data.json .
ENV PATH=/root/.local/bin:$PATH
ENV PORT=8002
EXPOSE 8002
RUN useradd -m appuser && chown -R appuser /app
USER appuser
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "4"]
```

### 4.5 Kubernetes HPA (auto-scaling)

```yaml
# k8s/chatbot/hpa.yaml (goes in autodrive-infra)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-hpa
  namespace: autodrive
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chatbot
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

---

## 5. Repo 3: autodrive-backend (Pritam)

> **Services included**: Auth, Car Catalog, Reviews (all in one Node.js Fastify monolith, OR split into 3 — team decision)

### 5.1 Repo structure

```
autodrive-backend/
├── src/
│   ├── app.js               ← Fastify app factory
│   ├── server.js            ← entry point
│   ├── config/
│   │   └── index.js         ← env vars
│   ├── plugins/
│   │   ├── jwt.js           ← JWT plugin
│   │   ├── redis.js         ← Redis connection
│   │   └── db.js            ← Postgres + Mongo connections
│   ├── routes/
│   │   ├── auth.js          ← /auth/*
│   │   ├── cars.js          ← /cars/*
│   │   └── reviews.js       ← /reviews/*
│   ├── models/
│   │   ├── user.js          ← Sequelize/Knex model
│   │   ├── car.js
│   │   ├── booking.js
│   │   └── review.js        ← Mongoose model
│   ├── middleware/
│   │   ├── auth.js          ← JWT verification middleware
│   │   └── rateLimit.js
│   └── services/
│       ├── mlClient.js      ← calls ML service for sentiment
│       └── emailService.js
├── migrations/
│   ├── 001_create_users.sql
│   ├── 002_create_cars.sql
│   └── 003_create_bookings.sql
├── tests/
│   ├── auth.test.js
│   ├── cars.test.js
│   └── reviews.test.js
├── Dockerfile
├── package.json
├── .env.example
├── .gitignore
└── .github/
    └── workflows/
        ├── ci.yml
        └── deploy.yml
```

### 5.2 package.json

```json
{
  "name": "autodrive-backend",
  "version": "1.0.0",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js",
    "test": "jest --coverage",
    "migrate": "node migrations/run.js"
  },
  "dependencies": {
    "@fastify/cors": "^9.0.0",
    "@fastify/jwt": "^8.0.0",
    "@fastify/oauth2": "^7.0.0",
    "@fastify/rate-limit": "^9.0.0",
    "fastify": "^4.27.0",
    "ioredis": "^5.3.2",
    "knex": "^3.1.0",
    "mongoose": "^8.3.0",
    "pg": "^8.11.5",
    "bcrypt": "^5.1.1",
    "zod": "^3.23.0",
    "pino": "^9.0.0"
  },
  "devDependencies": {
    "jest": "^29.7.0",
    "nodemon": "^3.1.0",
    "supertest": "^7.0.0"
  }
}
```

### 5.3 src/app.js

```javascript
import Fastify from 'fastify'
import cors from '@fastify/cors'
import jwt from '@fastify/jwt'
import rateLimit from '@fastify/rate-limit'
import { dbPlugin } from './plugins/db.js'
import { redisPlugin } from './plugins/redis.js'
import authRoutes from './routes/auth.js'
import carsRoutes from './routes/cars.js'
import reviewsRoutes from './routes/reviews.js'

export function buildApp(opts = {}) {
  const app = Fastify({ logger: true, ...opts })

  app.register(cors, { origin: process.env.ALLOWED_ORIGINS?.split(',') || '*' })
  app.register(rateLimit, { max: 100, timeWindow: '1 minute' })
  app.register(jwt, { secret: process.env.JWT_SECRET })
  app.register(dbPlugin)
  app.register(redisPlugin)

  app.register(authRoutes, { prefix: '/auth' })
  app.register(carsRoutes, { prefix: '/cars' })
  app.register(reviewsRoutes, { prefix: '/reviews' })

  app.get('/health', async () => ({ status: 'ok', service: 'autodrive-backend' }))

  return app
}
```

### 5.4 Database migrations

```sql
-- migrations/001_create_users.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE users (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name        VARCHAR(255) NOT NULL,
  email       VARCHAR(255) UNIQUE NOT NULL,
  password    VARCHAR(255),          -- null for OAuth users
  role        VARCHAR(50) DEFAULT 'user',
  oauth_provider VARCHAR(50),
  oauth_id    VARCHAR(255),
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);

-- migrations/002_create_cars.sql
CREATE TABLE cars (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  make        VARCHAR(100) NOT NULL,
  model       VARCHAR(100) NOT NULL,
  year        INTEGER NOT NULL,
  price       NUMERIC(12,2) NOT NULL,
  mileage     INTEGER DEFAULT 0,
  fuel_type   VARCHAR(50) NOT NULL,
  location    VARCHAR(255),
  description TEXT,
  images      TEXT[],
  available   BOOLEAN DEFAULT TRUE,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cars_make ON cars(make);
CREATE INDEX idx_cars_fuel_type ON cars(fuel_type);
CREATE INDEX idx_cars_price ON cars(price);

-- migrations/003_create_bookings.sql
CREATE TABLE bookings (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  car_id      UUID REFERENCES cars(id) ON DELETE CASCADE,
  user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
  date        DATE NOT NULL,
  time_slot   VARCHAR(10) NOT NULL,
  name        VARCHAR(255) NOT NULL,
  phone       VARCHAR(20) NOT NULL,
  status      VARCHAR(50) DEFAULT 'confirmed',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

### 5.5 src/routes/auth.js

```javascript
import bcrypt from 'bcrypt'
import { z } from 'zod'

const registerSchema = z.object({
  name: z.string().min(2).max(100),
  email: z.string().email(),
  password: z.string().min(8).regex(/[A-Z]/).regex(/[0-9]/)
})

export default async function authRoutes(fastify) {
  fastify.post('/register', async (req, reply) => {
    const body = registerSchema.parse(req.body)
    const { db } = fastify

    const existing = await db('users').where({ email: body.email }).first()
    if (existing) return reply.code(409).send({ error: 'Email already registered' })

    const hashed = await bcrypt.hash(body.password, 12)
    const [user] = await db('users')
      .insert({ name: body.name, email: body.email, password: hashed })
      .returning(['id', 'name', 'email', 'role'])

    const token = fastify.jwt.sign({ id: user.id, role: user.role }, { expiresIn: '7d' })
    return reply.code(201).send({ user, token })
  })

  fastify.post('/login', async (req, reply) => {
    const { email, password } = req.body
    const user = await fastify.db('users').where({ email }).first()
    if (!user) return reply.code(401).send({ error: 'Invalid credentials' })

    const valid = await bcrypt.compare(password, user.password)
    if (!valid) return reply.code(401).send({ error: 'Invalid credentials' })

    const token = fastify.jwt.sign({ id: user.id, role: user.role }, { expiresIn: '7d' })
    const { password: _, ...safeUser } = user
    return reply.send({ user: safeUser, token })
  })

  fastify.get('/me', { preHandler: [fastify.authenticate] }, async (req) => {
    const user = await fastify.db('users')
      .where({ id: req.user.id })
      .select('id', 'name', 'email', 'role', 'created_at')
      .first()
    return user
  })
}
```

### 5.6 src/routes/cars.js

```javascript
export default async function carsRoutes(fastify) {
  fastify.get('/', async (req) => {
    const { make, fuel_type, min_price, max_price, page = 1, limit = 12, sort = 'created_at_desc' } = req.query

    let query = fastify.db('cars').where({ available: true })

    if (make) query = query.whereIlike('make', `%${make}%`)
    if (fuel_type) query = query.where({ fuel_type })
    if (min_price) query = query.where('price', '>=', min_price)
    if (max_price) query = query.where('price', '<=', max_price)

    const [sortField, sortDir] = sort.split('_').reduce((acc, val, i, arr) => {
      if (i === arr.length - 1) return [arr.slice(0, -1).join('_'), val]
      return acc
    }, [])
    query = query.orderBy(sortField || 'created_at', sortDir || 'desc')

    const offset = (page - 1) * limit
    const [cars, [{ count }]] = await Promise.all([
      query.clone().limit(limit).offset(offset),
      query.clone().count('* as count')
    ])

    return { cars, total: Number(count), page: Number(page), pages: Math.ceil(count / limit) }
  })

  fastify.get('/:id', async (req, reply) => {
    const car = await fastify.db('cars').where({ id: req.params.id }).first()
    if (!car) return reply.code(404).send({ error: 'Car not found' })
    return car
  })

  fastify.post('/:id/book', { preHandler: [fastify.authenticate] }, async (req, reply) => {
    const { date, time_slot, name, phone } = req.body
    const car = await fastify.db('cars').where({ id: req.params.id, available: true }).first()
    if (!car) return reply.code(404).send({ error: 'Car not found or unavailable' })

    const conflict = await fastify.db('bookings')
      .where({ car_id: req.params.id, date, time_slot, status: 'confirmed' })
      .first()
    if (conflict) return reply.code(409).send({ error: 'Time slot already booked' })

    const [booking] = await fastify.db('bookings')
      .insert({ car_id: req.params.id, user_id: req.user.id, date, time_slot, name, phone })
      .returning('*')

    return reply.code(201).send(booking)
  })
}
```

### 5.7 .env.example

```env
# PostgreSQL
DATABASE_URL=postgresql://autodrive:localpass@localhost:5432/autodrive

# MongoDB (for reviews)
MONGODB_URL=mongodb://localhost:27017/autodrive_reviews

# Redis (sessions + rate limiting)
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=change-this-to-a-long-random-string-in-production

# OAuth (get from Google/GitHub developer console)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
OAUTH_CALLBACK_BASE=http://localhost:3000

# ML service
ML_SERVICE_URL=http://localhost:8003

# CORS (comma-separated, use * for dev)
ALLOWED_ORIGINS=http://localhost:3000

# Server
PORT=8000
NODE_ENV=development
```

### 5.8 Dockerfile

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci --only=production

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY src/ ./src/
COPY migrations/ ./migrations/
ENV NODE_ENV=production
EXPOSE 8000
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser
CMD ["node", "src/server.js"]
```

---

## 6. Repo 4: autodrive-ml (Samarth)

> **Services included**: Price Prediction (XGBoost/MLflow) + Sentiment Analysis (HuggingFace)

### 6.1 Repo structure

```
autodrive-ml/
├── src/
│   ├── main.py              ← FastAPI app
│   ├── config.py
│   ├── routes/
│   │   ├── price.py         ← /predict/price
│   │   └── sentiment.py     ← /sentiment
│   ├── models/
│   │   ├── price_model.py   ← XGBoost model class
│   │   └── sentiment_model.py ← HuggingFace pipeline
│   └── training/
│       ├── train_price.py   ← training script
│       └── generate_data.py ← synthetic training data
├── mlruns/                  ← MLflow experiment tracking (gitignored)
├── saved_models/            ← serialized models (gitignored)
├── tests/
│   ├── test_price.py
│   └── test_sentiment.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── .github/
    └── workflows/
        ├── ci.yml
        └── deploy.yml
```

### 6.2 src/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.price import router as price_router
from src.routes.sentiment import router as sentiment_router

app = FastAPI(title="AutoDrive ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(price_router, prefix="/predict")
app.include_router(sentiment_router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "autodrive-ml"}

@app.get("/ready")
async def ready():
    from src.models.price_model import PriceModel
    from src.models.sentiment_model import SentimentModel
    return {
        "status": "ready",
        "price_model": PriceModel.is_loaded(),
        "sentiment_model": SentimentModel.is_loaded()
    }
```

### 6.3 src/models/price_model.py

```python
import os
import pickle
import numpy as np
from pathlib import Path

MAKE_MAP = {"Toyota": 0, "Tesla": 1, "BMW": 2, "Mercedes": 3, "Ford": 4,
            "Honda": 5, "Hyundai": 6, "Audi": 7, "Kia": 8, "Other": 9}
FUEL_MAP = {"Gasoline": 0, "Hybrid": 1, "Electric": 2, "Plug-in Hybrid": 3, "Diesel": 4}
MODEL_PATH = Path("saved_models/price_model.pkl")

_model = None

class PriceModel:
    @classmethod
    def is_loaded(cls) -> bool:
        return _model is not None

    @classmethod
    def load(cls):
        global _model
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)

    @classmethod
    def predict(cls, make: str, model: str, year: int, mileage: int,
                fuel_type: str, location: str) -> dict:
        if _model is None:
            cls.load()
        if _model is None:
            # Fallback: rule-based estimate if model not trained yet
            base = 30000
            age_factor = max(0, (2024 - year) * 1500)
            mileage_factor = mileage * 0.05
            fuel_bonus = 5000 if fuel_type == "Electric" else 2000 if "Hybrid" in fuel_type else 0
            predicted = base - age_factor - mileage_factor + fuel_bonus
            return {"predicted_price": max(5000, round(predicted, -2)),
                    "confidence_interval": {"low": round(predicted * 0.9, -2), "high": round(predicted * 1.1, -2)},
                    "model_version": "rule-based-fallback"}

        features = np.array([[
            MAKE_MAP.get(make, 9),
            2024 - year,
            mileage / 1000,
            FUEL_MAP.get(fuel_type, 0)
        ]])
        predicted = float(_model.predict(features)[0])
        return {
            "predicted_price": round(predicted, -2),
            "confidence_interval": {
                "low": round(predicted * 0.88, -2),
                "high": round(predicted * 1.12, -2)
            },
            "model_version": "v1.0.0-xgboost"
        }
```

### 6.4 src/models/sentiment_model.py

```python
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)
_pipeline = None

class SentimentModel:
    @classmethod
    def is_loaded(cls) -> bool:
        return _pipeline is not None

    @classmethod
    def load(cls):
        global _pipeline
        try:
            _pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512
            )
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {e}")

    @classmethod
    def analyze(cls, text: str) -> dict:
        if _pipeline is None:
            cls.load()
        if _pipeline is None:
            return {"label": "neutral", "score": 0.5, "model": "unavailable"}

        result = _pipeline(text)[0]
        return {
            "label": result["label"].lower(),
            "score": round(result["score"], 4),
            "model": "distilbert-base-uncased-finetuned-sst-2-english"
        }
```

### 6.5 src/routes/price.py

```python
from fastapi import APIRouter
from pydantic import BaseModel, Field
from src.models.price_model import PriceModel

router = APIRouter()

class PricePredictRequest(BaseModel):
    make: str = Field(..., example="Toyota")
    model: str = Field(..., example="Camry")
    year: int = Field(..., ge=2000, le=2025, example=2020)
    mileage: int = Field(..., ge=0, example=45000)
    fuel_type: str = Field(..., example="Hybrid")
    location: str = Field(default="Unknown", example="Mumbai")

@router.post("/price")
async def predict_price(req: PricePredictRequest):
    return PriceModel.predict(
        make=req.make, model=req.model, year=req.year,
        mileage=req.mileage, fuel_type=req.fuel_type, location=req.location
    )
```

### 6.6 src/routes/sentiment.py

```python
from fastapi import APIRouter
from pydantic import BaseModel, Field
from src.models.sentiment_model import SentimentModel

router = APIRouter()

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="Great car, smooth ride!")

@router.post("/sentiment")
async def analyze_sentiment(req: SentimentRequest):
    return SentimentModel.analyze(req.text)
```

### 6.7 training/train_price.py (generates synthetic data + trains XGBoost)

```python
import numpy as np
import pickle
from pathlib import Path

# Run this once to generate a trained model: python -m src.training.train_price

MAKE_MAP = {"Toyota": 0, "Tesla": 1, "BMW": 2, "Mercedes": 3, "Ford": 4,
            "Honda": 5, "Hyundai": 6, "Audi": 7, "Kia": 8, "Other": 9}
FUEL_MAP = {"Gasoline": 0, "Hybrid": 1, "Electric": 2, "Plug-in Hybrid": 3, "Diesel": 4}

def generate_data(n=5000):
    np.random.seed(42)
    makes = list(MAKE_MAP.keys())
    fuels = list(FUEL_MAP.keys())

    X, y = [], []
    for _ in range(n):
        make = np.random.choice(makes)
        fuel = np.random.choice(fuels)
        age = np.random.randint(0, 15)
        mileage = np.random.randint(0, 200) * 1000

        base = {"Tesla": 55000, "BMW": 50000, "Mercedes": 55000, "Audi": 48000,
                "Toyota": 28000, "Honda": 25000, "Ford": 27000, "Hyundai": 23000,
                "Kia": 22000, "Other": 20000}[make]
        fuel_adj = {"Electric": 8000, "Plug-in Hybrid": 4000, "Hybrid": 2000,
                    "Gasoline": 0, "Diesel": 1000}[fuel]
        price = base + fuel_adj - (age * 2000) - (mileage * 0.04) + np.random.normal(0, 1500)
        price = max(5000, price)

        X.append([MAKE_MAP[make], age, mileage / 1000, FUEL_MAP[fuel]])
        y.append(price)

    return np.array(X), np.array(y)

def train():
    from xgboost import XGBRegressor
    X, y = generate_data()
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X, y)
    Path("saved_models").mkdir(exist_ok=True)
    with open("saved_models/price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to saved_models/price_model.pkl")

if __name__ == "__main__":
    train()
```

### 6.8 requirements.txt

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.8.0
transformers==4.45.0
torch==2.4.0
xgboost==2.1.1
numpy==1.26.4
scikit-learn==1.5.2
mlflow==2.17.0
httpx==0.27.2
pytest==8.3.3
pytest-asyncio==0.24.0
```

### 6.9 Dockerfile

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY saved_models/ ./saved_models/
ENV PATH=/root/.local/bin:$PATH
ENV PORT=8003
EXPOSE 8003
RUN useradd -m mluser && chown -R mluser /app
USER mluser
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]
```

---

## 7. Repo 5: autodrive-frontend (Venkata Mahesh)

### 7.1 Repo structure

```
autodrive-frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx           ← Homepage
│   │   ├── cars/
│   │   │   ├── page.tsx       ← Car listing
│   │   │   └── [id]/
│   │   │       └── page.tsx   ← Car detail
│   │   ├── chat/
│   │   │   └── page.tsx       ← Chatbot page
│   │   ├── login/
│   │   │   └── page.tsx
│   │   ├── register/
│   │   │   └── page.tsx
│   │   └── dashboard/
│   │       └── page.tsx
│   ├── components/
│   │   ├── CarCard.tsx
│   │   ├── CarFilters.tsx
│   │   ├── ChatWidget.tsx
│   │   ├── Navbar.tsx
│   │   └── BookingModal.tsx
│   ├── lib/
│   │   ├── api.ts             ← typed API client functions
│   │   └── auth.ts            ← auth helpers
│   ├── store/
│   │   └── useAuthStore.ts    ← Zustand auth state
│   └── types/
│       └── index.ts           ← shared TypeScript types
├── public/
├── Dockerfile
├── next.config.ts
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── .env.example
└── .github/
    └── workflows/
        ├── ci.yml
        └── deploy.yml
```

### 7.2 .env.example

```env
# Service URLs — these are the ONLY connection points to other services
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_CHATBOT_URL=http://localhost:8002
NEXT_PUBLIC_ML_URL=http://localhost:8003

# NextAuth (if using next-auth instead of custom auth)
NEXTAUTH_SECRET=change-this
NEXTAUTH_URL=http://localhost:3000
```

### 7.3 src/lib/api.ts (central API client — the KEY file)

```typescript
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL!
const CHATBOT_URL = process.env.NEXT_PUBLIC_CHATBOT_URL!
const ML_URL = process.env.NEXT_PUBLIC_ML_URL!

// ─── Types ───────────────────────────────────────────────────────
export interface Car {
  id: string
  make: string
  model: string
  year: number
  price: number
  mileage: number
  fuel_type: string
  location: string
  description: string
  images: string[]
  available: boolean
}

export interface User {
  id: string
  name: string
  email: string
  role: string
}

export interface CarsResponse {
  cars: Car[]
  total: number
  page: number
  pages: number
}

// ─── Auth ─────────────────────────────────────────────────────────
export const authAPI = {
  register: async (name: string, email: string, password: string) => {
    const res = await fetch(`${BACKEND_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password })
    })
    if (!res.ok) throw new Error((await res.json()).error)
    return res.json() as Promise<{ user: User; token: string }>
  },

  login: async (email: string, password: string) => {
    const res = await fetch(`${BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    })
    if (!res.ok) throw new Error((await res.json()).error)
    return res.json() as Promise<{ user: User; token: string }>
  },

  me: async (token: string) => {
    const res = await fetch(`${BACKEND_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    })
    if (!res.ok) throw new Error('Unauthorized')
    return res.json() as Promise<User>
  }
}

// ─── Cars ─────────────────────────────────────────────────────────
export const carsAPI = {
  list: async (params?: {
    make?: string; fuel_type?: string; min_price?: number;
    max_price?: number; page?: number; limit?: number; sort?: string
  }) => {
    const qs = new URLSearchParams(
      Object.entries(params || {}).filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString()
    const res = await fetch(`${BACKEND_URL}/cars${qs ? `?${qs}` : ''}`)
    if (!res.ok) throw new Error('Failed to fetch cars')
    return res.json() as Promise<CarsResponse>
  },

  get: async (id: string) => {
    const res = await fetch(`${BACKEND_URL}/cars/${id}`)
    if (!res.ok) throw new Error('Car not found')
    return res.json() as Promise<Car>
  },

  book: async (carId: string, booking: {
    date: string; time_slot: string; name: string; phone: string
  }, token: string) => {
    const res = await fetch(`${BACKEND_URL}/cars/${carId}/book`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify(booking)
    })
    if (!res.ok) throw new Error((await res.json()).error)
    return res.json()
  }
}

// ─── Chatbot ──────────────────────────────────────────────────────
export const chatAPI = {
  streamChat: async (
    message: string,
    sessionId: string,
    onToken: (token: string) => void,
    onAction: (action: { type: string; car_id: string }) => void,
    onDone: () => void
  ) => {
    const res = await fetch(`${CHATBOT_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId })
    })

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const lines = decoder.decode(value).split('\n')
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') { onDone(); return }
        try {
          const parsed = JSON.parse(data)
          if (parsed.token) onToken(parsed.token)
          if (parsed.action) onAction(parsed)
        } catch {}
      }
    }
  }
}

// ─── ML ───────────────────────────────────────────────────────────
export const mlAPI = {
  predictPrice: async (car: {
    make: string; model: string; year: number;
    mileage: number; fuel_type: string; location?: string
  }) => {
    const res = await fetch(`${ML_URL}/predict/price`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(car)
    })
    if (!res.ok) throw new Error('Price prediction failed')
    return res.json() as Promise<{ predicted_price: number; confidence_interval: { low: number; high: number } }>
  }
}
```

### 7.4 src/components/ChatWidget.tsx

```typescript
'use client'
import { useState, useRef, useEffect } from 'react'
import { chatAPI, carsAPI } from '@/lib/api'
import { v4 as uuidv4 } from 'uuid'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function ChatWidget() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const sessionId = useRef(uuidv4())
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return
    const userMsg = input.trim()
    setInput('')

    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])
    setIsStreaming(true)

    await chatAPI.streamChat(
      userMsg,
      sessionId.current,
      (token) => {
        setMessages(prev => {
          const msgs = [...prev]
          msgs[msgs.length - 1].content += token
          return msgs
        })
      },
      (action) => {
        if (action.type === 'BOOK_TEST_DRIVE') {
          // Redirect to booking modal or car detail page
          window.location.href = `/cars/${action.car_id}?book=true`
        }
      },
      () => setIsStreaming(false)
    )
  }

  return (
    <div className="flex flex-col h-[600px] w-full max-w-2xl mx-auto border rounded-2xl overflow-hidden bg-white shadow-xl">
      <div className="bg-gradient-to-r from-blue-600 to-violet-600 p-4 text-white font-semibold">
        AutoDrive AI Assistant
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {msg.content || (isStreaming && i === messages.length - 1 ? '▌' : '')}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <div className="p-4 border-t flex gap-2">
        <input
          className="flex-1 border rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
          placeholder="Ask about cars, prices, test drives..."
          disabled={isStreaming}
        />
        <button
          onClick={sendMessage}
          disabled={isStreaming || !input.trim()}
          className="bg-blue-600 text-white px-4 py-2 rounded-xl text-sm disabled:opacity-50 hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </div>
  )
}
```

### 7.5 Dockerfile

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
ENV PORT=3000
ENV NODE_ENV=production
CMD ["node", "server.js"]
```

---

## 8. CI/CD — GitHub Actions for Every Repo

> Each repo has identical structure. Only the service name and port change.

### 8.1 CI workflow (runs on every PR)

```yaml
# .github/workflows/ci.yml  (same in all 4 service repos)
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python  # Remove for Node repos
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt  # or: npm ci

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml  # or: npm test

      - name: Build Docker image (verify it builds)
        run: docker build -t autodrive-chatbot:test .
```

### 8.2 Deploy workflow (runs on merge to main)

```yaml
# .github/workflows/deploy.yml
name: Deploy to Azure

on:
  push:
    branches: [main]

env:
  REGISTRY: autodriveacr.azurecr.io   # your Azure Container Registry
  IMAGE_NAME: chatbot                  # change per service: backend, ml, frontend

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to Azure Container Registry
        uses: azure/docker-login@v1
        with:
          login-server: ${{ env.REGISTRY }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Build and push Docker image
        run: |
          docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
          docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest .
          docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

      - name: Deploy to Azure Container Apps
        uses: azure/container-apps-deploy-action@v1
        with:
          resourceGroup: autodrive-rg
          containerAppName: autodrive-${{ env.IMAGE_NAME }}
          imageToDeploy: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          acrName: autodriveacr
          acrUsername: ${{ secrets.ACR_USERNAME }}
          acrPassword: ${{ secrets.ACR_PASSWORD }}
```

### 8.3 GitHub Secrets to add (Settings → Secrets → Actions)

For each repo, add these secrets:

| Secret | Value |
|---|---|
| `ACR_USERNAME` | Azure Container Registry username |
| `ACR_PASSWORD` | Azure Container Registry password |
| `AZURE_CREDENTIALS` | Azure service principal JSON |

---

## 9. Azure Deployment — End-to-End

### 9.1 One-time Azure setup (run once, from autodrive-infra)

```bash
# Login
az login

# Variables — edit these
RESOURCE_GROUP="autodrive-rg"
LOCATION="eastus"
ACR_NAME="autodriveacr"         # must be globally unique
AKS_CLUSTER="autodrive-aks"

# 1. Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create Azure Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic
az acr login --name $ACR_NAME

# 3. Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_CLUSTER \
  --node-count 2 \
  --node-vm-size Standard_B2s \
  --enable-addons monitoring \
  --attach-acr $ACR_NAME \
  --generate-ssh-keys

# 4. Get credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER

# 5. PostgreSQL
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name autodrive-postgres \
  --database-name autodrive \
  --admin-user autodriveadmin \
  --admin-password "YourSecurePass123!" \
  --sku-name Standard_B1ms \
  --tier Burstable

# 6. Redis Cache
az redis create \
  --resource-group $RESOURCE_GROUP \
  --name autodrive-redis \
  --sku Basic \
  --vm-size C0 \
  --location $LOCATION

# 7. Azure OpenAI (for chatbot)
az cognitiveservices account create \
  --resource-group $RESOURCE_GROUP \
  --name autodrive-openai \
  --kind OpenAI \
  --sku S0 \
  --location eastus
```

### 9.2 Kubernetes deployments

```yaml
# k8s/chatbot/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot
  namespace: autodrive
spec:
  replicas: 2
  selector:
    matchLabels:
      app: chatbot
  template:
    metadata:
      labels:
        app: chatbot
    spec:
      containers:
        - name: chatbot
          image: autodriveacr.azurecr.io/chatbot:latest
          ports:
            - containerPort: 8002
          env:
            - name: MODE
              value: "azure"
            - name: AZURE_OPENAI_KEY
              valueFrom:
                secretKeyRef:
                  name: autodrive-secrets
                  key: azure-openai-key
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: autodrive-secrets
                  key: redis-url
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8002
            initialDelaySeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8002
            initialDelaySeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: chatbot
  namespace: autodrive
spec:
  selector:
    app: chatbot
  ports:
    - port: 8002
      targetPort: 8002
  type: ClusterIP
```

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autodrive
---
# k8s/ingress.yaml (NGINX Ingress — routes by path)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autodrive-ingress
  namespace: autodrive
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
    - host: autodrive.yourdomain.com
      http:
        paths:
          - path: /api/auth
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8000
          - path: /api/cars
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8000
          - path: /api/chat
            pathType: Prefix
            backend:
              service:
                name: chatbot
                port:
                  number: 8002
          - path: /api/ml
            pathType: Prefix
            backend:
              service:
                name: ml
                port:
                  number: 8003
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 3000
```

### 9.3 Deploy everything

```bash
# From autodrive-infra/
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/databases/
kubectl apply -f k8s/backend/
kubectl apply -f k8s/chatbot/
kubectl apply -f k8s/ml/
kubectl apply -f k8s/frontend/
kubectl apply -f k8s/ingress.yaml
```

---

## 10. Integration Week — Connecting Everything

> Do this in the final week. Each person should have their service working in isolation first.

### 10.1 Integration checklist

```
Week 1:  [ ] All 4 agree on API contracts (Section 2)
         [ ] Create autodrive-infra repo
         [ ] Each person creates their own repo

Week 2:  [ ] Chatbot: core RAG working locally
         [ ] Backend: /auth and /cars endpoints working
         [ ] ML: sentiment and price prediction working locally
         [ ] Frontend: car listing page working (with mock data)

Week 3:  [ ] Chatbot: connects to backend for live car data
         [ ] Backend: calls ML service for sentiment on reviews
         [ ] Frontend: connects to real backend API
         [ ] Frontend: chat widget connects to chatbot service

Week 4:  [ ] docker-compose up brings everything up
         [ ] All services deployed to Azure
         [ ] End-to-end test: user registers → browses cars → chats → books test drive
         [ ] Kubernetes HPA verified (load test with k6)
```

### 10.2 End-to-end test script

```bash
# scripts/integration-test.sh
#!/bin/bash
BASE_BACKEND="http://localhost:8000"
BASE_CHATBOT="http://localhost:8002"
BASE_ML="http://localhost:8003"
BASE_FRONTEND="http://localhost:3000"

echo "=== AutoDrive Integration Tests ==="

# Health checks
echo -n "Backend health... "
curl -sf $BASE_BACKEND/health && echo "OK" || echo "FAILED"

echo -n "Chatbot health... "
curl -sf $BASE_CHATBOT/health && echo "OK" || echo "FAILED"

echo -n "ML health... "
curl -sf $BASE_ML/health && echo "OK" || echo "FAILED"

echo -n "Frontend... "
curl -sf $BASE_FRONTEND && echo "OK" || echo "FAILED"

# Functional tests
echo ""
echo "=== Functional Tests ==="

echo -n "Register user... "
TOKEN=$(curl -sf -X POST $BASE_BACKEND/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@test.com","password":"TestPass123!"}' \
  | jq -r '.token')
[ "$TOKEN" != "null" ] && echo "OK" || echo "FAILED"

echo -n "List cars... "
curl -sf "$BASE_BACKEND/cars?limit=5" | jq '.total' && echo "cars found"

echo -n "Price prediction... "
curl -sf -X POST $BASE_ML/predict/price \
  -H "Content-Type: application/json" \
  -d '{"make":"Toyota","model":"Camry","year":2020,"mileage":45000,"fuel_type":"Hybrid"}' \
  | jq '.predicted_price'

echo -n "Sentiment analysis... "
curl -sf -X POST $BASE_ML/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"Great car, very smooth ride!"}' \
  | jq '.label'

echo -n "Chatbot response... "
curl -sf -X POST $BASE_CHATBOT/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me electric cars","session_id":"test-123"}' \
  | jq '.response' | head -c 100

echo ""
echo "=== Integration tests complete ==="
```

### 10.3 What each person delivers for demo

| Person | Service | Demo Points |
|---|---|---|
| **Ashad** | chatbot | RAG chat with car recommendations, test drive booking detection, streaming responses |
| **Venkata Mahesh** | frontend | Car listing with filters, chat widget, booking modal, auth flow |
| **Pritam** | backend | Auth with JWT+OAuth, car CRUD, booking system, reviews |
| **Samarth** | ml | Price prediction widget on car detail page, sentiment badges on reviews |

### 10.4 What impresses the professor

1. **Each service has independent CI/CD** — push to chatbot repo → only chatbot redeploys
2. **HPA proof** — show Grafana dashboard with pods scaling under load
3. **Complete isolation** — services only talk via HTTP, no shared code
4. **Observability** — Prometheus metrics + Grafana dashboard showing request rates per service
5. **Infrastructure as code** — `terraform apply` recreates everything from scratch

---

*Guide Version 1.0 — AutoDrive Team 2025*
