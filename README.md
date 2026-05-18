# CrediRisk

A full-stack credit risk assessment and management platform that predicts loan default probability using machine learning, explains predictions with SHAP values, and provides portfolio-level analytics for financial analysts.

## Features

- **Risk Assessment** — Predict loan default probability for customers, classified into High / Medium / Low risk tiers
- **ML Explainability** — SHAP-based feature importance and GPT-4o-powered natural language explanations
- **What-If Analysis** — Real-time scenario modelling via WebSocket to test how changing inputs shifts risk
- **Portfolio Analytics** — Dashboard with risk distribution, default probability histograms, and portfolio KPIs
- **Vector Similarity Search** — Find similar customer profiles using pgvector embeddings
- **Auth & RBAC** — JWT authentication with admin and analyst roles

## Tech Stack

**Backend**
- FastAPI + Uvicorn (Python)
- PostgreSQL + pgvector
- Redis (caching & rate limiting)
- PyTorch, XGBoost, scikit-learn, SHAP
- OpenAI GPT-4o (AI explanations)
- Alembic (migrations), Prometheus + Sentry (observability)

**Frontend**
- React 19 + TypeScript + Vite
- Recharts (data visualisation)
- React Router v7

**Infrastructure**
- Docker Compose (backend, frontend, PostgreSQL, Redis)

## Getting Started

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (optional — falls back to rule-based explanations)

### Run locally

```bash
git clone https://github.com/vtanisha/ETech-Lab.git
cd ETech-Lab
cp backend/.env.example backend/.env   # add your credentials
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

## Project Structure

```
ETech-Lab/
├── backend/        # FastAPI app — models, inference, auth, database
├── frontend/       # React TypeScript SPA
└── docker-compose.yml
```
