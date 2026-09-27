<div align="center">

# 🧭 CareerGPT

### AI-powered job application platform — discovery, tailoring, matching, and auto-apply, in one pipeline.

[![Python](https://img.shields.io/badge/python-3.10%2B-2563EB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-0B1D3A?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-2563EB?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/license-Proprietary-0B1D3A?style=flat-square)](#license)

[Overview](#overview) · [Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Configuration](#configuration) · [API Docs](#api-documentation) · [Testing](#testing) · [Roadmap](#roadmap)

</div>

---

## Overview

CareerGPT watches job boards, scores every posting against your resume with an LLM, tailors your resume and writes a cover letter for the ones worth applying to, and can submit the application itself through browser automation — all tracked in a single dashboard from **discovered** to **offer**.

It ships as two independently deployable pieces:

- **`backend/`** — a FastAPI service with LangGraph-orchestrated AI agents, Celery background workers, and a Qdrant-backed matching engine
- **`frontend/`** — a Streamlit client (navy/blue/white theme) that talks to the backend purely over REST

> 📸 *Add product screenshots or a demo GIF here once you have a running deployment — a visual of the Dashboard and Applications pipeline sells this project far better than text.*

---

## Features

**Discovery & Matching**
- 🔍 Multi-source scraping — LinkedIn, Indeed, RemoteOK, Wellfound
- 🎯 AI match scoring with honest strengths/gaps, not generic praise
- 📊 Semantic search via Qdrant + sentence-transformer embeddings

**Application Automation**
- 📄 Per-job resume tailoring with live ATS scoring, rendered to PDF
- ✍️ Cover letters and recruiter outreach, drafted and quality-scored
- 🤖 Auto-apply via Playwright — LinkedIn, Indeed, Greenhouse, and generic company career pages
- 📨 A follow-up agent that sends context-aware messages on schedule

**Presence & Tracking**
- 💼 LinkedIn content agent — drafts and (with your approval) publishes posts
- 📋 One pipeline view of every application, from `discovered` through `offer` / `rejected`

**Platform**
- 🔐 JWT auth (access + refresh), API keys, rate limiting, full audit logging
- 🐳 One-command Docker Compose stack: Postgres, Redis, Qdrant, API, worker, beat, frontend

---

## Architecture

```
┌──────────────────┐      REST API       ┌───────────────────────┐
│   Streamlit UI     │ ──────────────────▶ │    FastAPI Backend      │
│   (frontend/)       │ ◀────────────────── │    (backend/)             │
└──────────────────┘                      └────────────┬────────────┘
                                                          │
                          ┌───────────────────────────────┼───────────────────────────────┐
                          ▼                               ▼                               ▼
                  ┌────────────────┐            ┌───────────────────┐            ┌─────────────────┐
                  │   PostgreSQL     │            │  Redis + Celery     │            │    Qdrant          │
                  │   (data)         │            │  (background jobs)   │            │    (vector search)  │
                  └────────────────┘            └───────────────────┘            └─────────────────┘
                                                          │
                                                ┌──────────┴───────────┐
                                                ▼                      ▼
                                        ┌───────────────┐      ┌────────────────┐
                                        │   Groq LLM      │      │   Playwright     │
                                        │   (agents)      │      │   (auto-apply)   │
                                        └───────────────┘      └────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI · Pydantic v2 · SQLAlchemy 2.0 (async) · Alembic |
| Database | PostgreSQL |
| Background jobs | Celery + Redis |
| Vector search | Qdrant · `sentence-transformers` embeddings |
| LLM agents | LangGraph orchestration · Groq (primary) · OpenAI / Anthropic (optional) |
| Browser automation | Playwright |
| Document generation | ReportLab · WeasyPrint · pdfplumber · PyMuPDF |
| Frontend | Streamlit |
| Auth | JWT (access + refresh) · bcrypt |

## Project Structure

```
career-making/
├── backend/                    FastAPI application
│   ├── app/
│   │   ├── agents/               LangGraph agents — discovery, resume, matching,
│   │   │                         cover letter, outreach, linkedin, application, followup
│   │   ├── api/v1/                 REST route modules
│   │   ├── automation/             Job board scrapers + Playwright auto-apply
│   │   ├── core/                    Config, security, logging, exceptions, constants
│   │   ├── db/                       SQLAlchemy models + session management
│   │   ├── embeddings/               Sentence-embedding generation
│   │   ├── prompts/                  Centralized LLM prompt templates
│   │   ├── repositories/             Data-access layer
│   │   ├── services/                  Business logic
│   │   ├── vectorstore/               Qdrant collection management
│   │   ├── workers/                    Celery task definitions
│   │   └── workflows/                  LangGraph pipelines tying agents together
│   ├── alembic/                      Database migrations
│   ├── tests/                          Pytest suite
│   ├── main.py                          Entrypoint (uvicorn main:app)
│   └── requirements.txt
├── frontend/                    Streamlit application
│   ├── pages/                     Dashboard, Resumes, Job Search, Applications, LinkedIn, Settings
│   ├── utils/                       API client, auth guard, design system
│   └── streamlit_app.py             Entrypoint (login/register)
├── docker-compose.yml           Full stack: postgres, redis, qdrant, backend, worker, beat, frontend
└── README.md
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/Ismailshah-123/Career-making.git
cd Career-making

cp backend/.env.example backend/.env
# edit backend/.env — set APP_SECRET_KEY, DB_PASSWORD, and GROQ_API_KEY at minimum

docker compose up --build
```

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| Interactive API docs | http://localhost:8000/docs |

### Option B — Run locally

**Prerequisites:** Python 3.10+, PostgreSQL 16, Redis 7, and Qdrant (`docker run -p 6333:6333 qdrant/qdrant`).

<details>
<summary><strong>Backend setup</strong></summary>

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env — DB credentials, APP_SECRET_KEY, and GROQ_API_KEY
# (free Groq key: https://console.groq.com)

alembic upgrade head          # create the database schema
uvicorn main:app --reload     # http://localhost:8000
```

Start the background worker in a second terminal (needed for discovery, tailoring, auto-apply, follow-ups):
```bash
cd backend && source venv/bin/activate
celery -A app.workers.celery_app worker --loglevel=info
```
</details>

<details>
<summary><strong>Frontend setup</strong></summary>

```bash
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # points at http://localhost:8000/api/v1 by default
streamlit run streamlit_app.py     # http://localhost:8501
```
</details>

---

## Configuration

All backend configuration lives in `backend/.env` (see `backend/.env.example` for the full template).

| Variable | Required | Purpose |
|---|:---:|---|
| `APP_SECRET_KEY` | ✅ | JWT signing key — generate with `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | ✅ | PostgreSQL connection |
| `GROQ_API_KEY` | ✅ | Powers every AI agent — free tier at [console.groq.com](https://console.groq.com) |
| `REDIS_HOST` | ✅ | Celery broker / result backend |
| `QDRANT_HOST` | ✅ | Vector search for job matching |
| `SMTP_*` | Optional | Transactional email (verification, password reset, follow-ups) |
| `LINKEDIN_CLIENT_ID` / `_SECRET` | Optional | LinkedIn content agent + OAuth |
| `AWS_S3_*` | Optional | S3-compatible storage (defaults to local disk) |

The frontend only needs one variable, in `frontend/.env`:

| Variable | Default |
|---|---|
| `CAREERGPT_API_URL` | `http://localhost:8000/api/v1` |

## API Documentation

Once the backend is running:
- **Swagger UI** — http://localhost:8000/docs
- **ReDoc** — http://localhost:8000/redoc

## Testing

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

The suite covers security/token round-trips, input validators, application-status consistency, and — most importantly — full OpenAPI schema generation, which exercises every route's response model and dependency wiring in one shot.

---

## Roadmap

- [ ] Check in a baseline Alembic migration (`alembic revision --autogenerate -m "init"`)
- [ ] Implement `job_agent` (currently a reserved-but-unused stub; discovery runs through `discovery_agent`)
- [ ] Unify legacy application-status naming between `application_service.py` and the richer pipeline used elsewhere
- [ ] Expand the test suite with integration tests against a live Postgres/Redis/Qdrant stack
- [ ] CI pipeline (lint + test on every PR)

## Contributing

This is currently a solo/commercial project, not accepting outside contributions. If that changes, contribution guidelines will land here.

## License

**Proprietary — All Rights Reserved.** This codebase is not licensed for reuse, redistribution, or resale by anyone other than the owner. Swap this section for MIT / a commercial EULA / etc. before distributing publicly, if that's the intent.

---

<div align="center">
<sub>Built for candidates who want their search run like a pipeline, not a chore.</sub>
</div>
