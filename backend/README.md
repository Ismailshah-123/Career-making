# CareerGPT — Backend

FastAPI backend for CareerGPT. See the [project root README](../README.md) for
full setup instructions (Docker and local), architecture, and API docs.

Quick start:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DB credentials + GROQ_API_KEY
alembic upgrade head
uvicorn main:app --reload
```
API docs: http://localhost:8000/docs
