# CareerGPT — Frontend

Streamlit UI for CareerGPT. See the [project root README](../README.md) for
full setup instructions and architecture.

Quick start:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # points at your backend's /api/v1
streamlit run streamlit_app.py
```
Runs at http://localhost:8501 — requires the backend to be running.
