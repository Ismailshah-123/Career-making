"""
CareerGPT — Entry Point
==========================
Thin launcher for the FastAPI application. The actual app factory,
lifespan management, middleware, and routing all live in app/main.py
(app.main:app is the single source of truth).

Run (dev):   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Run (prod):  gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Run direct:  python main.py
"""

from __future__ import annotations

from app.main import app  # noqa: F401  (re-exported for `uvicorn main:app`)

if __name__ == "__main__":
    import uvicorn

    from app.core.config import get_settings

    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.app_workers,
        log_config=None,   # logging is handled by app.core.logging
        access_log=False,  # handled by RequestLoggingMiddleware
        loop="asyncio",
    )
