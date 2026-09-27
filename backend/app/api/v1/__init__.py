"""
app/api/v1/__init__.py
========================
Aggregates every v1 route module into a single APIRouter.

app/main.py imports `api_router` from this module and mounts it once
under the global "/api/v1" prefix, instead of importing and including
each route module individually. This keeps main.py thin and makes the
route inventory visible in exactly one place.

Mount order matters only for OpenAPI doc grouping, not for routing
correctness — FastAPI resolves paths independently of inclusion order.
health.py is included first since it's checked most frequently by
infra and is useful to see at the top of /docs.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    health,
    auth,
    users,
    resumes,
    jobs,
    applications,
    linkedin,
    recruiters,
    analytics,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(resumes.router)
api_router.include_router(jobs.router)
api_router.include_router(applications.router)
api_router.include_router(linkedin.router)
api_router.include_router(recruiters.router)
api_router.include_router(analytics.router)

__all__ = ["api_router"]
