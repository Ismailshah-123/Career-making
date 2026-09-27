"""
app/db/base_import.py
========================
Single place that imports every ORM model so `Base.metadata` is fully
populated before anything inspects it (Alembic autogenerate, Base.metadata.create_all()
in tests, etc.).

Importing app/db/base.py alone is NOT enough -- Base.metadata only knows
about a model once that model's module has actually been imported
somewhere. Import THIS module (not the individual model files) wherever
you need the complete schema.
"""

from __future__ import annotations

from app.db.base import Base  # noqa: F401

# Import every model so it registers itself on Base.metadata.
from app.db.models.user import User  # noqa: F401
from app.db.models.job import Job  # noqa: F401
from app.db.models.resume import Resume  # noqa: F401
from app.db.models.application import Application  # noqa: F401
from app.db.models.cover_letter import CoverLetter  # noqa: F401
from app.db.models.recruiter import Recruiter  # noqa: F401
from app.db.models.company import Company  # noqa: F401
from app.db.models.linkedin_post import LinkedInPost  # noqa: F401
from app.db.models.agent_run import AgentRun  # noqa: F401
from app.db.models.audit_log import AuditLog  # noqa: F401

__all__ = [
    "Base",
    "User", "Job", "Resume", "Application", "CoverLetter",
    "Recruiter", "Company", "LinkedInPost", "AgentRun", "AuditLog",
]
