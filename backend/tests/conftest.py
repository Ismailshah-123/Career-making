"""Shared pytest fixtures for the CareerGPT backend test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def app():
    """The FastAPI app object, imported once per test session."""
    from app.main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    """
    A TestClient WITHOUT triggering the lifespan (no live Postgres/Redis/
    Qdrant needed) -- enough to exercise routing, schema generation, and
    any endpoint that doesn't touch the DB.
    """
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)
