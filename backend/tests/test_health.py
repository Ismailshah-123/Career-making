"""
Smoke tests: the app assembles correctly.

These are intentionally the highest-leverage tests in this suite --
generating the OpenAPI schema forces FastAPI to validate every single
route's response_model, every dependency's type hints, and every
Pydantic schema in the app. A single broken route (bad response_model,
an unresolved forward ref, a duplicate path) fails these tests.
"""

from __future__ import annotations


def test_app_has_routes(app):
    assert len(app.routes) > 50, "Expected the full v1 API surface to be mounted"


def test_openapi_schema_generates(client):
    """This alone catches most FastAPI wiring bugs (bad response_model, etc.)."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert len(schema["paths"]) > 20


def test_docs_page_loads(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_all_v1_routes_are_mounted(app):
    paths = {route.path for route in app.routes}
    for expected in ("/api/v1/auth/login", "/api/v1/jobs/search", "/api/v1/applications/"):
        assert any(p.rstrip("/") == expected.rstrip("/") for p in paths), f"Missing route: {expected}"
