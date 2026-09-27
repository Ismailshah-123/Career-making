"""Unit tests for auth/security building blocks (no DB required)."""

from __future__ import annotations

import pytest


def test_password_hash_roundtrip():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("a-strong-Passw0rd!")
    assert verify_password("a-strong-Passw0rd!", hashed)
    assert not verify_password("wrong-password", hashed)


def test_access_token_roundtrip():
    from app.core.security import create_access_token, verify_access_token

    token = create_access_token("user-123", email="a@example.com", roles=["user"])
    payload = verify_access_token(token)
    assert payload.sub == "user-123"
    assert payload.user_id == "user-123"


def test_create_token_pair_shape():
    from app.core.security import create_token_pair

    pair = create_token_pair("user-123", "a@example.com", "user")
    assert set(pair.keys()) >= {"access_token", "refresh_token", "token_type", "expires_in"}
    assert pair["token_type"] == "bearer"


def test_expired_token_raises():
    from datetime import timedelta
    from app.core.security import create_access_token, verify_access_token
    from app.core.exceptions import TokenExpiredError

    token = create_access_token("user-123", expires_delta=timedelta(seconds=-10))
    with pytest.raises(TokenExpiredError):
        verify_access_token(token)


def test_validate_full_name_rejects_too_short():
    from app.utils.validators import validate_full_name
    from app.core.exceptions import ValidationError

    assert validate_full_name("  Jane   Doe ") == "Jane Doe"
    with pytest.raises(ValidationError):
        validate_full_name("J")


def test_normalize_url_adds_scheme():
    from app.utils.validators import normalize_url

    assert normalize_url("linkedin.com/in/jane") == "https://linkedin.com/in/jane"
    assert normalize_url("") == ""
