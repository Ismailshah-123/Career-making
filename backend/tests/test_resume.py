"""Unit tests for resume upload validation (no DB / no real file storage)."""

from __future__ import annotations

import asyncio
import pytest


def test_validate_upload_file_rejects_oversized(monkeypatch=None):
    from app.utils.file_utils import validate_upload_file
    from app.core.exceptions import FileTooLargeException

    big_content = b"x" * (2 * 1024 * 1024)  # 2 MB

    async def _run():
        await validate_upload_file(
            big_content, "avatar.png",
            content_type="image/png",
            allowed_mimes={"image/png", "image/jpeg"},
            max_size_mb=1,
        )

    with pytest.raises(FileTooLargeException):
        asyncio.run(_run())


def test_validate_upload_file_rejects_empty():
    from app.utils.file_utils import validate_upload_file
    from app.core.exceptions import InvalidFileTypeException

    async def _run():
        await validate_upload_file(
            b"", "avatar.png",
            content_type="image/png",
            allowed_mimes={"image/png"},
            max_size_mb=5,
        )

    with pytest.raises(InvalidFileTypeException):
        asyncio.run(_run())


def test_validate_skills_list_dedupes_and_caps():
    from app.utils.validators import validate_skills_list

    result = validate_skills_list(["Python", "python", "  Python ", "SQL"])
    assert result == ["Python", "SQL"]
