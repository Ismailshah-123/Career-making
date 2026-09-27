"""
Consistency tests for the constants that drive the application pipeline.

These exist specifically to catch the class of bug this codebase had
before its QA pass: enum members referenced somewhere but never defined,
or a transitions table with dangling/missing keys.
"""

from __future__ import annotations


def test_every_status_has_a_transition_entry():
    from app.core.constants import ApplicationStatus, APPLICATION_STATUS_TRANSITIONS

    for status in ApplicationStatus:
        assert status in APPLICATION_STATUS_TRANSITIONS, f"{status} has no transitions entry"


def test_transition_targets_are_valid_statuses():
    from app.core.constants import ApplicationStatus, APPLICATION_STATUS_TRANSITIONS

    valid = set(ApplicationStatus)
    for status, targets in APPLICATION_STATUS_TRANSITIONS.items():
        for target in targets:
            assert target in valid, f"{status} -> {target} is not a real ApplicationStatus"


def test_job_source_values_are_lowercase_slugs():
    from app.core.constants import JobSource

    for source in JobSource:
        assert source.value == source.value.lower()
        assert " " not in source.value
