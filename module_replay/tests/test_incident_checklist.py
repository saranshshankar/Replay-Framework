"""Tests for ci/infra/incidents_table.sql (DDL idempotency) and
ci/incident_checklist.py (candidate_incidents / render_checklist /
parse_ticked_incident_ids).

These tests run without a live database or network connection — all DB
access is injected as fixture data (plain dicts).

HLD/LLD refs: A5 (schema), B2 (DDL idempotency), B3 (checklist mechanism),
D-14 (null error_code never collapses), D-16 (Sentry-or-fallback).
"""

import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Make the ci package importable from the test suite without installing.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ci.incident_checklist import (  # noqa: E402
    CHECKLIST_MARKER,
    candidate_incidents,
    parse_ticked_incident_ids,
    render_checklist,
)

# ---------------------------------------------------------------------------
# DDL idempotency tests (name contains "ddl")
# ---------------------------------------------------------------------------

_DDL_PATH = _REPO_ROOT / "ci" / "infra" / "incidents_table.sql"


def test_ddl_file_exists():
    """The DDL file must exist at the expected path."""
    assert _DDL_PATH.exists(), f"DDL file not found: {_DDL_PATH}"


def test_ddl_create_table_if_not_exists():
    """The DDL must use CREATE TABLE IF NOT EXISTS (idempotent / re-runnable)."""
    content = _DDL_PATH.read_text()
    assert "CREATE TABLE IF NOT EXISTS module_replay_incidents" in content, (
        "Expected 'CREATE TABLE IF NOT EXISTS module_replay_incidents' in DDL"
    )
    # Exactly one such statement
    assert content.count("CREATE TABLE IF NOT EXISTS module_replay_incidents") == 1


def test_ddl_indexes_if_not_exists():
    """Both indexes must use CREATE INDEX IF NOT EXISTS (idempotent)."""
    content = _DDL_PATH.read_text()
    count = content.count("CREATE INDEX IF NOT EXISTS")
    assert count == 2, f"Expected 2 CREATE INDEX IF NOT EXISTS, found {count}"


def test_ddl_incident_id_primary_key():
    """incident_id must be the TEXT PRIMARY KEY."""
    content = _DDL_PATH.read_text()
    assert "incident_id" in content
    # The column definition must include PRIMARY KEY on the same logical line
    assert "TEXT PRIMARY KEY" in content


def test_ddl_no_shared_db_reference():
    """The DDL must not reference the deprecated shared_db."""
    content = _DDL_PATH.read_text()
    assert "shared_db" not in content.lower(), (
        "DDL must not reference 'shared_db' (deprecated)"
    )


def test_ddl_no_zone_column():
    """No 'zone' column — dropped per D-19."""
    content = _DDL_PATH.read_text()
    assert "zone" not in content.lower(), (
        "DDL must not contain 'zone' — dropped per D-19"
    )


def test_ddl_required_columns_present():
    """The design-critical plain-index columns must appear in the DDL (D-21: no condition)."""
    content = _DDL_PATH.read_text()
    for col in ("s3_bag_uri", "tenxcode_sha", "sentry_issue_url", "module_name", "status"):
        assert col in content, f"Expected column '{col}' in DDL"


def test_ddl_carries_no_condition_columns():
    """D-21: the RDS is a plain index — it must NOT DEFINE condition/verifier_type columns
    (known-failure checks live in module config, not the RDS row). Matches a column
    DEFINITION (line-anchored) so design comments mentioning the words are fine."""
    import re
    content = _DDL_PATH.read_text()
    assert not re.search(r"(?m)^\s*condition\s+\w", content), (
        "DDL must not define a 'condition' column (D-21 plain index)"
    )
    assert not re.search(r"(?m)^\s*verifier_type\s+\w", content), (
        "DDL must not define a 'verifier_type' column (D-21 plain index)"
    )


def test_ddl_full_column_set():
    """The full A5/D-15 column set must be present."""
    content = _DDL_PATH.read_text()
    required = [
        "incident_id",
        "display_id",
        "ts",
        "error_code",
        "severity",
        "module_name",
        "area_code",
        "event_code",
        "title",
        "s3_bag_uri",
        "trigger_source",
        "reason",
        "sentry_issue_url",
        "status",
        "fixed",
        "fixed_by_pr",
        "fixed_by_sha",
        "fixed_at",
        "fixed_by_run",
        "robot_id",
        "tenxcode_sha",
    ]
    for col in required:
        assert col in content, f"Expected column '{col}' in DDL"


def test_ddl_module_status_index():
    """The (module_name, status) index must be present."""
    content = _DDL_PATH.read_text()
    assert "module_name, status" in content or "module_name,status" in content, (
        "Expected composite (module_name, status) index"
    )


def test_ddl_error_code_index():
    """The error_code index must be present."""
    content = _DDL_PATH.read_text()
    assert "idx_module_replay_incidents_error_code" in content


def test_ddl_fr8_fixed_update_comment():
    """The FR-8 idempotent UPDATE template must be present as a comment."""
    content = _DDL_PATH.read_text()
    assert "status='fixed'" in content or "status = 'fixed'" in content, (
        "Expected the FR-8 fixed-mark UPDATE template in the DDL comments"
    )
    assert "status<>'fixed'" in content or "status <> 'fixed'" in content, (
        "Expected the WHERE status<>'fixed' idempotency guard in the DDL comments"
    )


# ---------------------------------------------------------------------------
# candidate_incidents — filter / dedup / sort
# ---------------------------------------------------------------------------

def _make_row(incident_id, error_code=None, status="open", ts=None, **kwargs):
    row = {
        "incident_id": incident_id,
        "error_code": error_code,
        "status": status,
        "ts": ts,
        "title": kwargs.get("title", f"Title for {incident_id}"),
        "condition": kwargs.get("condition", None),
        "sentry_issue_url": kwargs.get("sentry_issue_url", None),
        "display_id": kwargs.get("display_id", None),
    }
    row.update(kwargs)
    return row


def test_candidate_incidents_dedup_shared_error_code():
    """Two rows sharing one non-null error_code → only the first survives."""
    rows = [
        _make_row("INC-001", error_code="E0001"),
        _make_row("INC-002", error_code="E0001"),  # duplicate code
    ]
    result = candidate_incidents(rows)
    assert len(result) == 1
    assert result[0]["incident_id"] == "INC-001"


def test_candidate_incidents_distinct_codes_kept():
    """Two rows with distinct non-null error_codes are both kept."""
    rows = [
        _make_row("INC-001", error_code="E0001"),
        _make_row("INC-002", error_code="E0002"),
    ]
    result = candidate_incidents(rows)
    assert len(result) == 2
    ids = {r["incident_id"] for r in result}
    assert ids == {"INC-001", "INC-002"}


def test_candidate_incidents_null_error_code_never_collapses():
    """Two rows with null error_code are BOTH kept (D-14: missing code never collapses)."""
    rows = [
        _make_row("INC-001", error_code=None),
        _make_row("INC-002", error_code=None),
    ]
    result = candidate_incidents(rows)
    assert len(result) == 2, (
        "Null error_code rows must NOT be deduplicated — each is a distinct incident"
    )
    ids = {r["incident_id"] for r in result}
    assert ids == {"INC-001", "INC-002"}


def test_candidate_incidents_mixed_null_and_code():
    """Null-code rows are always kept; non-null code rows dedup among themselves."""
    rows = [
        _make_row("INC-001", error_code="E0001"),
        _make_row("INC-002", error_code="E0001"),  # dup of INC-001
        _make_row("INC-003", error_code=None),
        _make_row("INC-004", error_code=None),
    ]
    result = candidate_incidents(rows)
    ids = {r["incident_id"] for r in result}
    # INC-002 collapses into INC-001; both nulls kept
    assert ids == {"INC-001", "INC-003", "INC-004"}


def test_candidate_incidents_filters_non_open():
    """Non-open rows (fixed, invalid) are excluded."""
    rows = [
        _make_row("INC-001", status="open"),
        _make_row("INC-002", status="fixed"),
        _make_row("INC-003", status="invalid"),
    ]
    result = candidate_incidents(rows)
    assert len(result) == 1
    assert result[0]["incident_id"] == "INC-001"


def test_candidate_incidents_empty_string_code_is_null():
    """Empty-string error_code is treated as None — never collapses two such rows."""
    rows = [
        _make_row("INC-001", error_code=""),
        _make_row("INC-002", error_code=""),
    ]
    result = candidate_incidents(rows)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# render_checklist — markdown output format
# ---------------------------------------------------------------------------

def test_render_checklist_starts_with_marker():
    """render_checklist output must start with CHECKLIST_MARKER."""
    out = render_checklist("perception", [])
    assert out.startswith(CHECKLIST_MARKER)


def test_render_checklist_empty_incidents_golden_path():
    """Empty incident list → 'no open incidents' line (golden path, FR-9)."""
    out = render_checklist("perception", [])
    assert CHECKLIST_MARKER in out
    lower = out.lower()
    assert "no open incidents" in lower, (
        "Expected a 'no open incidents' line for the golden path"
    )


def test_render_checklist_two_incidents_two_lines():
    """Two incidents → two '- [ ]' task-list lines."""
    incidents = [
        _make_row("INC-001", error_code="E0001", title="Camera TimeSync collapse"),
        _make_row("INC-002", error_code="E0002", title="All-background seg"),
    ]
    out = render_checklist("perception", incidents)
    task_lines = [ln for ln in out.splitlines() if ln.startswith("- [ ]")]
    assert len(task_lines) == 2


def test_render_checklist_lines_contain_incident_id():
    """Each task-list line must contain its incident_id."""
    incidents = [
        _make_row("INC-AAA", title="Some incident"),
        _make_row("INC-BBB", title="Another incident"),
    ]
    out = render_checklist("perception", incidents)
    assert "INC-AAA" in out
    assert "INC-BBB" in out


def test_render_checklist_sentry_link_when_url_present():
    """When sentry_issue_url is set, render a Sentry link annotation."""
    incidents = [
        _make_row(
            "INC-001",
            title="Camera collapse",
            sentry_issue_url="https://sentry.example.com/issues/42",
        )
    ]
    out = render_checklist("perception", incidents)
    assert "[Sentry]" in out
    assert "sentry.example.com" in out


def test_render_checklist_fallback_when_no_sentry_url():
    """When sentry_issue_url is absent, render the error_code + display_id fallback (D-16)."""
    incidents = [
        _make_row(
            "INC-001",
            error_code="E0001",
            display_id="INC-20260601-1230-ROBOT01",
            title="Camera collapse",
            sentry_issue_url=None,
        )
    ]
    out = render_checklist("perception", incidents)
    assert "[Sentry]" not in out
    assert "E0001" in out
    assert "INC-20260601-1230-ROBOT01" in out


def test_render_checklist_fallback_no_code_label():
    """When error_code is also absent, render 'no code' in the fallback."""
    incidents = [
        _make_row(
            "INC-001",
            error_code=None,
            display_id=None,
            title="Unknown incident",
            sentry_issue_url=None,
        )
    ]
    out = render_checklist("perception", incidents)
    assert "no code" in out


# ---------------------------------------------------------------------------
# parse_ticked_incident_ids — round-trips and guard
# ---------------------------------------------------------------------------

def test_parse_ticked_returns_ticked_only():
    """Only ticked (- [x]) lines yield incident_ids; unticked are ignored."""
    body = f"""{CHECKLIST_MARKER}
## Open incidents for `perception`

- [x] ID-A — Camera collapse (E0001)
- [ ] ID-B — Depth all-zero (E0002)
"""
    result = parse_ticked_incident_ids(body)
    assert result == ["ID-A"]


def test_parse_ticked_case_insensitive():
    """Both [x] and [X] are treated as ticked."""
    body = f"""{CHECKLIST_MARKER}
- [X] ID-UPPER
- [x] ID-lower
"""
    result = parse_ticked_incident_ids(body)
    assert "ID-UPPER" in result
    assert "ID-lower" in result


def test_parse_ticked_no_marker_returns_empty():
    """A comment body WITHOUT the marker must return [] (T-0102-01: injection guard)."""
    body = """
- [x] INC-INJECTED — some attacker ticked this
"""
    result = parse_ticked_incident_ids(body)
    assert result == [], (
        "parse_ticked_incident_ids must return [] when CHECKLIST_MARKER is absent"
    )


def test_parse_ticked_empty_body():
    """An empty string returns []."""
    assert parse_ticked_incident_ids("") == []


def test_parse_ticked_round_trip():
    """render_checklist output that is then 'ticked' round-trips correctly."""
    incidents = [
        _make_row("INC-ALPHA", title="Alpha issue"),
        _make_row("INC-BETA", title="Beta issue"),
    ]
    rendered = render_checklist("perception", incidents)
    # Simulate the developer ticking INC-ALPHA
    ticked_body = rendered.replace("- [ ] INC-ALPHA", "- [x] INC-ALPHA")
    result = parse_ticked_incident_ids(ticked_body)
    assert result == ["INC-ALPHA"]
    assert "INC-BETA" not in result


def test_parse_ticked_deduplicates():
    """Duplicate incident_ids in the body are returned once."""
    body = f"""{CHECKLIST_MARKER}
- [x] INC-DUP
- [x] INC-DUP
"""
    result = parse_ticked_incident_ids(body)
    assert result.count("INC-DUP") == 1
