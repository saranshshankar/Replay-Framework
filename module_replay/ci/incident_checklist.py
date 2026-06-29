"""incident_checklist.py — PR task-list rendering + ticked-box parsing.

This module is imported by the 10xCode GitHub Actions workflow
``.github/workflows/incident-checklist.yml`` (plan 06 / the gated record-side).
The DB SELECT over ``module_replay_incidents`` and the GitHub-API comment
post/update live in the workflow, not here.  All public functions in this
module are pure (DB-free) and designed for unit-testing without a live
database or network connection.

Public API
----------
CHECKLIST_MARKER : str
    Hidden HTML comment embedded in every task-list body so the workflow
    can find-and-update the same PR comment idempotently (never spams).

candidate_incidents(rows) -> list[dict]
    Filter, dedup, and sort raw SELECT-result rows.

render_checklist(module_name, incidents) -> str
    Produce a markdown task-list string starting with ``CHECKLIST_MARKER``.

parse_ticked_incident_ids(comment_body) -> list[str]
    Extract ``- [x]``-checked incident_ids from a checklist comment body.

HLD/LLD reference: KT/PHASE-1.5-CI-ARCHITECTURE-HLD-LLD.md B3, D-16.
Design decisions:
  D-13  condition field carries the metric-condition expression.
  D-14  error_code is an OPTIONAL capture label; a missing code never blocks.
  D-16  Sentry = recognition anchor only; render link if present, else
        title+provenance fallback.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Marker — used both by render_checklist and as a guard in parse_ticked_incident_ids.
# The workflow finds the existing PR comment by searching for this marker and
# updates it in place (idempotent, never spams the thread).  B3: "find-by-
# hidden-marker idempotent comment editing".
# ---------------------------------------------------------------------------

CHECKLIST_MARKER = "<!-- module-replay-incident-checklist -->"

# Incident IDs are restricted to this charset (recovery UUIDs and display IDs
# like "INC-20260629-1430-unknown" both match). parse_ticked_incident_ids drops
# any ticked token that does not FULLY match. The gate interpolates these IDs
# into psql SQL, so this allowlist is the SQL-injection guard at the source
# (CR-01) — a PR-comment line like "- [x] x'; DROP TABLE incidents; --" yields
# nothing rather than an injectable token.
_INCIDENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def candidate_incidents(rows: list[dict]) -> list[dict]:
    """Return the de-duplicated, sorted open-incident candidates.

    Given the raw rows returned by a ``SELECT … WHERE module_name=? AND
    status='open'`` query (each row is a plain dict with schema keys), this
    function:

    1. Keeps only rows whose ``status`` equals ``'open'`` (defensive — the
       caller is expected to pre-filter, but this ensures purity).
    2. De-duplicates by ``error_code``: for any set of rows sharing the SAME
       non-null ``error_code``, keep only the FIRST row (earliest in the input
       list, typically the first-seen incident by that code).  Rows whose
       ``error_code`` is ``None`` or the empty string are NEVER collapsed
       (D-14: a missing code never gates / deduplicates).
    3. Returns the survivors sorted ascending by ``ts`` (``None`` sorts last).

    Parameters
    ----------
    rows:
        Raw SELECT result — a list of dicts.  Each dict must contain at least
        ``status``, ``error_code``, and ``ts`` keys.

    Returns
    -------
    list[dict]
        Filtered, deduped, and sorted incident dicts.
    """
    open_rows = [r for r in rows if r.get("status") == "open"]

    seen_codes: set[str] = set()
    deduped: list[dict] = []
    for row in open_rows:
        code = row.get("error_code") or None  # treat empty string as None
        if code is None:
            # Null/empty error_code → always kept (D-14)
            deduped.append(row)
        elif code not in seen_codes:
            seen_codes.add(code)
            deduped.append(row)
        # else: a duplicate non-null code — skip

    deduped.sort(key=lambda r: (r.get("ts") is None, r.get("ts")))
    return deduped


def render_checklist(module_name: str, incidents: list[dict]) -> str:
    """Render a markdown task-list for the given module's open incidents.

    The returned string:
    - Starts with ``CHECKLIST_MARKER`` (the hidden anchor for idempotent
      find-and-update of the PR comment).
    - Carries a heading naming the module.
    - Has one ``- [ ] <incident_id> — <title_or_condition>`` line per incident,
      followed by either a Sentry link or a title+provenance fallback (D-16).
    - When ``incidents`` is empty, renders a single "no open incidents" line
      so that the golden path (untagged → no incidents) is visibly clean in
      the PR comment (FR-9).

    Parameters
    ----------
    module_name:
        The module tag (e.g. ``'perception'``).
    incidents:
        List of incident dicts (pre-filtered by ``candidate_incidents``).
        Each dict may carry: ``incident_id``, ``title``, ``condition``,
        ``sentry_issue_url``, ``error_code``, ``display_id``.

    Returns
    -------
    str
        Markdown string beginning with ``CHECKLIST_MARKER``.
    """
    lines: list[str] = [
        CHECKLIST_MARKER,
        f"## Open incidents for `{module_name}`",
        "",
    ]

    if not incidents:
        lines.append(
            "_No open incidents for this module — golden path, no tagging required._"
        )
        return "\n".join(lines) + "\n"

    for inc in incidents:
        incident_id: str = inc.get("incident_id", "")
        # Prefer title, fall back to condition expression (D-13)
        description: str = inc.get("title") or inc.get("condition") or incident_id

        # Sentry-or-fallback annotation (D-16)
        sentry_url: Optional[str] = inc.get("sentry_issue_url") or None
        if sentry_url:
            annotation = f"([Sentry]({sentry_url}))"
        else:
            error_code: str = inc.get("error_code") or "no code"
            display_id: str = inc.get("display_id") or ""
            if display_id:
                annotation = f"({error_code} · {display_id})"
            else:
                annotation = f"({error_code})"

        lines.append(f"- [ ] {incident_id} — {description} {annotation}")

    lines.append("")
    return "\n".join(lines) + "\n"


def parse_ticked_incident_ids(comment_body: str) -> list[str]:
    """Extract incident_ids from ticked (``- [x]``) checklist lines.

    Only processes a body that contains ``CHECKLIST_MARKER`` — a stray PR
    comment cannot inject incident_ids into the gate (T-0102-01).

    Rules:
    - ``- [x] <incident_id>`` (case-insensitive x) → yields ``incident_id``.
    - ``- [ ] <incident_id>`` (unticked) → yields nothing.
    - Lines not matching the checklist pattern → ignored.
    - The ``incident_id`` token is the first whitespace-separated token after
      the ``[x]`` marker.

    Parameters
    ----------
    comment_body:
        The full text of the PR comment.

    Returns
    -------
    list[str]
        Incident IDs in the order they appear in the comment, deduplicated by
        first occurrence.  Returns ``[]`` if the marker is absent.
    """
    if CHECKLIST_MARKER not in comment_body:
        return []

    # Match lines of the form "- [x] <token> ..." (case-insensitive x)
    pattern = re.compile(r"^- \[[xX]\] (\S+)", re.MULTILINE)
    seen: set[str] = set()
    result: list[str] = []
    for match in pattern.finditer(comment_body):
        incident_id = match.group(1)
        if not _INCIDENT_ID_RE.match(incident_id):
            # Reject tokens carrying SQL/shell-significant characters. The gate
            # interpolates incident_ids into psql; only [A-Za-z0-9_-] passes
            # through (CR-01 — injection guard at the source).
            continue
        if incident_id not in seen:
            seen.add(incident_id)
            result.append(incident_id)
    return result
