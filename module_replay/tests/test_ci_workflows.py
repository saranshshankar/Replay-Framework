"""CI workflow hardening + security-invariant regression guard (plan 01-18 / repointed 01.1-03).

These tests parse the GitHub Actions workflow templates with ``yaml.safe_load``
and pin both the debuggability behavior (logs uploaded, every artifact ephemeral
via ``retention-days: 5``, a developer-facing artifact link) AND the CI security
invariants (``pull_request`` only — never ``pull_request_target``; OIDC-only AWS
auth with no long-lived keys; every ``uses:`` pinned to a concrete ref; minimal
permissions) so a future edit cannot silently regress them (T-18-01..T-18-05).

REPOINT (plan 01.1-03): When Path-A (commit 1d6ad25) moved the gate/nightly
templates from ``.github/workflows/`` into ``module_replay/ci/10xcode/``, the
original ``skipif`` keyed off the deleted ``.github/workflows/replay-gate.yml``
and made all 14 tests silently skip.  This file now points WORKFLOWS_DIR at
``module_replay/ci/10xcode/`` and covers:
  - replay-perception-gate.yml    (GATE)
  - replay-perception-nightly.yml (NIGHTLY)
  - replay-perception-viz.yml     (VIZ — path resolved; own security test in
                                    test_viz.py; not re-covered here)

Path-A nightly note: the nightly bundles all run artefacts (reports + logs +
bag) into ONE ``nightly-<run_id>`` directory at /tmp/nightly (a single
upload-artifact step).  There are NO separate ``report``/``logs``/``bag`` upload
paths.  ``test_nightly_uploads_artifacts`` is rewritten to assert the real
Path-A shape.

Plan 06 will extend this file to add sweep-template + merge-fragment coverage;
WORKFLOWS_DIR is resolved from ci/10xcode/ to compose cleanly.

PyYAML is a core dep (always importable). A ``${{ }}`` expression inside a YAML
flow-mapping breaks ``safe_load``, so the workflows MUST stay block style — if
that regresses, ``yaml.safe_load`` raises here and the suite goes red, which is
exactly the guard we want.
"""

import re
from pathlib import Path

import pytest
import yaml

# Test file lives at module_replay/tests/.
# parents[0] = module_replay/tests/
# parents[1] = module_replay/
# parents[2] = repo root
#
# Path-A: templates live under module_replay/ci/10xcode/ (not .github/workflows/).
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / "module_replay" / "ci" / "10xcode"
GATE = WORKFLOWS_DIR / "replay-perception-gate.yml"
NIGHTLY = WORKFLOWS_DIR / "replay-perception-nightly.yml"
VIZ = WORKFLOWS_DIR / "replay-perception-viz.yml"

# Skip-guard: confirm the templates are present before asserting on their content.
# A sparse checkout that omits module_replay/ci/ would otherwise produce a confusing
# FileNotFoundError rather than a clean skip.
pytestmark = pytest.mark.skipif(
    not (GATE.is_file() and NIGHTLY.is_file()),
    reason="perception CI templates not present in this checkout",
)


def _load(path: Path) -> dict:
    """Parse a workflow YAML. Raises if block style regressed to a broken flow map."""
    return yaml.safe_load(path.read_text())


def _upload_steps(workflow: dict) -> list[dict]:
    """All steps across all jobs that use actions/upload-artifact."""
    steps = []
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            uses = step.get("uses") or ""
            if uses.startswith("actions/upload-artifact"):
                steps.append(step)
    return steps


def _all_steps(workflow: dict) -> list[dict]:
    steps = []
    for job in (workflow.get("jobs") or {}).values():
        steps.extend(job.get("steps") or [])
    return steps


# --- NEW behavior: ephemeral retention on every upload --------------------


def test_all_uploads_have_retention_5():
    """EVERY upload-artifact step in BOTH workflows is ephemeral (retention-days: 5).

    T-18-04: bounds the exposure window of uploaded bag/logs/report artifacts.
    """
    for path in (GATE, NIGHTLY):
        wf = _load(path)
        uploads = _upload_steps(wf)
        assert uploads, f"{path.name}: expected at least one upload-artifact step"
        for step in uploads:
            with_block = step.get("with") or {}
            assert (
                with_block.get("retention-days") == 5
            ), f"{path.name}: upload step {step.get('name') or step.get('uses')!r} must set retention-days: 5"


# --- NEW behavior: gate uploads the logs ----------------------------------


def test_gate_uploads_logs():
    """replay-gate.yml uploads the run's logs (the failure evidence), from the GPU job.

    The /tmp/output/logs dir exists only on the GPU `replay` runner (01-17), not
    on the cheap metrics runner which only has the downloaded bag.
    """
    wf = _load(GATE)
    log_uploads = [
        s
        for s in _upload_steps(wf)
        if "logs" in ((s.get("with") or {}).get("path") or "")
        and "logs" in ((s.get("with") or {}).get("name") or "")
    ]
    assert log_uploads, "replay-gate.yml must upload a logs artifact (path+name contain 'logs')"

    # The logs upload must live in the `replay` (GPU) job — the metrics runner
    # never sees /tmp/output/logs.
    replay_job = wf["jobs"]["replay"]
    replay_uploads = [
        s
        for s in (replay_job.get("steps") or [])
        if (s.get("uses") or "").startswith("actions/upload-artifact")
        and "logs" in ((s.get("with") or {}).get("path") or "")
    ]
    assert replay_uploads, "the logs artifact must be uploaded from the GPU `replay` job"

    # Logs are the failure evidence: upload even when the replay step failed.
    for s in replay_uploads:
        assert (
            str(s.get("if", "")).strip() == "always()"
        ), "logs upload must use `if: always()` so a failed run still uploads its logs"


# --- NEW behavior: nightly uploads artifacts (it had none) ----------------


def test_nightly_uploads_artifacts():
    """replay-nightly.yml uploads at least one ephemeral artifact (all ephemeral).

    Path-A nightly shape: ONE ``nightly-<run_id>`` upload at /tmp/nightly that
    bundles reports + logs + bag together in the single run directory.  There are
    no separate ``report``/``logs``/``bag`` upload steps — those per-path
    assertions are replaced by the three checks below.

    (a) at least one upload-artifact step exists.
    (b) every upload sets retention-days: 5.
    (c) every uploaded path or artifact name contains "nightly" — confirming the
        run-dir bundle pattern rather than a separate per-type upload.
    """
    wf = _load(NIGHTLY)
    uploads = _upload_steps(wf)
    assert len(uploads) >= 1, "replay-nightly.yml must upload at least one artifact"

    # (b) every upload is ephemeral
    for s in uploads:
        with_block = s.get("with") or {}
        assert (
            with_block.get("retention-days") == 5
        ), (
            f"nightly upload step {s.get('name') or s.get('uses')!r} "
            "must set retention-days: 5"
        )

    # (c) Path-A bundles all artefacts into the nightly-<run_id> directory;
    #     the artifact name or path must contain "nightly".
    for s in uploads:
        with_block = s.get("with") or {}
        artifact_name = (with_block.get("name") or "").lower()
        artifact_path = (with_block.get("path") or "").lower()
        assert "nightly" in artifact_name or "nightly" in artifact_path, (
            f"nightly upload step {s.get('name') or s.get('uses')!r}: "
            "expected artifact name or path to contain 'nightly' "
            "(Path-A bundles all run artefacts into nightly-<run_id>/)"
        )


# --- NEW behavior: developer-facing artifact link -------------------------


def test_developer_link_surfaced():
    """replay-gate.yml surfaces the artifacts to the developer.

    Accept EITHER a step writing to $GITHUB_STEP_SUMMARY (minimal-permission job
    summary) OR a pinned actions/github-script step that posts an issue/PR
    comment (uses the declared pull-requests: write). Both satisfy the ask.
    """
    wf = _load(GATE)
    raw = GATE.read_text()

    has_summary = "GITHUB_STEP_SUMMARY" in raw
    has_comment = any(
        (s.get("uses") or "").startswith("actions/github-script") for s in _all_steps(wf)
    ) and bool(re.search(r"issue.*comment|createComment", raw, re.IGNORECASE))

    assert (
        has_summary or has_comment
    ), "replay-gate.yml must surface the artifact link (a $GITHUB_STEP_SUMMARY line OR a github-script PR comment)"


# --- SECURITY regression guard (these MUST stay true) ---------------------


def test_security_no_pull_request_target():
    """T-18-01: neither workflow may use pull_request_target (fork-PR secret exfil)."""
    for path in (GATE, NIGHTLY):
        raw = path.read_text()
        assert (
            "pull_request_target" not in raw
        ), f"{path.name}: pull_request_target is forbidden (secret exfiltration)"
        wf = _load(path)
        on = wf.get("on") or wf.get(True)  # PyYAML may parse bare `on:` as the bool True
        assert isinstance(on, dict), f"{path.name}: `on:` should be a mapping of triggers"
        assert "pull_request_target" not in on


def test_gate_triggers_on_pull_request_only():
    """replay-gate fires on pull_request (the safe event)."""
    wf = _load(GATE)
    on = wf.get("on") or wf.get(True)
    assert "pull_request" in on


def test_security_oidc_only_no_long_lived_keys():
    """T-18-02: OIDC only (id-token: write); no long-lived AWS keys anywhere."""
    for path in (GATE, NIGHTLY):
        raw = path.read_text()
        assert "AWS_SECRET" not in raw, f"{path.name}: no long-lived AWS_SECRET* keys"
        wf = _load(path)
        perms = wf.get("permissions") or {}
        assert (
            perms.get("id-token") == "write"
        ), f"{path.name}: id-token: write (OIDC federation) required"


def test_security_all_actions_pinned():
    """T-18-03: every `uses:` is pinned to a concrete ref (no bare/@main/@master)."""
    pin = re.compile(r"@(v?\d|[0-9a-f]{40})")  # @v4 / @v4.0.1 / @<40-hex sha>
    for path in (GATE, NIGHTLY):
        wf = _load(path)
        for step in _all_steps(wf):
            uses = step.get("uses")
            if not uses:
                continue
            assert "@" in uses, f"{path.name}: action {uses!r} must be pinned (no bare ref)"
            ref = uses.split("@", 1)[1]
            assert ref not in ("main", "master"), f"{path.name}: {uses!r} pins a floating branch"
            assert pin.search(uses), f"{path.name}: {uses!r} must pin a concrete version/sha"


def test_gate_minimal_permissions_intact():
    """replay-gate keeps minimal perms: contents: read + the now-USED pull-requests: write."""
    wf = _load(GATE)
    perms = wf.get("permissions") or {}
    assert perms.get("contents") == "read"
    assert perms.get("pull-requests") == "write"
    assert perms.get("id-token") == "write"


def test_both_workflows_parse():
    """Block style preserved — a ${{ }} in a flow-mapping would break safe_load."""
    assert isinstance(_load(GATE), dict)
    assert isinstance(_load(NIGHTLY), dict)


# --- README "Debugging a failed run" section (GAP e) ----------------------

README = Path(__file__).resolve().parent.parent / "README.md"


def _debug_section() -> str:
    """The README text from the 'Debugging a failed run' heading onward."""
    text = README.read_text()
    marker = "Debugging a failed run"
    assert marker in text, "README must have a 'Debugging a failed run' section"
    return text[text.index(marker):]


def test_readme_has_debug_section():
    """The section names the three artifact locations after a run (bag/logs/report)."""
    section = _debug_section()
    assert "<output>/replay_output" in section, "must name the output bag location"
    assert "<output>/logs/" in section, "must name the logs dir"
    assert "recorder.log" in section and "module.log" in section, "must name both log files"
    assert "<output>/reports/report.html" in section, "must name the report location"


def test_readme_documents_ci_artifact_download():
    """The section explains pulling the EPHEMERAL CI artifact from the Actions run."""
    section = _debug_section().lower()
    assert "artifact" in section, "must mention downloading the CI artifact"
    assert (
        "retention" in section or "5 day" in section or "few days" in section
    ), "must convey the artifacts are ephemeral (~5 day retention)"


def test_readme_notes_local_viz_followon():
    """Scope honesty: local rich-viz (--run-viz) is a follow-on / deferred (Tier 3 out)."""
    section = _debug_section()
    lowered = section.lower()
    assert "--run-viz" in section or "local" in lowered, "must reference the local-viz mode"
    assert (
        "follow-on" in lowered or "deferred" in lowered
    ), "must note local rich-viz is a follow-on / deferred"
