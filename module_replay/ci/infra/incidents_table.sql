-- incidents_table.sql
-- Applied ONCE by the DB owner to the EXISTING managed Postgres (the 10xCode RDS instance —
-- NOT the deprecated legacy database). NEVER run at gate runtime. Re-running this file is
-- safe (all statements are idempotent — the table and both indexes use IF NOT EXISTS guards).
--
-- Least-priv role split (provision after applying this DDL):
--   gate role      : SELECT + UPDATE(status, fixed, fixed_by_pr, fixed_by_sha, fixed_at, fixed_by_run)
--                    on module_replay_incidents ONLY
--   sync worker    : INSERT + UPDATE(s3_bag_uri) on module_replay_incidents ONLY
--
-- Example grant statements (DB owner runs these once):
--   GRANT SELECT, UPDATE (status, fixed, fixed_by_pr, fixed_by_sha, fixed_at, fixed_by_run)
--       ON module_replay_incidents TO <gate_role>;
--   GRANT INSERT, UPDATE (s3_bag_uri)
--       ON module_replay_incidents TO <worker_role>;
--
-- HLD/LLD reference: KT/PHASE-1.5-CI-ARCHITECTURE-HLD-LLD.md A5/B2
-- Design decisions: 01.1-CONTEXT.md D-15 (existing managed Postgres — the 10xCode RDS instance)
--                   D-19 (site provenance fields dropped; robot_id placeholder; tenxcode_sha via deploy env)

CREATE TABLE IF NOT EXISTS module_replay_incidents (
    incident_id      TEXT PRIMARY KEY,            -- = recovery event_id (UUID)
    display_id       TEXT,                         -- readable INC-YYYYMMDD-HHMM-<robot> (D-17 makeDisplayId)
    ts               TIMESTAMPTZ,                  -- incident time
    error_code       TEXT,                         -- nullable: optional FMEA label (D-14), never gates
    severity         TEXT,
    module_name      TEXT NOT NULL,                -- area->tag (PC/MN/SN); the FR-4 query key
    area_code        INTEGER,
    event_code       INTEGER,
    title            TEXT,
    -- D-21: the RDS is a PLAIN INDEX — known-failure checks live in module config
    -- (incident_detectors), never in the row; the gate runs the config detector set.
    s3_bag_uri       TEXT,                         -- nullable until the uploader sets it (FR-5)
    trigger_source   TEXT,
    reason           TEXT,
    sentry_issue_url TEXT,                         -- nullable recognition anchor (D-16)
    status           TEXT NOT NULL DEFAULT 'open', -- open | fixed | invalid
    fixed            BOOLEAN NOT NULL DEFAULT FALSE,
    fixed_by_pr      TEXT,
    fixed_by_sha     TEXT,
    fixed_at         TIMESTAMPTZ,
    fixed_by_run     TEXT,
    robot_id         TEXT,                         -- provenance, source TBD (NOT machine-id) — placeholder
    tenxcode_sha     TEXT                          -- provenance, deploy-time TENXCODE_SHA env (D-19)
);

-- FR-4 candidate-incident query index: SELECT WHERE module_name=... AND status='open'
CREATE INDEX IF NOT EXISTS idx_module_replay_incidents_module_status
    ON module_replay_incidents (module_name, status);

-- Dedup-by-code index: collapse multiple incidents sharing the same FMEA error_code
CREATE INDEX IF NOT EXISTS idx_module_replay_incidents_error_code
    ON module_replay_incidents (error_code);

-- FR-8 fixed-mark (run by the gate's RDS-mark step, NOT here; reads incident_verdict,
-- never a raw exit code):
-- UPDATE module_replay_incidents
--   SET status='fixed', fixed=true, fixed_by_pr=$2, fixed_by_sha=$3, fixed_by_run=$4, fixed_at=now()
--   WHERE incident_id=$1 AND status<>'fixed';
