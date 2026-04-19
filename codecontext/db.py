from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime REAL NOT NULL,
  extension TEXT,
  is_binary INTEGER NOT NULL,
  importance INTEGER NOT NULL,
  summary_tiny TEXT,
  summary_detailed TEXT,
  summary_symbols TEXT,
  summary_change TEXT,
  symbols_json TEXT,
  last_indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS project_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS patches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  backup_path TEXT NOT NULL,
  diff_text TEXT NOT NULL,
  created_at REAL NOT NULL,
  dry_run INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS snippet_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  line INTEGER,
  snippet_hash TEXT NOT NULL,
  sent_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS watcher_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  path TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  chunk_hash TEXT NOT NULL,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_path ON chunk_embeddings(path);

CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  symbol_name TEXT NOT NULL,
  symbol_type TEXT,
  line INTEGER,
  doc TEXT,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);

CREATE TABLE IF NOT EXISTS execution_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  goal TEXT NOT NULL,
  route_mode TEXT NOT NULL,
  intent TEXT,
  outbound_json TEXT,
  result_json TEXT,
  metrics_json TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS request_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  route_mode TEXT NOT NULL,
  local_only INTEGER NOT NULL,
  external_reasoning INTEGER NOT NULL,
  evidence_count INTEGER NOT NULL,
  evidence_chars INTEGER NOT NULL,
  evidence_tokens INTEGER NOT NULL,
  prompt_tokens_est INTEGER NOT NULL,
  avoided_tokens_est INTEGER NOT NULL,
  compaction_chars_saved INTEGER NOT NULL,
  fallback_used INTEGER NOT NULL,
  estimated_tokens INTEGER NOT NULL DEFAULT 0,
  estimated_cost REAL NOT NULL DEFAULT 0,
  route_chosen TEXT,
  cost_reason TEXT,
  estimated_savings_vs_external REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS boundary_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  correlation_id TEXT UNIQUE NOT NULL,
  session_key TEXT,
  request_text_raw TEXT,
  cleaned_goal TEXT,
  cleaned_goal_primary_source TEXT,
  cleaned_goal_fallback_source TEXT,
  matched_intercept_class TEXT,
  original_intercept_class TEXT,
  candidate_relevant INTEGER NOT NULL,
  route_mode TEXT,
  final_route_after_override TEXT,
  complexity_override_applied INTEGER NOT NULL DEFAULT 0,
  complexity_override_reason TEXT,
  classification_completed INTEGER NOT NULL DEFAULT 0,
  heavy_local_handling_triggered INTEGER NOT NULL DEFAULT 0,
  intercept_attempted INTEGER NOT NULL,
  backend_call_attempted INTEGER NOT NULL,
  backend_call_succeeded INTEGER NOT NULL,
  entered_execution_runs INTEGER NOT NULL,
  fallback_used INTEGER NOT NULL,
  old_path_used INTEGER NOT NULL,
  run_id INTEGER,
  reason_code TEXT,
  reason_detail TEXT,
  source_kind TEXT,
  plugin_id TEXT,
  plugin_path TEXT,
  plugin_version_marker TEXT,
  backend_cli_path TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_boundary_events_created_at ON boundary_events(created_at);
CREATE INDEX IF NOT EXISTS idx_boundary_events_class ON boundary_events(matched_intercept_class);
CREATE INDEX IF NOT EXISTS idx_boundary_events_session ON boundary_events(session_key);

CREATE TABLE IF NOT EXISTS pricing_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pricing_snapshot_id TEXT UNIQUE NOT NULL,
  provider TEXT NOT NULL,
  model_name TEXT NOT NULL,
  input_price_per_1m_tokens REAL NOT NULL,
  output_price_per_1m_tokens REAL NOT NULL,
  cached_input_price_per_1m_tokens REAL,
  currency TEXT NOT NULL,
  effective_from REAL,
  effective_to REAL,
  source TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS request_usage_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  timestamp REAL NOT NULL,
  session_key TEXT,
  run_id INTEGER,
  cron_run_id TEXT,
  route_mode TEXT NOT NULL,
  intercept_class TEXT,
  provider TEXT,
  actual_model_name TEXT,
  counterfactual_target_model TEXT,
  pricing_snapshot_id TEXT,
  prompt_tokens_actual INTEGER NOT NULL DEFAULT 0,
  completion_tokens_actual INTEGER NOT NULL DEFAULT 0,
  total_tokens_actual INTEGER NOT NULL DEFAULT 0,
  cost_actual_usd REAL NOT NULL DEFAULT 0,
  counterfactual_prompt_tokens INTEGER NOT NULL DEFAULT 0,
  counterfactual_completion_tokens INTEGER NOT NULL DEFAULT 0,
  counterfactual_total_tokens INTEGER NOT NULL DEFAULT 0,
  saved_cost_prompt_only_usd REAL NOT NULL DEFAULT 0,
  saved_cost_full_modeled_usd REAL NOT NULL DEFAULT 0,
  estimation_method TEXT,
  fallback_used INTEGER NOT NULL DEFAULT 0,
  reason_code TEXT,
  reason_detail TEXT,
  cleaned_goal_hash TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_usage_ledger_created ON request_usage_ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_request_usage_ledger_run_id ON request_usage_ledger(run_id);
CREATE INDEX IF NOT EXISTS idx_request_usage_ledger_cron_run_id ON request_usage_ledger(cron_run_id);

CREATE TABLE IF NOT EXISTS cron_run_usage_ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cron_run_id TEXT UNIQUE NOT NULL,
  cron_job_name TEXT NOT NULL,
  started_at REAL NOT NULL,
  finished_at REAL NOT NULL,
  status TEXT NOT NULL,
  local_only_count INTEGER NOT NULL DEFAULT 0,
  external_count INTEGER NOT NULL DEFAULT 0,
  fallback_count INTEGER NOT NULL DEFAULT 0,
  external_prompt_tokens_actual INTEGER NOT NULL DEFAULT 0,
  external_completion_tokens_actual INTEGER NOT NULL DEFAULT 0,
  external_cost_actual_usd REAL NOT NULL DEFAULT 0,
  counterfactual_prompt_tokens_saved INTEGER NOT NULL DEFAULT 0,
  counterfactual_completion_tokens_saved INTEGER NOT NULL DEFAULT 0,
  saved_cost_prompt_only_usd REAL NOT NULL DEFAULT 0,
  saved_cost_full_modeled_usd REAL NOT NULL DEFAULT 0,
  net_savings_conservative_usd REAL NOT NULL DEFAULT 0,
  net_savings_modeled_usd REAL NOT NULL DEFAULT 0,
  pricing_snapshot_ids TEXT,
  calc_version TEXT,
  notes TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cron_run_usage_ledger_created ON cron_run_usage_ledger(created_at);

CREATE TABLE IF NOT EXISTS call_graph (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  caller_path TEXT NOT NULL,
  caller_name TEXT NOT NULL,
  callee_path TEXT,
  callee_name TEXT NOT NULL,
  call_line INTEGER,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_call_graph_caller ON call_graph(caller_path, caller_name);
CREATE INDEX IF NOT EXISTS idx_call_graph_callee ON call_graph(callee_name);

CREATE TABLE IF NOT EXISTS import_graph (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  importer_path TEXT NOT NULL,
  imported_module TEXT NOT NULL,
  imported_path TEXT,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_import_graph_importer ON import_graph(importer_path);
CREATE INDEX IF NOT EXISTS idx_import_graph_imported ON import_graph(imported_path);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    cols = _table_columns(conn, table)
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=15000;")
    conn.execute("BEGIN;")
    conn.executescript(SCHEMA)

    # Lightweight additive migrations for existing DBs.
    _ensure_column(conn, "execution_runs", "intent", "TEXT")
    _ensure_column(conn, "execution_runs", "outbound_json", "TEXT")
    _ensure_column(conn, "execution_runs", "result_json", "TEXT")
    _ensure_column(conn, "execution_runs", "metrics_json", "TEXT")
    _ensure_column(conn, "execution_runs", "created_at", "REAL")
    _ensure_column(conn, "execution_runs", "updated_at", "REAL")

    _ensure_column(conn, "boundary_events", "session_key", "TEXT")
    _ensure_column(conn, "boundary_events", "request_text_raw", "TEXT")
    _ensure_column(conn, "boundary_events", "cleaned_goal", "TEXT")
    _ensure_column(conn, "boundary_events", "cleaned_goal_primary_source", "TEXT")
    _ensure_column(conn, "boundary_events", "cleaned_goal_fallback_source", "TEXT")
    _ensure_column(conn, "boundary_events", "matched_intercept_class", "TEXT")
    _ensure_column(conn, "boundary_events", "original_intercept_class", "TEXT")
    _ensure_column(conn, "boundary_events", "candidate_relevant", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "route_mode", "TEXT")
    _ensure_column(conn, "boundary_events", "final_route_after_override", "TEXT")
    _ensure_column(conn, "boundary_events", "complexity_override_applied", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "complexity_override_reason", "TEXT")
    _ensure_column(conn, "boundary_events", "classification_completed", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "heavy_local_handling_triggered", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "intercept_attempted", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "backend_call_attempted", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "backend_call_succeeded", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "entered_execution_runs", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "fallback_used", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "old_path_used", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "boundary_events", "run_id", "INTEGER")
    _ensure_column(conn, "boundary_events", "reason_code", "TEXT")
    _ensure_column(conn, "boundary_events", "reason_detail", "TEXT")
    _ensure_column(conn, "boundary_events", "source_kind", "TEXT")
    _ensure_column(conn, "boundary_events", "plugin_id", "TEXT")
    _ensure_column(conn, "boundary_events", "plugin_path", "TEXT")
    _ensure_column(conn, "boundary_events", "plugin_version_marker", "TEXT")
    _ensure_column(conn, "boundary_events", "backend_cli_path", "TEXT")
    _ensure_column(conn, "boundary_events", "created_at", "REAL")
    _ensure_column(conn, "boundary_events", "updated_at", "REAL")

    _ensure_column(conn, "request_metrics", "estimated_tokens", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "request_metrics", "estimated_cost", "REAL NOT NULL DEFAULT 0")
    _ensure_column(conn, "request_metrics", "route_chosen", "TEXT")
    _ensure_column(conn, "request_metrics", "cost_reason", "TEXT")
    _ensure_column(conn, "request_metrics", "estimated_savings_vs_external", "REAL NOT NULL DEFAULT 0")

    conn.commit()
    return conn
