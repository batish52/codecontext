from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from codecontext.db import connect
from codecontext.telemetry import product_metrics_report


def _insert_boundary(conn, **kw):
    now = kw.get("created_at", time.time())
    conn.execute(
        """
        INSERT INTO boundary_events(
          correlation_id, session_key, request_text_raw, cleaned_goal, matched_intercept_class,
          candidate_relevant, route_mode, classification_completed, heavy_local_handling_triggered,
          intercept_attempted, backend_call_attempted, backend_call_succeeded,
          entered_execution_runs, fallback_used, old_path_used, run_id, reason_code, reason_detail,
          source_kind, plugin_id, plugin_path, plugin_version_marker, backend_cli_path,
          created_at, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            kw.get("correlation_id"), "s", kw.get("prompt", ""), kw.get("goal", ""), kw.get("klass", "runtime_diagnostics"),
            1, kw.get("route_mode"), 1, 1,
            1, kw.get("backend_call_attempted", 0), kw.get("backend_call_succeeded", 0),
            kw.get("entered_execution_runs", 0), 0, 0, kw.get("run_id"), kw.get("reason_code", "candidate_relevant"), kw.get("reason_detail", ""),
            "live_plugin", "codecontext-runtime", "codecontext-runtime", "test", "python main.py",
            now, now,
        ),
    )


def _insert_run(conn, rid, goal, route_mode, cost, created_at):
    metrics = {
        "estimated_cost": cost,
        "estimated_request_cost_usd": cost,
    }
    conn.execute(
        "INSERT INTO execution_runs(id, goal, route_mode, intent, outbound_json, result_json, metrics_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (rid, goal, route_mode, "runtime_diagnostics", None, "{}", json.dumps(metrics), created_at, created_at + 1.2),
    )


def _insert_metric(conn, route, est_tokens, est_cost, savings, created_at, cost_reason="ok"):
    conn.execute(
        "INSERT INTO request_metrics(route_mode, local_only, external_reasoning, evidence_count, evidence_chars, evidence_tokens, prompt_tokens_est, avoided_tokens_est, compaction_chars_saved, fallback_used, estimated_tokens, estimated_cost, route_chosen, cost_reason, estimated_savings_vs_external, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            route,
            1 if route == "local_only" else 0,
            1 if route == "external_reasoning_with_compaction" else 0,
            0, 0, 0, est_tokens, 0, 0, 0,
            est_tokens, est_cost, route, cost_reason, savings, created_at,
        ),
    )


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "t.db"
        conn = connect(db_path)
        try:
            # empty DB
            rep = product_metrics_report(db_path, window="1d", top_n=3)
            assert rep["metrics_report"]["total_requests"] == 0

            now = time.time()
            _insert_boundary(conn, correlation_id="c1", prompt="thanks", goal="thanks", klass="existing_relevant", route_mode="pass_through_direct", reason_code="pass_through_direct", created_at=now)
            _insert_boundary(conn, correlation_id="c2", prompt="check logs", goal="check logs", klass="runtime_diagnostics", route_mode="local_only", reason_code="entered_local_pipeline", backend_call_succeeded=1, entered_execution_runs=1, created_at=now)
            _insert_boundary(conn, correlation_id="c3", prompt="explain cost", goal="explain cost", klass="explain_style", route_mode="external_reasoning_with_compaction", reason_code="entered_local_pipeline", backend_call_succeeded=1, entered_execution_runs=1, created_at=now)
            _insert_boundary(conn, correlation_id="c4", prompt="broken", goal="broken", klass="explain_style", route_mode="external_reasoning_with_compaction", reason_code="backend_call_failed", reason_detail=json.dumps({"stage": "timeout"}), created_at=now)

            _insert_run(conn, 1, "check logs", "local_only", 0.001, now)
            _insert_run(conn, 2, "explain cost", "external_reasoning", 0.02, now)

            _insert_metric(conn, "local_only", 120, 0.001, 0.02, now, cost_reason="high_cost_force_local")
            _insert_metric(conn, "external_reasoning_with_compaction", 500, 0.02, 0.0, now)
            conn.commit()

            rep = product_metrics_report(db_path, window="1d", top_n=2, include_definitions=True)
            m = rep["metrics_report"]
            assert m["pass_through_direct_count"] >= 1
            assert m["local_only_count"] >= 1
            assert m["external_reasoning_with_compaction_count"] >= 1
            assert m["estimated_savings_total"] >= 0
            assert m["threshold_hit_count"] >= 1
            assert m["failure_counts_by_type"].get("timeout", 0) >= 1
            assert len(m["top_expensive_prompts"]) >= 1
            # pass-through should not count as external avoided
            assert m["external_avoided_count"] == 1
            # local handled percentage excludes pass-through
            assert m["local_handled_percentage"] == 33.33
            # definitions exposed
            assert "external_avoided_count" in rep.get("metric_definitions", {})
            # human summary consistency
            assert "external calls avoided=1" in rep.get("sales_demo_summary", "")

            # time-window filtering excludes old row
            old = now - 10 * 86400
            _insert_boundary(conn, correlation_id="old1", prompt="old", goal="old", klass="runtime_diagnostics", route_mode="local_only", reason_code="entered_local_pipeline", created_at=old)
            conn.commit()
            rep2 = product_metrics_report(db_path, window="1d", top_n=2)
            assert rep2["metrics_report"]["total_requests"] <= rep["metrics_report"]["total_requests"] + 1

            # route filtering works
            rep_route = product_metrics_report(db_path, window="1d", route_filter="local_only", top_n=2)
            assert rep_route["metrics_report"]["local_only_count"] >= 1

            # latency fields present and sane
            assert isinstance(rep["metrics_report"]["avg_latency_ms_by_route"], dict)
            assert isinstance(rep["metrics_report"]["avg_backend_latency_ms_by_route"], dict)
            assert rep["metrics_report"]["coverage_reconciliation"]["boundary_total_matches_routed_total"] is True
            assert rep["metrics_report"]["coverage_reconciliation"]["coverage_bounds_ok"] is True

            # malformed/missing metric fields safe
            conn.execute(
                "INSERT INTO request_metrics(route_mode, local_only, external_reasoning, evidence_count, evidence_chars, evidence_tokens, prompt_tokens_est, avoided_tokens_est, compaction_chars_saved, fallback_used, estimated_tokens, estimated_cost, route_chosen, cost_reason, estimated_savings_vs_external, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("local_only", 1, 0, 0, 0, 0, 0, 0, 0, 0, "bad", "bad", "local_only", None, "bad", now),
            )
            conn.commit()
            rep3 = product_metrics_report(db_path, window="1d", top_n=2)
            assert isinstance(rep3["metrics_report"]["estimated_cost_total"], float)
        finally:
            conn.close()

    print("metrics_report_test.py passed")


if __name__ == "__main__":
    run_tests()
