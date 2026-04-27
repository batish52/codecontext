from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from datetime import datetime

from .costing import estimate_evidence_tokens, estimate_project_summary_tokens, estimate_request_cost_usd, estimate_text_tokens
from .db import connect
from .usage_ledger import persist_cron_run_summary_conn
from .utils import any_word, contains_word


INTEGRATED_CLASSES = [
    "runtime_diagnostics",
    "plugin_runtime_inspection",
    "config_manifest_lookup",
    "repo_navigation",
    "simple_edit_intent",
    "log_error_triage",
    "code_review_or_engine_audit",
    "product_metrics_or_cost_analysis",
    "explain_style",
    "exact_body",
]

FRESHNESS_TARGET_CLASSES = {"runtime_diagnostics", "exact_body", "explain_style"}
BOUNDARY_COMPLIANCE_CLASSES = {
    "runtime_diagnostics",
    "plugin_runtime_inspection",
    "exact_body",
    "explain_style",
    "config_manifest_lookup",
    "repo_navigation",
    "simple_edit_intent",
    "log_error_triage",
    "code_review_or_engine_audit",
    "product_metrics_or_cost_analysis",
}


def normalize_boundary_class(name: str | None) -> str:
    text = (name or "").strip().lower()
    if text in BOUNDARY_COMPLIANCE_CLASSES:
        return text
    if text in {"symbol_or_exact_body", "exact_symbol_body"}:
        return "exact_body"
    if text in {"config_lookup", "config", "manifest_lookup"}:
        return "config_manifest_lookup"
    if text in {"project_structure", "project_summary", "repo_structure"}:
        return "repo_navigation"
    if text in {"bug_hunt", "triage"}:
        return "log_error_triage"
    if text in {"code_edit", "edit_intent"}:
        return "simple_edit_intent"
    return text or "other"


def persist_boundary_event(db_path, event: dict) -> None:
    max_attempts = 4
    for attempt in range(max_attempts):
        conn = None
        try:
            conn = connect(db_path)
            persist_boundary_event_conn(conn, event)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "database is locked" in msg and attempt < max_attempts - 1:
                time.sleep(0.05 * (2 ** attempt))
                continue
            raise
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


def persist_boundary_event_conn(conn, event: dict) -> None:
    now = time.time()

    # Robustness: ensure entered_local_pipeline carries route_perf inside reason_detail
    # even when emitters provide scheduler metrics at top-level fields.
    try:
        if str(event.get("reason_code") or "") == "entered_local_pipeline":
            detail_raw = event.get("reason_detail")
            detail_obj = json.loads(detail_raw) if isinstance(detail_raw, str) and detail_raw.strip() else {}
            if not isinstance(detail_obj, dict):
                detail_obj = {}

            rp = detail_obj.get("route_perf")
            if not isinstance(rp, dict):
                top_rp = event.get("route_perf") if isinstance(event.get("route_perf"), dict) else None
                if isinstance(top_rp, dict):
                    detail_obj["route_perf"] = top_rp
                else:
                    inferred = {}
                    for k in ("timeout_rate", "queue_wait_p95_ms", "p95_ms", "p99_ms", "fallback_rate", "downgrade_count"):
                        v = event.get(k)
                        if isinstance(v, (int, float)):
                            inferred[k] = float(v)
                    if inferred:
                        detail_obj["route_perf"] = inferred

            if detail_obj and (not isinstance(detail_raw, str) or detail_raw.strip() == "" or detail_obj.get("route_perf")):
                event["reason_detail"] = json.dumps(detail_obj, ensure_ascii=False)
    except Exception:
        pass

    conn.execute(
        """
        INSERT INTO boundary_events(
          correlation_id, session_key, request_text_raw, cleaned_goal,
          cleaned_goal_primary_source, cleaned_goal_fallback_source,
          matched_intercept_class, original_intercept_class,
          candidate_relevant, route_mode, final_route_after_override,
          complexity_override_applied, complexity_override_reason,
          classification_completed, heavy_local_handling_triggered,
          intercept_attempted, backend_call_attempted, backend_call_succeeded,
          entered_execution_runs, fallback_used, old_path_used, run_id, reason_code, reason_detail,
          source_kind, plugin_id, plugin_path, plugin_version_marker, backend_cli_path,
          created_at, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(correlation_id) DO UPDATE SET
          session_key=excluded.session_key,
          request_text_raw=excluded.request_text_raw,
          cleaned_goal=excluded.cleaned_goal,
          cleaned_goal_primary_source=COALESCE(excluded.cleaned_goal_primary_source, boundary_events.cleaned_goal_primary_source),
          cleaned_goal_fallback_source=COALESCE(excluded.cleaned_goal_fallback_source, boundary_events.cleaned_goal_fallback_source),
          matched_intercept_class=excluded.matched_intercept_class,
          original_intercept_class=COALESCE(excluded.original_intercept_class, boundary_events.original_intercept_class),
          candidate_relevant=excluded.candidate_relevant,
          route_mode=COALESCE(excluded.route_mode, boundary_events.route_mode),
          final_route_after_override=COALESCE(excluded.final_route_after_override, boundary_events.final_route_after_override),
          complexity_override_applied=excluded.complexity_override_applied,
          complexity_override_reason=COALESCE(excluded.complexity_override_reason, boundary_events.complexity_override_reason),
          classification_completed=excluded.classification_completed,
          heavy_local_handling_triggered=excluded.heavy_local_handling_triggered,
          intercept_attempted=excluded.intercept_attempted,
          backend_call_attempted=excluded.backend_call_attempted,
          backend_call_succeeded=excluded.backend_call_succeeded,
          entered_execution_runs=excluded.entered_execution_runs,
          fallback_used=excluded.fallback_used,
          old_path_used=excluded.old_path_used,
          run_id=COALESCE(excluded.run_id, boundary_events.run_id),
          reason_code=CASE
            WHEN boundary_events.reason_code = 'entered_local_pipeline' AND excluded.reason_code != 'entered_local_pipeline'
              THEN boundary_events.reason_code
            ELSE excluded.reason_code
          END,
          reason_detail=CASE
            WHEN boundary_events.reason_code = 'entered_local_pipeline' AND excluded.reason_code != 'entered_local_pipeline'
              THEN boundary_events.reason_detail
            ELSE excluded.reason_detail
          END,
          source_kind=COALESCE(excluded.source_kind, boundary_events.source_kind),
          plugin_id=COALESCE(excluded.plugin_id, boundary_events.plugin_id),
          plugin_path=COALESCE(excluded.plugin_path, boundary_events.plugin_path),
          plugin_version_marker=COALESCE(excluded.plugin_version_marker, boundary_events.plugin_version_marker),
          backend_cli_path=COALESCE(excluded.backend_cli_path, boundary_events.backend_cli_path),
          updated_at=excluded.updated_at
        """,
        (
            event["correlation_id"],
            event.get("session_key"),
            event.get("request_text_raw"),
            event.get("cleaned_goal"),
            event.get("cleaned_goal_primary_source"),
            event.get("cleaned_goal_fallback_source"),
            normalize_boundary_class(event.get("matched_intercept_class")),
            normalize_boundary_class(event.get("original_intercept_class")),
            int(bool(event.get("candidate_relevant", False))),
            event.get("route_mode"),
            event.get("final_route_after_override"),
            int(bool(event.get("complexity_override_applied", False))),
            event.get("complexity_override_reason"),
            int(bool(event.get("classification_completed", False))),
            int(bool(event.get("heavy_local_handling_triggered", False))),
            int(bool(event.get("intercept_attempted", False))),
            int(bool(event.get("backend_call_attempted", False))),
            int(bool(event.get("backend_call_succeeded", False))),
            int(bool(event.get("entered_execution_runs", False))),
            int(bool(event.get("fallback_used", False))),
            int(bool(event.get("old_path_used", False))),
            event.get("run_id"),
            event.get("reason_code"),
            event.get("reason_detail"),
            event.get("source_kind"),
            event.get("plugin_id"),
            event.get("plugin_path"),
            event.get("plugin_version_marker"),
            event.get("backend_cli_path"),
            event.get("created_at", now),
            now,
        ),
    )


def build_route_metrics(route_result: dict) -> dict:
    mode = route_result.get("mode", "unknown")
    outbound = route_result.get("outbound_payload") or {}
    repo = outbound.get("repository_context") or {}
    evidence = repo.get("evidence") or []
    project_summary = repo.get("project_summary") or route_result.get("project_summary") or {}
    evidence_chars = sum(len(item.get("snippet_text", "")) for item in evidence)
    evidence_tokens = estimate_evidence_tokens(evidence)
    project_tokens = estimate_project_summary_tokens(project_summary if isinstance(project_summary, dict) else {"summary": project_summary})
    goal_tokens = estimate_text_tokens((outbound.get("request") or {}).get("goal", route_result.get("route", {}).get("user_goal", "")))
    prompt_tokens_est = goal_tokens + project_tokens + evidence_tokens
    # Bug #24: previously this hardcoded `14000` as the naive-baseline
    # token count for savings calculations, while context_pack.py used
    # `config.naive_baseline_tokens`. If an operator tuned the config
    # value to 20_000, context_pack used 20k but the route metrics
    # still reported savings against 14k. Read the baseline from the
    # route_result (set by the gateway from AppConfig) and fall back
    # to 14000 only when the caller didn't supply one — that's the
    # historical literal, so older callers are unchanged.
    naive_baseline = int(route_result.get("naive_baseline_tokens") or 14000)
    avoided_tokens_est = max(0, naive_baseline - prompt_tokens_est) if mode == "external_reasoning" else max(0, naive_baseline - goal_tokens)
    debug = route_result.get("debug") or {}
    estimated_request_cost_usd = estimate_request_cost_usd(prompt_tokens_est) if mode == "external_reasoning" else 0.0
    estimated_external_cost_usd = estimate_request_cost_usd(max(prompt_tokens_est, naive_baseline))
    return {
        "route_mode": mode,
        "local_only": int(mode == "local_only"),
        "external_reasoning": int(mode == "external_reasoning"),
        "evidence_count": len(evidence),
        "evidence_chars": evidence_chars,
        "evidence_tokens": evidence_tokens,
        "prompt_tokens_est": prompt_tokens_est,
        "avoided_tokens_est": avoided_tokens_est,
        "compaction_chars_saved": int(debug.get("compaction_chars_saved", 0)),
        "fallback_used": int(bool(debug.get("fallback_used", False))),
        "estimated_request_cost_usd": estimated_request_cost_usd,
        "estimated_tokens": prompt_tokens_est,
        "estimated_cost": estimated_request_cost_usd,
        "route_chosen": mode,
        "cost_reason": str(debug.get("costDecisionReason") or "backend_route_metrics"),
        "estimated_savings_vs_external": max(0.0, estimated_external_cost_usd - estimated_request_cost_usd),
    }


def persist_route_metrics(db_path, metrics: dict) -> None:
    conn = connect(db_path)
    try:
        persist_route_metrics_conn(conn, metrics)
        conn.commit()
    finally:
        conn.close()


def persist_route_metrics_conn(conn, metrics: dict) -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO request_metrics(route_mode, local_only, external_reasoning, evidence_count, evidence_chars, evidence_tokens, prompt_tokens_est, avoided_tokens_est, compaction_chars_saved, fallback_used, estimated_tokens, estimated_cost, route_chosen, cost_reason, estimated_savings_vs_external, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            metrics["route_mode"],
            metrics["local_only"],
            metrics["external_reasoning"],
            metrics["evidence_count"],
            metrics["evidence_chars"],
            metrics["evidence_tokens"],
            metrics["prompt_tokens_est"],
            metrics["avoided_tokens_est"],
            metrics["compaction_chars_saved"],
            metrics["fallback_used"],
            int(metrics.get("estimated_tokens", metrics.get("prompt_tokens_est", 0))),
            float(metrics.get("estimated_cost", metrics.get("estimated_request_cost_usd", 0.0))),
            metrics.get("route_chosen", metrics.get("route_mode")),
            metrics.get("cost_reason", "backend_route_metrics"),
            float(metrics.get("estimated_savings_vs_external", 0.0)),
            now,
        ),
    )


def benchmark_summary(db_path) -> dict:
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM request_metrics ORDER BY id DESC LIMIT 200").fetchall()
        if not rows:
            return {
                "requests": 0,
                "local_only_ratio": 0.0,
                "external_ratio": 0.0,
                "avg_tokens_per_external": 0,
                "avg_evidence_chars_per_external": 0,
                "estimated_api_cost_reduction_usd": 0.0,
            }
        total = len(rows)
        local_only = sum(int(r["local_only"]) for r in rows)
        external = sum(int(r["external_reasoning"]) for r in rows)
        external_rows = [r for r in rows if int(r["external_reasoning"]) == 1]
        avg_tokens_external = round(sum(int(r["prompt_tokens_est"]) for r in external_rows) / max(1, len(external_rows)))
        avg_chars_external = round(sum(int(r["evidence_chars"]) for r in external_rows) / max(1, len(external_rows)))
        avoided = sum(int(r["avoided_tokens_est"]) for r in rows)
        avoided_cost = estimate_request_cost_usd(avoided)
        return {
            "requests": total,
            "local_only_ratio": round(local_only / total, 3),
            "external_ratio": round(external / total, 3),
            "avg_tokens_per_external": avg_tokens_external,
            "avg_evidence_chars_per_external": avg_chars_external,
            "estimated_api_cost_reduction_usd": avoided_cost,
        }
    finally:
        conn.close()


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


def _classify_integrated_request(goal: str, route: dict | None = None) -> str:
    text = (goal or "").lower()
    route = route or {}
    intent = (route.get("intent") or "").lower()

    if ("exact" in text and any(word in text for word in ("function", "class", "method", "body"))) or "exact-body" in text:
        return "exact_body"
    if intent == "runtime_diagnostics" or "how much" in text and "local" in text and "external" in text or "fallback" in text or "plugin working" in text or text.strip() == "check logs":
        return "runtime_diagnostics"
    if any(word in text for word in ("config", "setting", "settings", "manifest", "package.json", "pyproject", "requirements", ".env", " env ")) or intent == "config_lookup":
        return "config_manifest_lookup"
    if any(phrase in text for phrase in ("project tree", "repo structure", "project structure", "repo layout", "project layout")) or (intent == "project_summary" and any(word in text for word in ("tree", "structure", "layout", "files", "folders", "directories"))):
        return "repo_navigation"
    if intent == "code_edit" or any(word in text for word in ("edit", "change", "update", "modify", "rewrite", "refactor", "fix")):
        return "simple_edit_intent"
    if intent == "bug_hunt" or any(word in text for word in ("error", "exception", "traceback", "stack trace", "failing", "failure", "broken")):
        return "log_error_triage"
    if any(word in text for word in ("explain", "why", "architecture", "overview")):
        return "explain_style"
    return "other"


def _safe_avg(values: list[float | int]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _freshness_label(row: dict) -> str:
    request_class = row.get("request_class")
    route = row.get("route") or {}
    debug = row.get("debug") or {}
    goal = (row.get("goal") or "").lower()
    route_action = debug.get("route_action") or ""

    if request_class == "runtime_diagnostics":
        if route.get("intent") == "runtime_diagnostics" or route_action == "runtime_diagnostics_local_only":
            return "fresh/current"
        return "pre-fix residue"

    if request_class == "exact_body":
        if route_action == "exact_body_local_only" or debug.get("exact_body_local_only_eligible") is not None or debug.get("exact_body_details"):
            return "fresh/current"
        return "stale historical"

    if request_class == "explain_style":
        if route_action == "explain_style_local_only" or debug.get("explain_style_local_only_eligible") is not None:
            return "fresh/current"
        if route.get("intent") == "symbol_lookup" and "explain" in goal:
            return "pre-fix residue"
        return "stale historical"

    return "window aggregate"


def _aggregate_rows(rows: list[dict]) -> dict:
    total = len(rows)
    local_only = sum(1 for r in rows if r["is_local_only"])
    external = sum(1 for r in rows if r["is_external_reasoning"])
    fallback = sum(1 for r in rows if r["used_fallback"])
    return {
        "requests": total,
        "local_only_ratio": round(local_only / total, 3) if total else 0.0,
        "external_reasoning_ratio": round(external / total, 3) if total else 0.0,
        "fallback_rate": round(fallback / total, 3) if total else 0.0,
        "avg_evidence_count": _safe_avg([r["metrics"].get("evidence_count", 0) for r in rows]),
        "avg_evidence_chars": _safe_avg([r["metrics"].get("evidence_chars", 0) for r in rows]),
        "avg_estimated_prompt_tokens": _safe_avg([r["metrics"].get("prompt_tokens_est", 0) for r in rows]),
        "estimated_avoided_tokens": sum(int(r["metrics"].get("avoided_tokens_est", 0)) for r in rows),
        "estimated_cost_reduction_usd": round(estimate_request_cost_usd(sum(int(r["metrics"].get("avoided_tokens_est", 0)) for r in rows)), 6),
    }


def _load_integrated_rows(db_path, limit: int = 200) -> list[dict]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, goal, route_mode, result_json, metrics_json, created_at, updated_at FROM execution_runs ORDER BY id DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
        out = []
        for row in rows:
            result = json.loads(row["result_json"]) if row["result_json"] else {}
            metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
            route = (result or {}).get("route") or {}
            debug = (result or {}).get("debug") or {}
            request_class = _classify_integrated_request(row["goal"], route)
            row_out = {
                "id": row["id"],
                "goal": row["goal"],
                "route_mode": row["route_mode"],
                "result": result,
                "metrics": metrics,
                "route": route,
                "debug": debug,
                "request_class": request_class,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "is_local_only": row["route_mode"] == "local_only" or metrics.get("local_only") == 1,
                "is_external_reasoning": row["route_mode"] == "external_reasoning" or metrics.get("external_reasoning") == 1,
                "used_fallback": metrics.get("fallback_used") == 1 or bool(debug.get("fallback_used")),
            }
            row_out["freshness_label"] = _freshness_label(row_out)
            out.append(row_out)
        return out
    finally:
        conn.close()




def _aggregate_scheduler_observability(conn, limit: int = 400) -> dict:
    rows = conn.execute(
        "SELECT reason_detail FROM boundary_events WHERE reason_code='entered_local_pipeline' ORDER BY id DESC LIMIT ?",
        (max(1, limit),),
    ).fetchall()

    keys = ["timeout_rate", "queue_wait_p95_ms", "p95_ms", "p99_ms", "fallback_rate", "downgrade_count"]
    values = {k: [] for k in keys}

    for row in rows:
        detail = _row_get(row, "reason_detail", "")
        try:
            parsed = json.loads(str(detail or "")) if detail else None
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            continue

        route_perf = parsed.get("route_perf")
        if isinstance(route_perf, dict):
            source = route_perf
        else:
            source = parsed

        for k in keys:
            v = source.get(k)
            if isinstance(v, (int, float)):
                values[k].append(float(v))

    def pack(arr):
        if not arr:
            return {"n": 0, "latest": None, "median": None, "max": None}
        arr_f = [float(x) for x in arr]
        return {
            "n": len(arr_f),
            "latest": arr_f[0],
            "median": float(statistics.median(arr_f)),
            "max": float(max(arr_f)),
        }

    return {
        "measurement_source": "boundary_events.reason_detail.route_perf (entered_local_pipeline)",
        "window_events": len(rows),
        "timeout_rate": pack(values["timeout_rate"]),
        "queue_wait_p95_ms": pack(values["queue_wait_p95_ms"]),
        "p95_ms": pack(values["p95_ms"]),
        "p99_ms": pack(values["p99_ms"]),
        "fallback_rate": pack(values["fallback_rate"]),
        "downgrade_count": pack(values["downgrade_count"]),
    }

def integrated_path_benchmark(db_path, limit: int = 200, rows=None) -> dict:
    conn = connect(db_path)
    try:
        # Bug #5: this function previously re-called _load_integrated_rows
        # unconditionally, even when the caller (e.g. the leakage audit)
        # had already loaded the same rows for the same window. Accept
        # an optional pre-loaded `rows` so the audit path doesn't pay
        # the DB hit twice.
        if rows is None:
            rows = _load_integrated_rows(db_path, limit=limit)
        total = len(rows)
        class_breakdown = {}
        fresh_behavior_summary = {}
        for request_class in INTEGRATED_CLASSES:
            class_rows = [r for r in rows if r["request_class"] == request_class]
            class_breakdown[request_class] = _aggregate_rows(class_rows)
            if request_class in FRESHNESS_TARGET_CLASSES:
                fresh_rows = [r for r in class_rows if r["freshness_label"] == "fresh/current"]
                stale_rows = [r for r in class_rows if r["freshness_label"] != "fresh/current"]
                pre_fix_rows = [r for r in class_rows if r["freshness_label"] == "pre-fix residue"]
                # Bug #4: previously _aggregate_rows(fresh_rows) was
                # called five times in a row to pull different keys
                # out of the same dict. Cache once.
                fresh_agg = _aggregate_rows(fresh_rows)
                class_breakdown[request_class]["freshness_split"] = {
                    "window_aggregate": _aggregate_rows(class_rows),
                    "fresh_current": fresh_agg,
                    "historical_residue": _aggregate_rows(stale_rows),
                    "pre_fix_residue": _aggregate_rows(pre_fix_rows),
                }
                fresh_behavior_summary[request_class] = {
                    "fresh_requests": len(fresh_rows),
                    "fresh_local_only_ratio": fresh_agg["local_only_ratio"],
                    "fresh_external_reasoning_ratio": fresh_agg["external_reasoning_ratio"],
                    "fresh_fallback_rate": fresh_agg["fallback_rate"],
                    "fresh_avg_prompt_tokens": fresh_agg["avg_estimated_prompt_tokens"],
                    "fresh_avg_evidence_count": fresh_agg["avg_evidence_count"],
                }

        return {
            "measurement_source": "execution_runs + stored route metrics (integrated runtime path)",
            "window_requests": total,
            "totals": _aggregate_rows(rows) | {"intercepted_requests": total},
            "current_behavior_summary": fresh_behavior_summary,
            "by_request_class": class_breakdown,
            "scheduler_observability": _aggregate_scheduler_observability(conn, limit=max(200, limit * 2)),
        }
    finally:
        conn.close()


def integrated_path_leakage_audit(db_path, limit: int = 200) -> dict:
    rows = _load_integrated_rows(db_path, limit=limit)
    benchmark = integrated_path_benchmark(db_path, limit=limit, rows=rows)
    leakage = {}
    for request_class in INTEGRATED_CLASSES:
        class_rows = [r for r in rows if r["request_class"] == request_class]
        weak_evidence = [
            {
                "id": r["id"],
                "goal": r["goal"],
                "route_mode": r["route_mode"],
                "evidence_count": r["metrics"].get("evidence_count", 0),
                "prompt_tokens_est": r["metrics"].get("prompt_tokens_est", 0),
                "fallback_used": bool(r["metrics"].get("fallback_used", 0)),
                "freshness": r["freshness_label"],
            }
            for r in class_rows
            if r["is_external_reasoning"] and r["metrics"].get("evidence_count", 0) <= 1
        ]
        probable_should_have_stayed_local = [
            {
                "id": r["id"],
                "goal": r["goal"],
                "route_mode": r["route_mode"],
                "evidence_count": r["metrics"].get("evidence_count", 0),
                "route_action": r["debug"].get("route_action"),
                "freshness": r["freshness_label"],
            }
            for r in class_rows
            if r["is_external_reasoning"]
            and r["metrics"].get("evidence_count", 0) <= 1
            and request_class in {"config_manifest_lookup", "repo_navigation"}
        ]
        fallback_rows = [
            {
                "id": r["id"],
                "goal": r["goal"],
                "evidence_count": r["metrics"].get("evidence_count", 0),
                "route_action": r["debug"].get("route_action"),
                "freshness": r["freshness_label"],
            }
            for r in class_rows if r["used_fallback"]
        ]
        noisy_rows = [
            {
                "id": r["id"],
                "goal": r["goal"],
                "intent": r["route"].get("intent"),
                "route_mode": r["route_mode"],
                "freshness": r["freshness_label"],
            }
            for r in class_rows
            if request_class == "explain_style" and (r["route"].get("intent") or "") not in {"code_understanding", "project_summary"}
        ]
        freshness_counts = {
            "fresh_current": sum(1 for r in class_rows if r["freshness_label"] == "fresh/current"),
            "stale_historical": sum(1 for r in class_rows if r["freshness_label"] == "stale historical"),
            "pre_fix_residue": sum(1 for r in class_rows if r["freshness_label"] == "pre-fix residue"),
        }
        leakage[request_class] = {
            "requests_in_window": len(class_rows),
            "freshness_counts": freshness_counts,
            "requests_that_still_bypass_interception": [],
            "intercepted_but_probably_should_have_stayed_local": probable_should_have_stayed_local[:5],
            "escalated_with_weak_evidence": weak_evidence[:5],
            "frequent_fallback_examples": fallback_rows[:5],
            "routing_noise_examples": noisy_rows[:5],
        }

    ranked_cost_leak = sorted(
        (
            {
                "request_class": cls,
                "external_reasoning_ratio": stats["external_reasoning_ratio"],
                "avg_estimated_prompt_tokens": stats["avg_estimated_prompt_tokens"],
                "estimated_cost_reduction_usd": stats["estimated_cost_reduction_usd"],
            }
            for cls, stats in benchmark["by_request_class"].items() if stats["requests"] > 0
        ),
        key=lambda x: (x["external_reasoning_ratio"], x["avg_estimated_prompt_tokens"]),
        reverse=True,
    )
    freshness_views = {}
    for request_class in sorted(FRESHNESS_TARGET_CLASSES):
        class_rows = [r for r in rows if r["request_class"] == request_class]
        freshness_views[request_class] = {
            "fresh_current": _aggregate_rows([r for r in class_rows if r["freshness_label"] == "fresh/current"]),
            "stale_historical": _aggregate_rows([r for r in class_rows if r["freshness_label"] == "stale historical"]),
            "pre_fix_residue": _aggregate_rows([r for r in class_rows if r["freshness_label"] == "pre-fix residue"]),
            "fresh_examples": [
                {"id": r["id"], "goal": r["goal"], "route_mode": r["route_mode"], "freshness": r["freshness_label"]}
                for r in class_rows if r["freshness_label"] == "fresh/current"
            ][:5],
            "residue_examples": [
                {"id": r["id"], "goal": r["goal"], "route_mode": r["route_mode"], "freshness": r["freshness_label"]}
                for r in class_rows if r["freshness_label"] != "fresh/current"
            ][:5],
        }
    return {
        "measurement_source": benchmark["measurement_source"],
        "window_requests": benchmark["window_requests"],
        "leakage_by_request_class": leakage,
        "top_cost_leak_classes": ranked_cost_leak[:3],
        "freshness_views": freshness_views,
        "notes": {
            "requests_that_still_bypass_interception": "Execution_runs records integrated-path invocations only; true bypasses require comparing session traffic to runtime intercept logs, which is not yet persisted in this DB.",
            "focus": "This audit measures leakiness after interception, using the real integrated runtime path as the primary source.",
        },
    }


def boundary_compliance_report(db_path, limit: int = 500) -> dict:
    conn = connect(db_path)
    try:
        boundary_rows = conn.execute(
            "SELECT * FROM boundary_events ORDER BY id DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()
        event_rows = []
        for row in boundary_rows:
            event_rows.append({
                "id": row["id"],
                "correlation_id": row["correlation_id"],
                "session_key": row["session_key"],
                "request_text_raw": row["request_text_raw"],
                "cleaned_goal": row["cleaned_goal"],
                "matched_intercept_class": normalize_boundary_class(row["matched_intercept_class"]),
                "candidate_relevant": bool(row["candidate_relevant"]),
                "intercept_attempted": bool(row["intercept_attempted"]),
                "backend_call_attempted": bool(row["backend_call_attempted"]),
                "backend_call_succeeded": bool(row["backend_call_succeeded"]),
                "entered_execution_runs": bool(row["entered_execution_runs"]),
                "fallback_used": bool(row["fallback_used"]),
                "old_path_used": bool(row["old_path_used"]),
                "run_id": row["run_id"],
                "reason_code": row["reason_code"],
                "reason_detail": row["reason_detail"],
                "plugin_id": row["plugin_id"],
                "plugin_path": row["plugin_path"],
                "plugin_version_marker": row["plugin_version_marker"],
                "backend_cli_path": row["backend_cli_path"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })

        by_class = {}
        for request_class in sorted(BOUNDARY_COMPLIANCE_CLASSES):
            class_rows = [r for r in event_rows if r["matched_intercept_class"] == request_class and r["candidate_relevant"]]
            total_relevant = len(class_rows)
            intercepted = sum(1 for r in class_rows if r["intercept_attempted"])
            entered = sum(1 for r in class_rows if r["entered_execution_runs"])
            bypassed = sum(1 for r in class_rows if not r["entered_execution_runs"])
            fallback = sum(1 for r in class_rows if r["fallback_used"])
            old_path = sum(1 for r in class_rows if r["old_path_used"])
            backend_fail = sum(1 for r in class_rows if r["backend_call_attempted"] and not r["backend_call_succeeded"])
            by_class[request_class] = {
                "total_relevant_seen": total_relevant,
                "intercepted": intercepted,
                "entered_local_pipeline": entered,
                "bypassed": bypassed,
                "fallback_count": fallback,
                "old_path_used_count": old_path,
                "backend_failures": backend_fail,
                "compliance_rate": round(entered / total_relevant, 3) if total_relevant else 0.0,
                "examples": {
                    "bypassed": [
                        {
                            "correlation_id": r["correlation_id"],
                            "cleaned_goal": r["cleaned_goal"],
                            "reason_code": r["reason_code"],
                            "reason_detail": r["reason_detail"],
                        }
                        for r in class_rows if not r["entered_execution_runs"]
                    ][:5],
                    "entered": [
                        {
                            "correlation_id": r["correlation_id"],
                            "cleaned_goal": r["cleaned_goal"],
                            "run_id": r["run_id"],
                            "fallback_used": r["fallback_used"],
                        }
                        for r in class_rows if r["entered_execution_runs"]
                    ][:5],
                },
            }

        issue_classification = {}
        for request_class, stats in by_class.items():
            if stats["total_relevant_seen"] == 0:
                issue = "no fresh visibility yet"
            elif stats["bypassed"] > 0 and stats["backend_failures"] == 0:
                issue = "interception coverage issue"
            elif stats["backend_failures"] > 0:
                issue = "plugin/runtime enforcement issue"
            elif stats["fallback_count"] > 0:
                issue = "fallback issue"
            else:
                issue = "in-path compliant / residue outside boundary scope"
            issue_classification[request_class] = issue

        provenance_rows = [r for r in event_rows if any(r.get(k) for k in ("plugin_id", "plugin_path", "plugin_version_marker", "backend_cli_path"))]
        # Use a None-safe sort key: sort tuples by their string
        # representation with None mapped to "" so mixed None/str across
        # the four fields (which is the norm — not every row has every
        # field) doesn't raise TypeError from Python 3's None < str.
        identity_tuples = sorted(
            {
                (
                    r.get("plugin_id") or None,
                    r.get("plugin_path") or None,
                    r.get("plugin_version_marker") or None,
                    r.get("backend_cli_path") or None,
                )
                for r in provenance_rows
            },
            key=lambda t: tuple((x if x is not None else "") for x in t),
        )
        provenance_summary = {
            "events_with_provenance": len(provenance_rows),
            "events_missing_provenance": len(event_rows) - len(provenance_rows),
            "distinct_plugin_identities": [
                {
                    "plugin_id": plugin_id,
                    "plugin_path": plugin_path,
                    "plugin_version_marker": plugin_version_marker,
                    "backend_cli_path": backend_cli_path,
                }
                for plugin_id, plugin_path, plugin_version_marker, backend_cli_path in identity_tuples
            ],
            "all_events_same_identity": len(identity_tuples) <= 1 and len(provenance_rows) > 0,
            "mixed_identity_detected": len(identity_tuples) > 1,
            "canonical_identity": (
                {
                    "plugin_id": identity_tuples[0][0],
                    "plugin_path": identity_tuples[0][1],
                    "plugin_version_marker": identity_tuples[0][2],
                    "backend_cli_path": identity_tuples[0][3],
                }
                if len(identity_tuples) == 1 else None
            ),
            "sample_event_ids": [r["id"] for r in provenance_rows[:5]],
        }

        return {
            "measurement_source": "plugin/runtime boundary events compared with execution_runs entry",
            "window_events": len(event_rows),
            "boundary_provenance": provenance_summary,
            "by_request_class": by_class,
            "issue_classification": issue_classification,
            "notes": {
                "relevant_seen": "Counts relevant candidate requests observed at the plugin/runtime boundary.",
                "entered_local_pipeline": "Counts boundary events that successfully produced an execution_runs entry.",
                "bypassed": "Relevant candidate requests seen at the boundary but not entering execution_runs.",
                "boundary_provenance": "Summarizes persisted plugin identity fields from sampled boundary events so reports can directly prove which plugin path produced them.",
            },
        }
    finally:
        conn.close()


def recent_observed_prompts_audit(db_path, limit: int = 25, text_filter: str | None = None, covered_class: str | None = None, only_missed: bool = False, source_kind: str | None = None) -> dict:
    conn = connect(db_path)
    try:
        summary_sql = """
            SELECT
              COALESCE(NULLIF(TRIM(b.source_kind), ''), 'unknown') AS source_kind,
              COUNT(*) AS total
            FROM boundary_events b
            WHERE 1=1
        """
        detail_sql = """
            SELECT
              b.id,
              b.correlation_id,
              b.session_key,
              b.request_text_raw,
              b.cleaned_goal,
              b.cleaned_goal_primary_source,
              b.cleaned_goal_fallback_source,
              b.matched_intercept_class,
              b.original_intercept_class,
              b.candidate_relevant,
              b.final_route_after_override,
              b.complexity_override_applied,
              b.complexity_override_reason,
              b.intercept_attempted,
              b.backend_call_attempted,
              b.backend_call_succeeded,
              b.entered_execution_runs,
              b.fallback_used,
              b.old_path_used,
              b.run_id,
              b.reason_code,
              b.reason_detail,
              COALESCE(NULLIF(TRIM(b.source_kind), ''), 'unknown') AS source_kind,
              b.created_at,
              e.route_mode
            FROM boundary_events b
            LEFT JOIN execution_runs e ON e.id = b.run_id
            WHERE 1=1
        """
        filter_clauses: list[str] = []
        filter_params: list[object] = []

        if text_filter:
            filter_clauses.append("(LOWER(COALESCE(b.request_text_raw, '')) LIKE ? OR LOWER(COALESCE(b.cleaned_goal, '')) LIKE ?)")
            needle = f"%{text_filter.lower()}%"
            filter_params.extend([needle, needle])

        if covered_class:
            filter_clauses.append("b.matched_intercept_class = ?")
            filter_params.append(normalize_boundary_class(covered_class))

        if source_kind:
            filter_clauses.append("COALESCE(NULLIF(TRIM(b.source_kind), ''), 'unknown') = ?")
            filter_params.append(source_kind.strip())

        if only_missed:
            filter_clauses.append("(b.entered_execution_runs = 0 OR b.intercept_attempted = 0 OR b.backend_call_succeeded = 0 OR b.old_path_used = 1)")

        if filter_clauses:
            clause = " AND " + " AND ".join(filter_clauses)
            summary_sql += clause
            detail_sql += clause

        summary_sql += " GROUP BY COALESCE(NULLIF(TRIM(b.source_kind), ''), 'unknown') ORDER BY total DESC, source_kind ASC"
        detail_sql += " ORDER BY b.id DESC LIMIT ?"

        summary_rows = conn.execute(summary_sql, filter_params).fetchall()
        rows = conn.execute(detail_sql, [*filter_params, max(1, int(limit))]).fetchall()

        observed = []
        for row in rows:
            raw_prompt = row["request_text_raw"] or ""
            observed.append({
                "id": row["id"],
                "timestamp": datetime.fromtimestamp(float(row["created_at"])).isoformat() if row["created_at"] else None,
                "raw_prompt_text": raw_prompt[:400],
                "raw_prompt_truncated": len(raw_prompt) > 400,
                "cleaned_goal": row["cleaned_goal"],
                "cleaned_goal_primary_source": row["cleaned_goal_primary_source"],
                "cleaned_goal_fallback_source": row["cleaned_goal_fallback_source"],
                "candidate_relevant": bool(row["candidate_relevant"]),
                "matched_intercept_class": normalize_boundary_class(row["matched_intercept_class"]),
                "original_intercept_class": normalize_boundary_class(row["original_intercept_class"]),
                "old_intent_label": row["reason_code"],
                # Bug #29: this used to be exposed as "task_type",
                # but the value is actually the persisted reason_code
                # (e.g. "candidate_relevant_soft", "entered_local_pipeline"),
                # not a CanonicalTaskType. Consumers who tried to
                # compare it to "explain_mechanism"/"diagnose_runtime"
                # silently got no matches. Rename to task_type_proxy
                # to make the semantics explicit, and keep the legacy
                # "task_type" key briefly for backward compatibility.
                "task_type_proxy": row["reason_code"],
                "task_type": row["reason_code"],  # deprecated alias of task_type_proxy
                "evidence_source_type": row["reason_detail"],
                "source_kind": row["source_kind"],
                "intercept_attempted": bool(row["intercept_attempted"]),
                "backend_call_attempted": bool(row["backend_call_attempted"]),
                "backend_call_succeeded": bool(row["backend_call_succeeded"]),
                "entered_execution_runs": bool(row["entered_execution_runs"]),
                "run_id": row["run_id"],
                "final_route_mode": row["route_mode"],
                "final_route_after_override": row["final_route_after_override"],
                "complexity_override_applied": bool(row["complexity_override_applied"]),
                "complexity_override_reason": row["complexity_override_reason"],
                "fallback_used": bool(row["fallback_used"]),
                "old_path_used": bool(row["old_path_used"]),
                "reason_code": row["reason_code"],
                "reason_detail": row["reason_detail"],
            })

        source_kind_summary = {row["source_kind"]: int(row["total"]) for row in summary_rows}

        return {
            "recent_observed_prompts_audit": {
                "measurement_source": "persisted boundary_events left-joined to execution_runs when present",
                "filters": {
                    "limit": max(1, int(limit)),
                    "text_filter": text_filter,
                    "covered_class": normalize_boundary_class(covered_class) if covered_class else None,
                    "only_missed": bool(only_missed),
                    "source_kind": source_kind.strip() if source_kind else None,
                },
                "total_observed": len(observed),
                "source_kind_counts": source_kind_summary,
                "status_buckets": {
                    "observed_and_entered_execution_runs": sum(1 for r in observed if r["entered_execution_runs"]),
                    "observed_not_intercepted": sum(1 for r in observed if not r["intercept_attempted"]),
                    "observed_backend_failed": sum(1 for r in observed if r["backend_call_attempted"] and not r["backend_call_succeeded"]),
                    "observed_old_path_used": sum(1 for r in observed if r["old_path_used"]),
                    "observed_intercepted_not_entered": sum(1 for r in observed if r["intercept_attempted"] and not r["entered_execution_runs"]),
                },
                "recent_prompts": observed,
                "notes": {
                    "missing_recent_prompts": "If total_observed is 0, then no matching prompts were found in persisted boundary_events for the selected filters.",
                    "old_intent_label": "This report reuses persisted reason_code as the nearest available legacy/intention label from boundary data.",
                    "task_type_and_evidence_source_type": "Boundary persistence does not currently store canonical task_type/evidence_source_type separately for every request, so this report surfaces the nearest persisted boundary labels and final route mode when present.",
                    "source_kind": "source_kind distinguishes live plugin traffic from controlled or audit-generated traffic when the boundary writer provides that origin.",
                },
            }
        }
    finally:
        conn.close()



def _classify_runtime_operational_debug_prompt(prompt: str) -> dict | None:
    text = (prompt or "").strip()
    lowered = text.lower()
    if not lowered:
        return None

    if any(phrase in lowered for phrase in [
        "architecture tradeoffs",
        "how does the codecontext routing gateway work",
        "project tree",
        "package.json settings",
        "config setting",
        "exact searchengine",
        "bm25",
        "symbol retrieval",
    ]):
        return None

    # Bug #6: substring matches on `"log"` here matched
    # `"login"`/`"blog"`/`"dialog"`/`"catalog"`, and `"health"`
    # matched `"healthcare"`/`"wealth"`. Use word-boundary matching
    # for single-token signals while keeping multi-word phrases as
    # substring checks (those are uniquely identifying by shape).
    if "traceback" in lowered:
        evidence_source_type = "traceback"
        likely_routing = "external"
        family = "traceback"
    elif any_word(lowered, ("startup", "started", "boot", "initialize")) or "startup complete" in lowered or re.search(r"\binitializ", lowered):
        evidence_source_type = "runtime_startup_path"
        family = "runtime_startup"
        likely_routing = "external" if any_word(lowered, ("why", "fail", "failed", "stuck")) else "local_only"
    elif any_word(lowered, ("log", "logs")) or "what logs should i inspect" in lowered:
        evidence_source_type = "runtime_logs"
        family = "runtime_logs"
        likely_routing = "external" if contains_word(lowered, "why") else "local_only"
    elif any_word(lowered, ("fallback",)) or "local vs external" in lowered or "execution trace" in lowered:
        evidence_source_type = "runtime_execution_trace"
        family = "fallback_status" if contains_word(lowered, "fallback") else "route_mix_summary"
        likely_routing = "external" if contains_word(lowered, "why") else "local_only"
    elif (
        "is the plugin working" in lowered
        or any_word(lowered, (
            "status", "health", "running", "run_live_engine", "engine",
            "process", "stall", "stalled", "idle",
        ))
        or "doing nothing" in lowered
    ):
        evidence_source_type = "runtime_process_state"
        family = "runtime_status"
        likely_routing = "external" if (
            contains_word(lowered, "why")
            or any_word(lowered, ("stall", "stalled", "idle"))
            or "doing nothing" in lowered
        ) else "local_only"
    else:
        return None

    return {
        "task_type": "runtime_operational_debug",
        "evidence_source_type": evidence_source_type,
        "likely_routing": likely_routing,
        "family": family,
    }



def runtime_operational_debug_audit(db_path, limit: int = 500, capture_cron_summary: bool = True, cron_job_name: str = "runtime-operational-debug-soak-audit", window_seconds: int = 3 * 3600) -> dict:
    conn = connect(db_path)
    try:
        boundary_rows = conn.execute(
            "SELECT * FROM boundary_events ORDER BY id DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()

        candidates = []
        for row in boundary_rows:
            prompt = row["cleaned_goal"] or row["request_text_raw"] or ""
            classified = _classify_runtime_operational_debug_prompt(prompt)
            if not classified:
                continue
            candidates.append({
                "id": row["id"],
                "prompt": prompt,
                "matched_intercept_class": normalize_boundary_class(row["matched_intercept_class"]),
                "run_id": row["run_id"],
                "reason_detail": row["reason_detail"],
                "task_type": classified["task_type"],
                "evidence_source_type": classified["evidence_source_type"],
                "likely_routing": classified["likely_routing"],
                "family": classified["family"],
                "created_at": row["created_at"],
            })

        evidence_dist = {}
        family_dist = {}
        for item in candidates:
            evidence_dist[item["evidence_source_type"]] = evidence_dist.get(item["evidence_source_type"], 0) + 1
            family_dist[item["family"]] = family_dist.get(item["family"], 0) + 1

        local_examples = list(dict.fromkeys(item["prompt"] for item in candidates if item["likely_routing"] == "local_only"))[:5]
        external_examples = list(dict.fromkeys(item["prompt"] for item in candidates if item["likely_routing"] == "external"))[:5]

        cron_summary = None
        if capture_cron_summary:
            try:
                with conn:
                    cron_summary = persist_cron_run_summary_conn(
                        conn,
                        cron_job_name=cron_job_name,
                        window_seconds=max(1, int(window_seconds)),
                        status="ok",
                        notes="auto-captured during runtime-operational-debug-audit",
                    )
            except Exception as exc:
                cron_summary = {"error": str(exc)}

        return {
            "runtime_operational_debug_audit": {
                "measurement_source": "persisted boundary_events only",
                "total_prompts": len(candidates),
                "evidence_source_type_distribution": evidence_dist,
                "families": family_dist,
                "likely_local_only": sum(1 for item in candidates if item["likely_routing"] == "local_only"),
                "likely_external": sum(1 for item in candidates if item["likely_routing"] == "external"),
                "representative_examples": {
                    "local_only": local_examples,
                    "external": external_examples,
                },
                "sample_candidates": candidates[:10],
                "cron_usage_summary": cron_summary,
                "notes": {
                    "scope": "Observability-only audit over persisted boundary events.",
                    "routing_policy": "This report does not change routing or interception behavior.",
                    "finance_ledger": "Per-run cron savings summary is persisted from request_usage_ledger rows.",
                },
            }
        }
    finally:
        conn.close()



def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _row_get(row, key: str, default=None):
    try:
        if hasattr(row, "keys") and key in row.keys():
            return row[key]
        return default
    except Exception:
        return default


def _parse_window_seconds(window: str | None) -> int | None:
    if not window:
        return None
    text = str(window).strip().lower()
    m = re.match(r"^(\d+)([smhdw])$", text)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return n * mult


def _metric_definitions() -> dict:
    return {
        "total_requests": "Count of boundary_events rows in scope where route_mode is one of pass_through_direct/local_only/external_reasoning_with_compaction.",
        "pass_through_direct_count": "Count of boundary_events with route_mode=pass_through_direct.",
        "local_only_count": "Count of boundary_events with route_mode=local_only.",
        "external_reasoning_with_compaction_count": "Count of boundary_events with route_mode=external_reasoning_with_compaction.",
        "avg_latency_ms_by_route": "Average (execution_runs.updated_at - execution_runs.created_at)*1000 grouped by route_mode (external_reasoning mapped to external_reasoning_with_compaction).",
        "avg_backend_latency_ms_by_route": "Average backend duration_ms parsed from boundary_events.reason_detail JSON for backend call rows when available.",
        "avg_end_to_end_latency_ms_by_route": "Alias of avg_latency_ms_by_route from execution_runs lifecycle timings.",
        "avg_continue_latency_ms": "Average duration_ms for post_output_continue backend calls from boundary failure/success details when available; null if unavailable.",
        "estimated_outbound_tokens_total": "Sum(request_metrics.estimated_tokens fallback request_metrics.prompt_tokens_est) in scope.",
        "estimated_cost_total": "Sum(request_metrics.estimated_cost fallback request_metrics.estimated_request_cost_usd) in scope.",
        "estimated_savings_total": "Sum(request_metrics.estimated_savings_vs_external) in scope.",
        "estimated_savings_by_route": "Sum(request_metrics.estimated_savings_vs_external) grouped by route_chosen/route_mode.",
        "threshold_hit_count": "Count of request_metrics rows whose cost_reason contains 'high_cost'.",
        "failure_counts_by_type": "Counts of backend_call_failed rows classified by stage/error from boundary_events.reason_detail.",
        "local_handled_percentage": "100 * local_only_count / (local_only_count + external_reasoning_with_compaction_count). pass_through excluded.",
        "external_avoided_count": "Count of relevant local_only boundary rows that entered execution_runs successfully (candidate_relevant=1, route_mode=local_only, entered_execution_runs=1, backend_call_succeeded=1).",
        "soft_routed_count": "Count of medium-confidence soft-routing boundary events (local_try_then_fallback or soft reason codes).",
        "recovered_from_pass_through_count": "Approximate count of requests recovered from weak/soft relevance into local pipeline handling.",
        "effective_coverage_percent": "Coverage proxy = (local_only + external_reasoning_with_compaction + soft_routed_count) / total_observed_path_events.",
        "effective_coverage_numerator": "Numerator used for effective_coverage_percent: local_only + external_reasoning_with_compaction + soft_routed_count.",
        "effective_coverage_denominator": "Denominator used for effective_coverage_percent: routed_total + soft_routed_count.",
        "coverage_reconciliation": "Integrity check payload proving coverage numerator/denominator reconcile with boundary_events route totals.",
        "top_expensive_prompts": "Top-N execution_runs by estimated cost from metrics_json.",
        "top_expensive_classes": "Top-N classes by estimated cost using boundary_events class mapped by run_id.",
    }


def product_metrics_report(db_path, window: str | None = None, route_filter: str | None = None, top_n: int = 5, include_definitions: bool = False) -> dict:
    conn = connect(db_path)
    try:
        return _product_metrics_report_impl(conn, window, route_filter, top_n, include_definitions)
    finally:
        conn.close()


def _product_metrics_report_impl(conn, window, route_filter, top_n, include_definitions):
    now = time.time()
    window_seconds = _parse_window_seconds(window)
    cutoff = now - window_seconds if window_seconds else None

    where_metric = " WHERE created_at >= ?" if cutoff else ""
    where_runs = " WHERE created_at >= ?" if cutoff else ""
    where_boundary = " WHERE created_at >= ?" if cutoff else ""
    params = (cutoff,) if cutoff else tuple()

    metric_rows = conn.execute(f"SELECT * FROM request_metrics{where_metric} ORDER BY id DESC LIMIT 2000", params).fetchall()
    run_rows = conn.execute(f"SELECT id, goal, route_mode, intent, created_at, updated_at, metrics_json FROM execution_runs{where_runs} ORDER BY id DESC LIMIT 2000", params).fetchall()
    boundary_rows = conn.execute(f"SELECT id, cleaned_goal, request_text_raw, route_mode, matched_intercept_class, candidate_relevant, entered_execution_runs, reason_code, reason_detail, backend_call_succeeded, run_id, created_at FROM boundary_events{where_boundary} ORDER BY id DESC LIMIT 4000", params).fetchall()

    def route_ok(r: str) -> bool:
        if not route_filter:
            return True
        return (r or "") == route_filter

    pass_through_count = 0
    local_only_count = 0
    external_compaction_count = 0
    soft_routed_count = 0
    recovered_from_pass_through_count = 0
    threshold_hit_count = 0
    failure_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    backend_latency_by_route: dict[str, dict] = {}
    continue_latency_ms_values: list[float] = []
    run_id_to_class: dict[int, str] = {}

    for b in boundary_rows:
        cls = normalize_boundary_class(b["matched_intercept_class"])
        class_counts[cls] = class_counts.get(cls, 0) + 1
        mode = (b["route_mode"] or "").strip()
        if mode == "pass_through_direct":
            pass_through_count += 1
        elif mode == "local_only":
            local_only_count += 1
        elif mode == "external_reasoning_with_compaction":
            external_compaction_count += 1

        reason_code = str(b["reason_code"] or "")
        reason_detail = str(b["reason_detail"] or "")

        # Soft-route observability: count medium-confidence safe
        # auto-engage attempts and outcomes. An event counts as
        # soft-routed if EITHER its mode is local_try_then_fallback OR
        # its reason_code is one of the soft-route markers. Previously
        # this branched twice for the same event (mode check AND
        # reason-code check), inflating the total when both fired.
        is_soft_routed = (
            mode == "local_try_then_fallback"
            or reason_code in {
                "candidate_relevant_soft",
                "soft_candidate_detected",
                "soft_route_selected",
                "soft_route_non_local_result",
                "soft_route_fallback_non_local_result",
            }
        )
        if is_soft_routed:
            soft_routed_count += 1

        # Recovery observability: a prompt that previously would have dropped to pass-through
        # but is now kept in handled paths (local or compact external) via soft-route continuation.
        if reason_code == "soft_route_non_local_result":
            recovered_from_pass_through_count += 1
        elif reason_code == "fallback_reason":
            parsed_reason = None
            try:
                parsed_reason = json.loads(reason_detail) if reason_detail else None
            except Exception:
                parsed_reason = None
            if isinstance(parsed_reason, dict):
                if str(parsed_reason.get("fallback_reason") or "") == "soft_route_non_local_result":
                    recovered_from_pass_through_count += 1

        if _to_int(b["run_id"], 0) > 0:
            run_id_to_class[_to_int(b["run_id"], 0)] = cls

        detail = str(b["reason_detail"] or "")
        parsed = None
        try:
            parsed = json.loads(detail) if detail else None
        except Exception:
            parsed = None

        if b["reason_code"] == "backend_call_failed":
            ftype = "backend_failure"
            if isinstance(parsed, dict):
                ftype = str(parsed.get("stage") or parsed.get("error") or "backend_failure")
            elif "ETIMEDOUT" in detail:
                ftype = "timeout"
            elif "UnicodeEncodeError" in detail:
                ftype = "encoding_error"
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

        if isinstance(parsed, dict) and _to_float(parsed.get("durationMs"), -1) >= 0:
            dms = _to_float(parsed.get("durationMs"), 0.0)
            op = str(parsed.get("opName") or "")
            route_for_backend = mode or "other"
            bucket = backend_latency_by_route.setdefault(route_for_backend, {"count": 0, "total_ms": 0.0})
            bucket["count"] += 1
            bucket["total_ms"] += dms
            if op.startswith("auto-continue-request"):
                continue_latency_ms_values.append(dms)

    total_metrics = 0
    estimated_tokens_total = 0
    estimated_cost_total = 0.0
    estimated_savings_total = 0.0
    # Bug #30: previously this dict initialized only three routes,
    # so savings from `local_try_then_fallback` events were counted
    # in the total but disappeared from the per-route breakdown.
    savings_by_route: dict[str, float] = {
        "pass_through_direct": 0.0,
        "local_only": 0.0,
        "external_reasoning_with_compaction": 0.0,
        "local_try_then_fallback": 0.0,
    }

    for m in metric_rows:
        route = str(_row_get(m, "route_chosen", None) or _row_get(m, "route_mode", "") or "")
        if not route_ok(route):
            continue
        total_metrics += 1
        estimated_tokens_total += _to_int(_row_get(m, "estimated_tokens", None), _to_int(_row_get(m, "prompt_tokens_est", 0), 0))
        estimated_cost_total += _to_float(_row_get(m, "estimated_cost", None), _to_float(_row_get(m, "estimated_request_cost_usd", 0.0), 0.0))
        sv = _to_float(_row_get(m, "estimated_savings_vs_external", 0.0), 0.0)
        estimated_savings_total += sv
        if route in savings_by_route:
            savings_by_route[route] += sv
        if "high_cost" in str(_row_get(m, "cost_reason", "") or ""):
            threshold_hit_count += 1

    latency_by_route: dict[str, dict] = {}
    expensive_prompts: list[dict] = []
    expensive_classes: dict[str, float] = {}
    for r in run_rows:
        route_mode = "external_reasoning_with_compaction" if (r["route_mode"] == "external_reasoning") else (r["route_mode"] or "")
        if not route_ok(route_mode):
            continue
        latency_ms = max(0.0, (_to_float(r["updated_at"]) - _to_float(r["created_at"])) * 1000.0)
        # Exclude malformed/stale outliers from route-latency averages.
        if latency_ms <= 0 or latency_ms > (30 * 60 * 1000):
            continue
        bucket = latency_by_route.setdefault(route_mode, {"count": 0, "total_ms": 0.0})
        bucket["count"] += 1
        bucket["total_ms"] += latency_ms

        metrics = {}
        try:
            metrics = json.loads(r["metrics_json"] or "{}")
        except Exception:
            metrics = {}
        est_cost = _to_float(metrics.get("estimated_cost"), _to_float(metrics.get("estimated_request_cost_usd"), 0.0))
        prompt = str(r["goal"] or "")
        if est_cost > 0:
            rid = _to_int(r["id"])
            expensive_prompts.append({
                "id": rid,
                "goal": prompt,
                "route": route_mode,
                "estimated_cost": round(est_cost, 6),
            })
            cls = run_id_to_class.get(rid, "other")
            expensive_classes[cls] = expensive_classes.get(cls, 0.0) + est_cost

    avg_latency_by_route = {
        route: round(v["total_ms"] / max(1, v["count"]), 2)
        for route, v in latency_by_route.items()
    }
    avg_backend_latency_by_route = {
        route: round(v["total_ms"] / max(1, v["count"]), 2)
        for route, v in backend_latency_by_route.items()
    }
    avg_continue_latency_ms = round(sum(continue_latency_ms_values) / len(continue_latency_ms_values), 2) if continue_latency_ms_values else None

    expensive_prompts.sort(key=lambda x: x["estimated_cost"], reverse=True)
    top_expensive_prompts = expensive_prompts[: max(1, top_n)]
    top_classes_by_frequency = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_n)]
    top_expensive_classes = sorted(
        [{"class": k, "estimated_cost": round(v, 6)} for k, v in expensive_classes.items()],
        key=lambda x: x["estimated_cost"],
        reverse=True,
    )[: max(1, top_n)]

    routed_total = pass_through_count + local_only_count + external_compaction_count
    handled_total = local_only_count + external_compaction_count
    local_handled_pct = round((local_only_count / max(1, handled_total)) * 100.0, 2)
    effective_coverage_numerator = local_only_count + external_compaction_count + soft_routed_count
    effective_coverage_denominator = routed_total + soft_routed_count
    effective_coverage_percent = round((effective_coverage_numerator / max(1, effective_coverage_denominator)) * 100.0, 2)
    coverage_reconciliation = {
        "numerator": effective_coverage_numerator,
        "denominator": effective_coverage_denominator,
        "routed_total": routed_total,
        "boundary_total_considered": pass_through_count + local_only_count + external_compaction_count,
        "boundary_total_matches_routed_total": (pass_through_count + local_only_count + external_compaction_count) == routed_total,
        "coverage_bounds_ok": 0 <= effective_coverage_numerator <= max(1, effective_coverage_denominator),
    }
    external_avoided_count = sum(
        1
        for b in boundary_rows
        if (b["route_mode"] == "local_only"
            and _to_int(b["candidate_relevant"], 0) == 1
            and _to_int(b["entered_execution_runs"], 0) == 1
            and _to_int(b["backend_call_succeeded"], 0) == 1)
    )

    summary = {
        "window": window or "all",
        "total_requests": routed_total,
        "pass_through_direct_count": pass_through_count,
        "local_only_count": local_only_count,
        "external_reasoning_with_compaction_count": external_compaction_count,
        "avg_latency_ms_by_route": avg_latency_by_route,
        "avg_end_to_end_latency_ms_by_route": avg_latency_by_route,
        "avg_backend_latency_ms_by_route": avg_backend_latency_by_route,
        "avg_continue_latency_ms": avg_continue_latency_ms,
        "estimated_outbound_tokens_total": estimated_tokens_total,
        "estimated_cost_total": round(estimated_cost_total, 6),
        "estimated_savings_total": round(estimated_savings_total, 6),
        "estimated_savings_by_route": {k: round(v, 6) for k, v in savings_by_route.items()},
        "threshold_hit_count": threshold_hit_count,
        "failure_counts_by_type": failure_counts,
        "top_classes_by_frequency": [{"class": k, "count": v} for k, v in top_classes_by_frequency],
        "top_expensive_prompts": top_expensive_prompts,
        "top_expensive_classes": top_expensive_classes,
        "local_handled_percentage": local_handled_pct,
        "external_avoided_count": external_avoided_count,
        "soft_routed_count": soft_routed_count,
        "recovered_from_pass_through_count": recovered_from_pass_through_count,
        "effective_coverage_percent": effective_coverage_percent,
        "effective_coverage_numerator": effective_coverage_numerator,
        "effective_coverage_denominator": effective_coverage_denominator,
        "coverage_reconciliation": coverage_reconciliation,
    }

    human = (
        f"In the last {summary['window']} requests: "
        f"total estimated token savings={summary['estimated_outbound_tokens_total']}, "
        f"total estimated dollar savings={summary['estimated_savings_total']:.6f} USD, "
        f"local-only handling rate={local_handled_pct}%, "
        f"external calls avoided={external_avoided_count}, "
        f"avg end-to-end latency by route={avg_latency_by_route}"
    )

    out = {
        "metrics_report": summary,
        "sales_demo_summary": human,
        "notes": {
            "route_filter": route_filter,
            "top_n": top_n,
            "window_seconds": window_seconds,
            "measurement_source": "boundary_events + execution_runs + request_metrics",
        },
    }
    if include_definitions:
        out["metric_definitions"] = _metric_definitions()
    return out


def runtime_diagnostics_summary(db_path, goal: str, limit: int = 8) -> dict:
    conn = connect(db_path)
    try:
        lowered = (goal or "").lower()
        metric_rows = conn.execute("SELECT * FROM request_metrics ORDER BY id DESC LIMIT 200").fetchall()
        run_rows = conn.execute(
            "SELECT id, goal, route_mode, result_json, metrics_json, created_at, updated_at FROM execution_runs ORDER BY id DESC LIMIT ?",
            (max(1, limit),),
        ).fetchall()

        benchmark = benchmark_summary(db_path)
        total = len(metric_rows)
        local_only = sum(int(r["local_only"]) for r in metric_rows)
        external = sum(int(r["external_reasoning"]) for r in metric_rows)
        fallback_count = sum(int(r["fallback_used"]) for r in metric_rows)

        recent_runs = []
        recent_intercepts = 0
        recent_local_only = 0
        recent_external = 0
        recent_fallbacks = 0
        for row in run_rows:
            result = json.loads(row["result_json"]) if row["result_json"] else {}
            metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
            debug = (result or {}).get("debug") or {}
            route = (result or {}).get("route") or {}
            mode = row["route_mode"] or (result or {}).get("mode")
            intercepted = mode in {"local_only", "external_reasoning"}
            if intercepted:
                recent_intercepts += 1
            if mode == "local_only":
                recent_local_only += 1
            if mode == "external_reasoning":
                recent_external += 1
            if debug.get("fallback_used") or metrics.get("fallback_used"):
                recent_fallbacks += 1
            recent_runs.append({
                "id": row["id"],
                "goal": row["goal"],
                "mode": mode,
                "intent": route.get("intent"),
                "route_action": debug.get("route_action"),
                "fallback_used": bool(debug.get("fallback_used") or metrics.get("fallback_used")),
                "evidence_count": debug.get("evidence_count", metrics.get("evidence_count")),
                "created_at": _iso(row["created_at"]),
                "updated_at": _iso(row["updated_at"]),
            })

        summary = {
            "question": goal,
            "benchmark": benchmark,
            "totals": {
                "tracked_requests": total,
                "local_only_requests": local_only,
                "external_reasoning_requests": external,
                "fallback_requests": fallback_count,
            },
            "recent_window": {
                "runs_examined": len(recent_runs),
                "recent_intercepts": recent_intercepts,
                "recent_local_only": recent_local_only,
                "recent_external_reasoning": recent_external,
                "recent_fallbacks": recent_fallbacks,
            },
            "plugin_status": {
                "working": recent_intercepts > 0 or total > 0,
                "basis": "execution_runs and request_metrics",
                "last_run_at": recent_runs[0]["updated_at"] if recent_runs else None,
            },
            "recent_runs": recent_runs,
            "evidence_policy": {
                "mode": "diagnostics_local_only",
                "suppressed_generic_repo_evidence": True,
                "preferred_sources": ["request_metrics", "execution_runs", "route debug", "benchmark summary"],
            },
        }

        if "local" in lowered and "external" in lowered:
            summary["answer_focus"] = {
                "kind": "local_vs_external",
                "local_only_ratio": benchmark.get("local_only_ratio", 0.0),
                "external_ratio": benchmark.get("external_ratio", 0.0),
                "local_only_requests": local_only,
                "external_reasoning_requests": external,
            }
        elif "fallback" in lowered:
            summary["answer_focus"] = {
                "kind": "fallback",
                "fallback_requests": fallback_count,
                "recent_fallbacks": recent_fallbacks,
                "did_fallback_happen": fallback_count > 0,
            }
        elif "log" in lowered or "debug" in lowered:
            summary["answer_focus"] = {
                "kind": "logs",
                "recent_runs": recent_runs,
                "note": "No separate log file lookup was needed; this answer uses local run-state and telemetry first.",
            }
        elif "working" in lowered or "plugin" in lowered or "intercept" in lowered:
            summary["answer_focus"] = {
                "kind": "plugin_status",
                "working": summary["plugin_status"]["working"],
                "last_run_at": summary["plugin_status"]["last_run_at"],
                "recent_intercepts": recent_intercepts,
            }
        else:
            summary["answer_focus"] = {
                "kind": "general_runtime_diagnostics",
                "working": summary["plugin_status"]["working"],
                "recent_intercepts": recent_intercepts,
                "recent_fallbacks": recent_fallbacks,
            }
        return summary
    finally:
        conn.close()
