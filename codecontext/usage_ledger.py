from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from .costing import estimate_text_tokens

CALC_VERSION = "usage-ledger-v1"
DEFAULT_CURRENCY = "USD"


def _stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _tokenize(text: str, model_name: str | None) -> tuple[int, str]:
    model_name = (model_name or "").strip()
    # Exact tokenizer when available.
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.encoding_for_model(model_name or "gpt-4")
        return len(enc.encode(text or "")), "tiktoken"
    except Exception:
        # Reproducible fallback.
        return estimate_text_tokens(text or ""), "approx_tokens"


def _pricing_for(provider: str | None, model: str | None) -> tuple[float, float, str]:
    provider = (provider or "openai-codex").strip()
    model = (model or "gpt-5.3-codex").strip()
    # Conservative defaults; explicit snapshot persisted for auditability.
    # input/output are USD per 1M tokens.
    if "gpt-5" in model:
        return 5.0, 15.0, f"default:{provider}/{model}"
    return 5.0, 15.0, f"default:{provider}/{model}"


def ensure_pricing_snapshot_conn(conn, provider: str | None, model_name: str | None) -> str:
    input_per_1m, output_per_1m, source = _pricing_for(provider, model_name)
    now = time.time()
    key = f"{provider or 'openai-codex'}|{model_name or 'gpt-5.3-codex'}|{input_per_1m}|{output_per_1m}|{DEFAULT_CURRENCY}"
    snapshot_id = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]
    conn.execute(
        """
        INSERT INTO pricing_snapshots(
          pricing_snapshot_id, provider, model_name,
          input_price_per_1m_tokens, output_price_per_1m_tokens,
          cached_input_price_per_1m_tokens, currency,
          effective_from, effective_to, source, created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(pricing_snapshot_id) DO NOTHING
        """,
        (
            snapshot_id,
            provider or "openai-codex",
            model_name or "gpt-5.3-codex",
            input_per_1m,
            output_per_1m,
            input_per_1m,
            DEFAULT_CURRENCY,
            now,
            None,
            source,
            now,
        ),
    )
    return snapshot_id


def _request_class(goal: str, intent: str | None = None) -> str:
    text = (goal or "").lower()
    intent = (intent or "").lower()
    if ("exact" in text and any(w in text for w in ("function", "class", "method", "body"))) or "exact-body" in text:
        return "exact_body"
    if intent == "runtime_diagnostics" or "fallback" in text or "plugin working" in text:
        return "runtime_diagnostics"
    if any(w in text for w in ("config", "setting", "manifest", "package.json", "requirements", ".env")):
        return "config_manifest_lookup"
    if any(w in text for w in ("project tree", "repo structure", "layout", "folders", "directories")):
        return "repo_navigation"
    if intent == "code_edit" or any(w in text for w in ("edit", "change", "update", "modify", "rewrite", "refactor", "fix")):
        return "simple_edit_intent"
    if intent == "bug_hunt" or any(w in text for w in ("error", "exception", "traceback", "failure", "broken")):
        return "log_error_triage"
    if any(w in text for w in ("explain", "why", "architecture", "overview")):
        return "explain_style"
    return "other"


def record_request_usage_conn(
    conn,
    *,
    goal: str,
    route_mode: str,
    run_id: int | None,
    session_key: str | None = None,
    cron_run_id: str | None = None,
    request_id: str | None = None,
    provider: str | None = None,
    actual_model_name: str | None = None,
    counterfactual_target_model: str | None = None,
    metrics: dict | None = None,
    fallback_used: bool = False,
    reason_code: str | None = None,
    reason_detail: str | None = None,
    intent: str | None = None,
) -> dict:
    now = time.time()
    metrics = metrics or {}
    request_id = request_id or str(uuid.uuid4())
    provider = provider or "openai-codex"
    actual_model_name = actual_model_name or "gpt-5.3-codex"
    counterfactual_target_model = counterfactual_target_model or actual_model_name

    pricing_snapshot_id = ensure_pricing_snapshot_conn(conn, provider, actual_model_name)
    row = conn.execute(
        "SELECT input_price_per_1m_tokens, output_price_per_1m_tokens FROM pricing_snapshots WHERE pricing_snapshot_id=?",
        (pricing_snapshot_id,),
    ).fetchone()
    in_price = _to_float(row[0] if row else 5.0)
    out_price = _to_float(row[1] if row else 15.0)

    prompt_tokens_actual = _to_int(metrics.get("prompt_tokens_actual"), _to_int(metrics.get("prompt_tokens_est"), 0))
    completion_tokens_actual = _to_int(metrics.get("completion_tokens_actual"), 0)
    total_tokens_actual = prompt_tokens_actual + completion_tokens_actual

    cf_prompt_tokens = _to_int(metrics.get("counterfactual_prompt_tokens"), 0)
    cf_completion_tokens = _to_int(metrics.get("counterfactual_completion_tokens"), 0)
    token_method = metrics.get("estimation_method")
    if cf_prompt_tokens <= 0:
        cf_prompt_tokens, tok_method = _tokenize(goal or "", counterfactual_target_model)
        if not token_method:
            token_method = f"counterfactual_prompt:{tok_method}"
    if not token_method:
        token_method = "counterfactual_prompt:unknown"
    cf_total_tokens = cf_prompt_tokens + max(0, cf_completion_tokens)

    cost_actual_usd = _to_float(metrics.get("cost_actual_usd"), 0.0)
    if cost_actual_usd <= 0 and "external" in (route_mode or ""):
        cost_actual_usd = round((prompt_tokens_actual / 1_000_000.0) * in_price + (completion_tokens_actual / 1_000_000.0) * out_price, 8)

    saved_prompt_only = round((cf_prompt_tokens / 1_000_000.0) * in_price, 8) if route_mode == "local_only" else 0.0
    saved_full_modeled = round(((cf_prompt_tokens / 1_000_000.0) * in_price) + ((cf_completion_tokens / 1_000_000.0) * out_price), 8) if route_mode == "local_only" else 0.0

    intercept_class = _request_class(goal, intent)
    cleaned_goal_hash = _stable_hash(goal)

    conn.execute(
        """
        INSERT INTO request_usage_ledger(
          request_id, timestamp, session_key, run_id, cron_run_id,
          route_mode, intercept_class, provider, actual_model_name,
          counterfactual_target_model, pricing_snapshot_id,
          prompt_tokens_actual, completion_tokens_actual, total_tokens_actual,
          cost_actual_usd, counterfactual_prompt_tokens, counterfactual_completion_tokens,
          counterfactual_total_tokens, saved_cost_prompt_only_usd, saved_cost_full_modeled_usd,
          estimation_method, fallback_used, reason_code, reason_detail,
          cleaned_goal_hash, created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            request_id,
            now,
            session_key,
            run_id,
            cron_run_id,
            route_mode,
            intercept_class,
            provider,
            actual_model_name,
            counterfactual_target_model,
            pricing_snapshot_id,
            prompt_tokens_actual,
            completion_tokens_actual,
            total_tokens_actual,
            cost_actual_usd,
            cf_prompt_tokens,
            cf_completion_tokens,
            cf_total_tokens,
            saved_prompt_only,
            saved_full_modeled,
            token_method,
            int(bool(fallback_used)),
            reason_code,
            reason_detail,
            cleaned_goal_hash,
            now,
        ),
    )
    return {
        "request_id": request_id,
        "pricing_snapshot_id": pricing_snapshot_id,
        "saved_cost_prompt_only_usd": saved_prompt_only,
        "saved_cost_full_modeled_usd": saved_full_modeled,
        "cost_actual_usd": cost_actual_usd,
    }


def persist_cron_run_summary_conn(
    conn,
    *,
    cron_job_name: str,
    window_seconds: int = 3 * 3600,
    started_at: float | None = None,
    finished_at: float | None = None,
    status: str = "ok",
    cron_run_id: str | None = None,
    calc_version: str = CALC_VERSION,
    notes: str | None = None,
) -> dict:
    finished_at = finished_at or time.time()
    started_at = started_at or (finished_at - window_seconds)
    cron_run_id = cron_run_id or f"{cron_job_name}:{int(finished_at)}:{uuid.uuid4().hex[:8]}"

    rows = conn.execute(
        """
        SELECT * FROM request_usage_ledger
        WHERE created_at >= ? AND created_at <= ?
        ORDER BY created_at ASC
        """,
        (started_at, finished_at),
    ).fetchall()

    local_only_count = sum(1 for r in rows if (r[6] or "") == "local_only")
    external_count = sum(1 for r in rows if "external" in str(r[6] or ""))
    fallback_count = sum(int(r[22] or 0) for r in rows)
    ext_prompt = sum(_to_int(r[12]) for r in rows if "external" in str(r[6] or ""))
    ext_completion = sum(_to_int(r[13]) for r in rows if "external" in str(r[6] or ""))
    ext_cost = round(sum(_to_float(r[15]) for r in rows if "external" in str(r[6] or "")), 8)

    cf_prompt_saved = sum(_to_int(r[16]) for r in rows if (r[6] or "") == "local_only")
    cf_completion_saved = sum(_to_int(r[17]) for r in rows if (r[6] or "") == "local_only")
    saved_prompt_only = round(sum(_to_float(r[19]) for r in rows if (r[6] or "") == "local_only"), 8)
    saved_full = round(sum(_to_float(r[20]) for r in rows if (r[6] or "") == "local_only"), 8)
    net_conservative = round(saved_prompt_only - ext_cost, 8)
    net_modeled = round(saved_full - ext_cost, 8)
    snapshots = sorted({str(r[11]) for r in rows if r[11]})

    conn.execute(
        """
        INSERT INTO cron_run_usage_ledger(
          cron_run_id, cron_job_name, started_at, finished_at, status,
          local_only_count, external_count, fallback_count,
          external_prompt_tokens_actual, external_completion_tokens_actual, external_cost_actual_usd,
          counterfactual_prompt_tokens_saved, counterfactual_completion_tokens_saved,
          saved_cost_prompt_only_usd, saved_cost_full_modeled_usd,
          net_savings_conservative_usd, net_savings_modeled_usd,
          pricing_snapshot_ids, calc_version, notes, created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            cron_run_id,
            cron_job_name,
            started_at,
            finished_at,
            status,
            local_only_count,
            external_count,
            fallback_count,
            ext_prompt,
            ext_completion,
            ext_cost,
            cf_prompt_saved,
            cf_completion_saved,
            saved_prompt_only,
            saved_full,
            net_conservative,
            net_modeled,
            json.dumps(snapshots, ensure_ascii=False),
            calc_version,
            notes,
            time.time(),
        ),
    )

    conn.execute(
        "UPDATE request_usage_ledger SET cron_run_id = ? WHERE created_at >= ? AND created_at <= ? AND (cron_run_id IS NULL OR cron_run_id='')",
        (cron_run_id, started_at, finished_at),
    )

    return {
        "cron_run_id": cron_run_id,
        "cron_job_name": cron_job_name,
        "rows_aggregated": len(rows),
        "local_only_count": local_only_count,
        "external_count": external_count,
        "external_cost_actual_usd": ext_cost,
        "saved_cost_prompt_only_usd": saved_prompt_only,
        "saved_cost_full_modeled_usd": saved_full,
        "net_savings_conservative_usd": net_conservative,
        "net_savings_modeled_usd": net_modeled,
        "pricing_snapshot_ids": snapshots,
    }


def usage_ledger_report_conn(conn, limit: int = 50) -> dict:
    rows = conn.execute("SELECT * FROM request_usage_ledger ORDER BY id DESC LIMIT ?", (max(1, limit),)).fetchall()
    items = []
    for r in rows:
        items.append({
            "request_id": r["request_id"],
            "route_mode": r["route_mode"],
            "intercept_class": r["intercept_class"],
            "provider": r["provider"],
            "actual_model_name": r["actual_model_name"],
            "pricing_snapshot_id": r["pricing_snapshot_id"],
            "prompt_tokens_actual": r["prompt_tokens_actual"],
            "completion_tokens_actual": r["completion_tokens_actual"],
            "cost_actual_usd": r["cost_actual_usd"],
            "counterfactual_prompt_tokens": r["counterfactual_prompt_tokens"],
            "saved_cost_prompt_only_usd": r["saved_cost_prompt_only_usd"],
            "saved_cost_full_modeled_usd": r["saved_cost_full_modeled_usd"],
            "estimation_method": r["estimation_method"],
            "fallback_used": bool(r["fallback_used"]),
            "created_at": r["created_at"],
        })
    return {"rows": len(items), "items": items}


def cron_savings_report_conn(conn, limit: int = 20) -> dict:
    rows = conn.execute("SELECT * FROM cron_run_usage_ledger ORDER BY id DESC LIMIT ?", (max(1, limit),)).fetchall()
    items = []
    for r in rows:
        items.append({
            "cron_run_id": r["cron_run_id"],
            "cron_job_name": r["cron_job_name"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "status": r["status"],
            "local_only_count": r["local_only_count"],
            "external_count": r["external_count"],
            "external_cost_actual_usd": r["external_cost_actual_usd"],
            "saved_cost_prompt_only_usd": r["saved_cost_prompt_only_usd"],
            "saved_cost_full_modeled_usd": r["saved_cost_full_modeled_usd"],
            "net_savings_conservative_usd": r["net_savings_conservative_usd"],
            "net_savings_modeled_usd": r["net_savings_modeled_usd"],
            "pricing_snapshot_ids": json.loads(r["pricing_snapshot_ids"] or "[]"),
            "calc_version": r["calc_version"],
            "created_at": r["created_at"],
        })
    return {"rows": len(items), "items": items}
