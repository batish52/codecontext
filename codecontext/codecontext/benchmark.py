from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Any

from .config import AppConfig
from .gateway import CodeContextGateway

DATASET_DIR = Path(__file__).parent / "benchmarks" / "datasets"
RESULTS_DIR = Path(__file__).parent / "benchmarks" / "results"


def _now_ts() -> float:
    return time.time()


def _load_dataset(name: str) -> list[dict]:
    path = DATASET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset must be a list")
    out = []
    for i, row in enumerate(data):
        if isinstance(row, str):
            out.append({"prompt": row, "type": "general", "id": i + 1})
        elif isinstance(row, dict) and row.get("prompt"):
            out.append({"id": row.get("id", i + 1), "prompt": str(row["prompt"]), "type": str(row.get("type", "general"))})
    return out


def _is_irrelevant_control(item: dict) -> bool:
    if item.get("type") == "irrelevant":
        return True
    p = str(item.get("prompt", "")).lower()
    return any(x in p for x in ["weather", "joke", "good morning", "hello there", "what time is it", "thanks"]) and "codecontext" not in p


def _default_evaluator(config: AppConfig) -> Callable[[str], dict]:
    bridge = CodeContextGateway(config)

    def _eval(prompt: str) -> dict:
        return bridge.route_request(prompt, top_k=4, token_budget=1200)

    return _eval


def _config_snapshot(config: AppConfig) -> dict:
    return {
        "max_escalation_cost_per_request": config.max_escalation_cost_per_request,
        "allow_external_for_explain_style": config.allow_external_for_explain_style,
        "allow_external_for_product_workflow": config.allow_external_for_product_workflow,
        "prefer_local_for_plugin_runtime": config.prefer_local_for_plugin_runtime,
        "prefer_local_for_phase_workflow": config.prefer_local_for_phase_workflow,
        "index_schema_version": config.index_schema_version,
    }


def run_benchmark(root: Path, dataset: str, runs: int = 1, evaluator: Callable[[str], dict] | None = None) -> dict:
    config = AppConfig(root)
    prompts = _load_dataset(dataset)
    eval_fn = evaluator or _default_evaluator(config)

    records: list[dict] = []
    for run_idx in range(max(1, runs)):
        for item in prompts:
            prompt = item["prompt"]
            started = _now_ts()
            if _is_irrelevant_control(item):
                route = "pass_through_direct"
                rec = {
                    "dataset": dataset,
                    "run_iteration": run_idx + 1,
                    "prompt_id": item["id"],
                    "prompt": prompt,
                    "prompt_type": item.get("type", "general"),
                    "route": route,
                    "escalated": False,
                    "escalation_allowed": False,
                    "escalation_reason": "irrelevant_control",
                    "escalation_blocked_by_cost_policy": False,
                    "estimated_extra_cost_of_escalation": 0.0,
                    "local_retry_attempted": False,
                    "estimated_tokens": max(1, len(prompt) // 4),
                    "estimated_cost": 0.0,
                    "estimated_cost_if_external": 0.0,
                    "estimated_savings_vs_external": 0.0,
                    "latency_ms_end_to_end": round((_now_ts() - started) * 1000, 2),
                    "latency_ms_backend": None,
                    "failure_type": None,
                    "ok": True,
                    "timestamp": _now_ts(),
                }
                records.append(rec)
                continue

            try:
                result = eval_fn(prompt)
                # Previously this was `result.get("result", {})` — but
                # gateway.route_request returns the route dict directly
                # (`{"mode": ..., "route": ..., "debug": ..., "metrics": ...}`),
                # with no "result" wrapper. Reading the wrapper meant `rr`
                # was always empty and `mode` always defaulted to
                # "local_only" — every benchmark record said local_only/$0
                # regardless of what actually happened. Unwrap only if a
                # "result" key is genuinely present (defensive).
                rr = result if isinstance(result, dict) else {}
                if isinstance(rr.get("result"), dict):
                    rr = rr["result"]
                metrics = (result or {}).get("metrics", {}) or {}
                debug = rr.get("debug", {}) if isinstance(rr, dict) else {}
                mode = rr.get("mode", "local_only")
                route = "external_reasoning_with_compaction" if mode == "external_reasoning" else mode
                est_cost = float(metrics.get("estimated_cost", metrics.get("estimated_request_cost_usd", 0.0)) or 0.0)
                est_tokens = int(metrics.get("estimated_tokens", metrics.get("prompt_tokens_est", max(1, len(prompt) // 4))) or max(1, len(prompt) // 4))
                est_savings = float(metrics.get("estimated_savings_vs_external", 0.0) or 0.0)
                est_external = est_cost + est_savings
                rec = {
                    "dataset": dataset,
                    "run_iteration": run_idx + 1,
                    "prompt_id": item["id"],
                    "prompt": prompt,
                    "prompt_type": item.get("type", "general"),
                    "route": route,
                    "escalated": route == "external_reasoning_with_compaction",
                    "escalation_allowed": bool(debug.get("escalation_allowed", route == "external_reasoning_with_compaction")),
                    "escalation_reason": debug.get("escalation_reason"),
                    "escalation_blocked_by_cost_policy": bool(debug.get("escalation_blocked_by_cost_policy", False)),
                    "estimated_extra_cost_of_escalation": float(debug.get("estimated_extra_cost_of_escalation", 0.0) or 0.0),
                    "local_retry_attempted": bool(debug.get("local_retry_attempted", False)),
                    "estimated_tokens": est_tokens,
                    "estimated_cost": est_cost,
                    "estimated_cost_if_external": float(est_external),
                    "estimated_savings_vs_external": float(est_savings),
                    "latency_ms_end_to_end": round((_now_ts() - started) * 1000, 2),
                    "latency_ms_backend": round((_now_ts() - started) * 1000, 2),
                    "failure_type": None,
                    "ok": True,
                    "timestamp": _now_ts(),
                }
                records.append(rec)
            except Exception as e:
                # Exception records used to stamp route=external_reasoning_with_compaction
                # with escalated=True and cost=$0 — which looks like a
                # free successful external call in downstream savings
                # aggregation. A failure is not an escalation and is
                # not free. Record it as its own `route="error"` /
                # `escalated=False`.
                records.append(
                    {
                        "dataset": dataset,
                        "run_iteration": run_idx + 1,
                        "prompt_id": item["id"],
                        "prompt": prompt,
                        "prompt_type": item.get("type", "general"),
                        "route": "error",
                        "escalated": False,
                        "escalation_allowed": False,
                        "escalation_reason": "exception",
                        "escalation_blocked_by_cost_policy": False,
                        "estimated_extra_cost_of_escalation": 0.0,
                        "local_retry_attempted": False,
                        "estimated_tokens": 0,
                        "estimated_cost": 0.0,
                        "estimated_cost_if_external": 0.0,
                        "estimated_savings_vs_external": 0.0,
                        "latency_ms_end_to_end": round((_now_ts() - started) * 1000, 2),
                        "latency_ms_backend": None,
                        "failure_type": type(e).__name__,
                        "ok": False,
                        "timestamp": _now_ts(),
                    }
                )

    payload = {
        "dataset": dataset,
        "runs": max(1, runs),
        "timestamp": _now_ts(),
        "config_snapshot": _config_snapshot(config),
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{dataset}_{int(payload['timestamp'])}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["result_path"] = str(out)
    return payload


def _latest_result(dataset: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(RESULTS_DIR.glob(f"{dataset}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no benchmark results for dataset={dataset}")
    return candidates[0]


def _load_result(dataset: str) -> dict:
    return json.loads(_latest_result(dataset).read_text(encoding="utf-8"))


def _aggregate(records: list[dict]) -> dict:
    total = len(records)
    by_route = {"pass_through_direct": 0, "local_only": 0, "external_reasoning_with_compaction": 0}
    failures: dict[str, int] = {}
    prompt_type_counts: dict[str, int] = {}
    expensive_prompts = sorted(records, key=lambda r: float(r.get("estimated_cost", 0.0)), reverse=True)

    tokens = 0
    cost = 0.0
    baseline_external = 0.0
    savings = 0.0
    escalations = 0
    relevant = 0
    latencies: dict[str, list[float]] = {"pass_through_direct": [], "local_only": [], "external_reasoning_with_compaction": []}

    for r in records:
        route = str(r.get("route") or "pass_through_direct")
        if route in by_route:
            by_route[route] += 1
        pt = str(r.get("prompt_type") or "general")
        prompt_type_counts[pt] = prompt_type_counts.get(pt, 0) + 1

        tokens += int(r.get("estimated_tokens", 0) or 0)
        c = float(r.get("estimated_cost", 0.0) or 0.0)
        cost += c
        baseline_external += float(r.get("estimated_cost_if_external", c) or c)
        savings += float(r.get("estimated_savings_vs_external", 0.0) or 0.0)

        if route != "pass_through_direct":
            relevant += 1
        if r.get("escalated"):
            escalations += 1
        if route in latencies and r.get("latency_ms_end_to_end") is not None:
            latencies[route].append(float(r["latency_ms_end_to_end"]))
        if not r.get("ok"):
            ft = str(r.get("failure_type") or "unknown_failure")
            failures[ft] = failures.get(ft, 0) + 1

    avg_latency = {k: round(sum(v) / len(v), 2) if v else None for k, v in latencies.items()}
    external_avoided_rate = round((by_route["local_only"] / max(1, relevant)) * 100.0, 2)
    savings_pct = round(((baseline_external - cost) / max(1e-9, baseline_external)) * 100.0, 2) if baseline_external > 0 else 0.0
    escalation_rate = round((escalations / max(1, total)) * 100.0, 2)

    return {
        "total_requests": total,
        "route_counts": by_route,
        "route_percentages": {k: round((v / max(1, total)) * 100.0, 2) for k, v in by_route.items()},
        "escalation_rate": escalation_rate,
        "estimated_total_cost": round(cost, 6),
        "estimated_all_external_baseline_cost": round(baseline_external, 6),
        "estimated_savings_absolute": round(baseline_external - cost, 6),
        "estimated_savings_percent": savings_pct,
        "estimated_outbound_tokens_total": tokens,
        "avg_latency_ms_by_route": avg_latency,
        "failure_counts": failures,
        "external_avoided_rate": external_avoided_rate,
        "external_avoided_count": by_route["local_only"],
        "top_expensive_prompts": [
            {"prompt": r.get("prompt"), "route": r.get("route"), "estimated_cost": round(float(r.get("estimated_cost", 0.0) or 0.0), 6), "prompt_type": r.get("prompt_type")}
            for r in expensive_prompts[:5]
        ],
        "top_prompt_types": sorted([{"prompt_type": k, "count": v} for k, v in prompt_type_counts.items()], key=lambda x: x["count"], reverse=True),
    }


def benchmark_report(dataset: str) -> dict:
    run = _load_result(dataset)
    agg = _aggregate(run.get("records", []))
    return {
        "dataset": run.get("dataset"),
        "timestamp": run.get("timestamp"),
        "runs": run.get("runs"),
        "config_snapshot": run.get("config_snapshot"),
        "summary": agg,
        "result_path": str(_latest_result(dataset)),
    }


def benchmark_sales_summary(dataset: str) -> dict:
    rep = benchmark_report(dataset)
    s = rep["summary"]
    txt = (
        f"In this dataset of {s['total_requests']} requests: "
        f"{s['route_percentages']['local_only']}% handled locally, "
        f"{s['route_percentages']['external_reasoning_with_compaction']}% escalated intelligently, "
        f"{s['route_percentages']['pass_through_direct']}% passed through, "
        f"Estimated cost reduced by {s['estimated_savings_percent']}%, "
        f"External calls reduced by {s['external_avoided_rate']}%, "
        f"Average latency by route = {s['avg_latency_ms_by_route']}"
    )
    return {"dataset": dataset, "sales_summary": txt, "summary": s}


def benchmark_leakage_audit(dataset: str) -> dict:
    run = _load_result(dataset)
    records = run.get("records", [])

    suspicious_escalations = [
        r for r in records
        if r.get("route") == "external_reasoning_with_compaction"
        and (r.get("escalation_blocked_by_cost_policy") or (r.get("estimated_extra_cost_of_escalation", 0.0) > run.get("config_snapshot", {}).get("max_escalation_cost_per_request", 9999)))
    ]

    unnecessary_local_escalation = [
        r for r in records
        if r.get("route") == "local_only" and r.get("escalation_allowed") is True and not r.get("local_retry_attempted", False)
    ]

    costs = [float(r.get("estimated_cost", 0.0) or 0.0) for r in records]
    mean_cost = statistics.mean(costs) if costs else 0.0
    stdev_cost = statistics.pstdev(costs) if len(costs) > 1 else 0.0
    outlier_cut = mean_cost + 2 * stdev_cost
    high_cost_outliers = [
        {"prompt": r.get("prompt"), "estimated_cost": r.get("estimated_cost"), "route": r.get("route")}
        for r in records
        if float(r.get("estimated_cost", 0.0) or 0.0) > outlier_cut and outlier_cut > 0
    ]

    cluster: dict[str, dict] = {}
    for r in records:
        key = str(r.get("prompt_type") or "general")
        c = cluster.setdefault(key, {"total": 0, "external": 0, "avg_savings": 0.0})
        c["total"] += 1
        if r.get("route") == "external_reasoning_with_compaction":
            c["external"] += 1
        c["avg_savings"] += float(r.get("estimated_savings_vs_external", 0.0) or 0.0)
    suspicious_clusters = []
    for k, v in cluster.items():
        ext_ratio = v["external"] / max(1, v["total"])
        avg_sv = v["avg_savings"] / max(1, v["total"])
        if ext_ratio > 0.7 and avg_sv < 0.001:
            suspicious_clusters.append({"prompt_type": k, "external_ratio": round(ext_ratio, 3), "avg_savings": round(avg_sv, 6)})

    return {
        "dataset": dataset,
        "total_records": len(records),
        "suspicious_escalations": suspicious_escalations[:20],
        "unnecessary_local_escalation_candidates": unnecessary_local_escalation[:20],
        "suspicious_escalation_clusters": suspicious_clusters,
        "high_cost_outliers": high_cost_outliers[:20],
        "result_path": str(_latest_result(dataset)),
    }
