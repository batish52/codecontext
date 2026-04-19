from __future__ import annotations

import json
import re
import traceback
from typing import Any

VISIBILITY_MODES = {"normal", "debug_summary", "raw_debug"}

INTERNAL_KEYS = {
    "boundary_events", "execution_runs", "request_metrics", "metric_definitions",
    "sample_candidates", "reason_detail", "coverage_reconciliation", "debug",
}

SENSITIVE_DETAIL_KEYS = {
    "stdout", "stderr", "traceback", "raw_output", "reason_detail",
    "exception", "stack", "stacktrace", "error_detail",
}

TRACE_PATTERNS = {
    "exec_trace": [
        re.compile(r"\bexec\b.*\b(completed|failed|started)\b", re.IGNORECASE),
        re.compile(r"\bprocess exited with code\b", re.IGNORECASE),
    ],
    "file_read_trace": [
        re.compile(r"\bread\b.*\bwith lines\b", re.IGNORECASE),
        re.compile(r"\b(offset|line|lines)\b\s*[:=]", re.IGNORECASE),
    ],
    "search_trace": [
        re.compile(r"\brg\b\s", re.IGNORECASE),
        re.compile(r"\bgrep\b\s", re.IGNORECASE),
        re.compile(r"\bsearch hits\b", re.IGNORECASE),
    ],
    "code_navigation_trace": [
        re.compile(r"\bservices/[^\s:]+\.py:\d+", re.IGNORECASE),
        re.compile(r"\b[a-zA-Z]:\\[^\s]+", re.IGNORECASE),
        re.compile(r"\b(file paths?|line numbers?|source references?)\b", re.IGNORECASE),
    ],
}


def _truncate_text(text: str, limit: int = 1600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text)-limit} chars]"


def _classify_trace_text(text: str) -> str | None:
    for label, pats in TRACE_PATTERNS.items():
        if any(p.search(text) for p in pats):
            return label
    return None


def _extract_tool_trace_summary(text: str) -> dict:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    tool_counts = {"exec": 0, "read": 0, "search": 0, "other": 0}
    findings = 0
    for ln in lines:
        low = ln.lower()
        if "exec" in low:
            tool_counts["exec"] += 1
        elif low.startswith("read") or " read " in low:
            tool_counts["read"] += 1
        elif "rg " in low or "grep" in low or "search" in low:
            tool_counts["search"] += 1
        else:
            tool_counts["other"] += 1
        if re.search(r":\d+", ln) or " hit" in low or "match" in low:
            findings += 1
    return {
        "line_count": len(lines),
        "tool_counts": tool_counts,
        "findings": findings,
    }


def classify_tool_output(payload: Any) -> str:
    if isinstance(payload, str):
        low = payload.lower()
        if any(k in low for k in ("boundary_events", "execution_runs", "request_metrics", "sqlite", "pragma table_info")):
            return "telemetry"
        if "traceback" in low:
            return "traceback"
        trace_kind = _classify_trace_text(payload)
        if trace_kind:
            return trace_kind
        if payload.strip().startswith("{") or payload.strip().startswith("["):
            return "raw_json"
        if any(x in low for x in ("error", "failed", "exception")):
            return "logs"
        return "user_facing_result"

    if isinstance(payload, dict):
        keys = set(payload.keys())
        if "runtime_operational_debug_audit" in keys or "measurement_source" in keys:
            return "internal_audit"
        if "metrics_report" in keys:
            return "metrics_report"
        if keys & {"error", "traceback", "stderr"}:
            return "traceback"
        if keys & {"stdout", "stderr", "status", "durationMs", "opName", "lifecycleStage"}:
            return "tool_trace"
        if keys & INTERNAL_KEYS:
            return "telemetry"
        return "structured_summary"

    if isinstance(payload, list):
        return "raw_json"
    return "unknown"


def _normal_summary(kind: str, payload: Any) -> str:
    if kind in {"tool_trace", "exec_trace", "file_read_trace", "search_trace", "code_navigation_trace"}:
        return "Internal tool trace captured and hidden in normal mode."

    if kind in {"internal_audit", "telemetry", "metrics_report", "raw_json", "structured_summary", "unknown"}:
        if isinstance(payload, dict):
            if "runtime_operational_debug_audit" in payload:
                a = payload.get("runtime_operational_debug_audit", {})
                return (
                    f"Audit complete: {a.get('total_prompts', 0)} prompts reviewed; "
                    f"mostly local-handled ({a.get('likely_local_only', 0)} local vs {a.get('likely_external', 0)} external-likely)."
                )
            if "totals" in payload and isinstance(payload.get("totals"), dict):
                t = payload["totals"]
                return (
                    f"Benchmark complete: {t.get('requests', 0)} requests, "
                    f"local ratio {round(float(t.get('local_only_ratio', 0))*100,1)}%, "
                    f"external ratio {round(float(t.get('external_reasoning_ratio', 0))*100,1)}%."
                )
            if "metrics_report" in payload:
                m = payload.get("metrics_report", {})
                return (
                    f"Metrics summary: local-only {m.get('local_only_count', 0)}, "
                    f"external {m.get('external_reasoning_with_compaction_count', 0)}, "
                    f"estimated savings ${m.get('estimated_savings_total', 0)}."
                )
        if isinstance(payload, dict):
            if "requests" in payload:
                return f"Operation complete: requests={payload.get('requests')}"
            if "rows" in payload:
                return f"Operation complete: rows={payload.get('rows')}"
            keys = [k for k in payload.keys() if k not in INTERNAL_KEYS][:4]
            if keys:
                return f"Operation complete. Key fields: {', '.join(keys)}."
        return "Internal tool output processed successfully."

    if kind == "traceback":
        return "The operation failed internally; a clean fallback or retry is needed."

    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return _truncate_text(text, 500)


def _debug_summary(kind: str, payload: Any) -> dict:
    out = {"tool_output_class": kind}
    if kind in {"tool_trace", "exec_trace", "file_read_trace", "search_trace", "code_navigation_trace"}:
        trace_text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        out["trace_summary"] = _extract_tool_trace_summary(trace_text)
        if isinstance(payload, dict):
            out["top_level_keys"] = list(payload.keys())[:20]
        return out

    if isinstance(payload, dict):
        out["top_level_keys"] = list(payload.keys())[:20]
        if "runtime_operational_debug_audit" in payload:
            a = payload.get("runtime_operational_debug_audit", {})
            out["stats"] = {
                "total_prompts": a.get("total_prompts"),
                "likely_local_only": a.get("likely_local_only"),
                "likely_external": a.get("likely_external"),
                "families": a.get("families"),
            }
        elif "totals" in payload and isinstance(payload.get("totals"), dict):
            t = payload["totals"]
            out["stats"] = {
                "requests": t.get("requests"),
                "local_only_ratio": t.get("local_only_ratio"),
                "external_reasoning_ratio": t.get("external_reasoning_ratio"),
                "estimated_cost_reduction_usd": t.get("estimated_cost_reduction_usd") or t.get("estimated_cost_reduction"),
            }
    elif isinstance(payload, str):
        out["preview"] = _truncate_text(payload, 600)
    return out


def _sanitize_string(text: str, mode: str) -> Any:
    if mode == "raw_debug":
        return text
    trace_kind = _classify_trace_text(text) or ("traceback" if "traceback" in text.lower() else None)
    if trace_kind:
        if mode == "debug_summary":
            return {
                "hidden": True,
                "tool_output_class": trace_kind,
                "trace_summary": _extract_tool_trace_summary(text),
            }
        return "Internal diagnostic details hidden in normal mode."
    return text


def sanitize_payload_for_visibility(payload: Any, mode: str = "normal") -> Any:
    mode = mode if mode in VISIBILITY_MODES else "normal"
    if mode == "raw_debug":
        return payload
    if isinstance(payload, str):
        return _sanitize_string(payload, mode)
    if isinstance(payload, list):
        return [sanitize_payload_for_visibility(item, mode) for item in payload[:200]]
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if k in INTERNAL_KEYS:
                if mode == "debug_summary":
                    out[k] = "[internal omitted]"
                continue
            if k in SENSITIVE_DETAIL_KEYS:
                if isinstance(v, (int, float, bool)) or v is None:
                    out[k] = v
                elif mode == "debug_summary":
                    raw_text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    out[k] = {
                        "hidden": True,
                        "trace_summary": _extract_tool_trace_summary(raw_text),
                    }
                else:
                    out[k] = "[hidden]"
                continue
            out[k] = sanitize_payload_for_visibility(v, mode)
        return out
    return payload


def format_tool_output(payload: Any, visibility_mode: str = "normal") -> dict:
    mode = visibility_mode if visibility_mode in VISIBILITY_MODES else "normal"
    safe_payload = sanitize_payload_for_visibility(payload, mode)
    kind = classify_tool_output(safe_payload)
    user_text = _normal_summary(kind, safe_payload)
    envelope = {
        "user_text": user_text,
        "visibility_mode": mode,
        "tool_output_class": kind,
        "raw_available": kind != "user_facing_result",
    }
    if mode in {"debug_summary", "raw_debug"}:
        envelope["debug_summary"] = _debug_summary(kind, safe_payload)
    if mode == "raw_debug":
        raw = payload if isinstance(payload, (dict, list)) else str(payload)
        raw_s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        envelope["raw_output"] = _truncate_text(raw_s, 12000)
    return envelope


def safe_format_exception(exc: Exception, visibility_mode: str = "normal") -> dict:
    payload = {
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    return format_tool_output(payload, visibility_mode=visibility_mode)
