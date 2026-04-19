from __future__ import annotations

import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import AppConfig
from .embeddings import EmbeddingProvider
from .executor import AutoExecutor
from .gateway import CodeContextGateway
from .telemetry import product_metrics_report

SERVICE_NAME = "codecontext-api"
SERVICE_VERSION = "phase4-minimal-v2"

DEFAULT_ROUTE_POLICY = {
    "modes": ["pass_through_direct", "local_only", "external_reasoning_with_compaction"],
    "notes": "Uses CodeContext router/executor flow",
}

DEFAULT_TIMEOUTS = {
    "auto_start_timeout_ms": 90000,
    "auto_continue_timeout_ms": 60000,
}

DEFAULT_COST_ROUTING = {
    "model": "openai/gpt-5.1-codex",
    "pricing": {
        "openai/gpt-5.1-codex": {"input_per_1k": 0.003, "output_per_1k": 0.012}
    },
    "max_cost_per_request": 0.015,
    "max_tokens_per_request": 2200,
    "avg_output_tokens": 420,
    "force_external_for_explain_style": True,
    "force_local_for_high_cost": True,
}


def _configure_utf8_stdio() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
        # Client disconnected before response could be sent (e.g. timeout).
        # This is expected when requests take longer than the client's timeout.
        pass


def _safe_bool(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def _clean_prompt(raw: str) -> tuple[str, list[str]]:
    text = str(raw or "").replace("\r\n", "\n").strip()
    rules = []

    before = text
    text = re.sub(r"Sender \(untrusted metadata\):\s*```json[\s\S]*?```", "", text).strip()
    if text != before:
        rules.append("stripped_sender_metadata_block")

    m = re.search(r"\[[^\]]+\]\s*([\s\S]*)$", text)
    if m and m.group(1).strip():
        extracted = m.group(1).strip()
        if extracted != text:
            text = extracted
            rules.append("extracted_last_timestamped_message_body")

    lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("```")]
    if len(lines) > 1:
        metadata_re = re.compile(r"^\[.*\]$|^System:|^Assistant:|^Human:")
        user_lines = [ln for ln in lines if not metadata_re.match(ln)]
        if user_lines:
            text = "\n".join(user_lines)
            if len(user_lines) < len(lines):
                rules.append("stripped_metadata_lines_preserved_user_content")

    return text.strip(), rules


def _safe_error(code: str, message: str, status: int = 400):
    return status, {"ok": False, "error": {"code": code, "message": message}, "timestamp": _now_iso()}


def _route_for_api(bridge: CodeContextGateway, executor: AutoExecutor, prompt: str, dry_run: bool, top_k: int, token_budget: int):
    cleaned, rules = _clean_prompt(prompt)
    if not cleaned:
        return _safe_error("validation_error", "prompt is required and cannot be empty", 422)

    route = bridge.route_request(cleaned, top_k=top_k, token_budget=token_budget)
    route_obj = route.get("route") or {}
    result_obj = route.get("result") or {}
    # Read mode from top-level first (where the bridge puts it),
    # then fall back to nested locations, then default to local_only.
    mode = route.get("mode") or result_obj.get("mode") or route_obj.get("mode") or "local_only"
    chosen_route = "external_reasoning_with_compaction" if mode == "external_reasoning" else (mode or "local_only")

    cost_estimate = (route.get("metrics") or {}).get("estimated_request_cost_usd", 0.0)
    response = {
        "ok": True,
        "trace_id": f"api_{int(time.time()*1000)}",
        "cleaned_prompt": cleaned,
        "cleanup_rules": rules,
        "intercept_class": route_obj.get("intent") or "unknown",
        "candidate_relevant": True,
        "chosen_route": chosen_route,
        "class_reason": "router_intent",
        "relevance_reason": "route_request_invoked",
        "route_reason": (route.get("debug") or {}).get("route_action") or "router_decision",
        "cost_estimate": cost_estimate,
        "cost_threshold_hit": "high_cost" in str((route.get("metrics") or {}).get("cost_reason", "")),
        "cost_decision_reason": (route.get("metrics") or {}).get("cost_reason") or "router_cost_policy",
        "estimated_savings_vs_external": (route.get("metrics") or {}).get("estimated_savings_vs_external", 0.0),
        "dry_run": dry_run,
        "timestamp": _now_iso(),
    }

    if dry_run:
        response["result"] = {"mode": chosen_route, "route": route_obj}
        return 200, response

    started = executor.start(cleaned, top_k=top_k, token_budget=token_budget)
    response["run_id"] = started.get("run_id")
    response["result"] = started.get("result")
    response["metrics"] = started.get("metrics")
    # Phase 2: surface the LLM call outcome (if any) and the real $ cost
    # that landed in the usage ledger, so API consumers can show real spend
    # instead of the counterfactual estimate.
    llm_response = started.get("llm_response")
    if llm_response is not None:
        response["llm_response"] = llm_response
    usage = started.get("usage_ledger") or {}
    # Only surface cost_actual_usd when there was a real external spend.
    # For local_only runs the ledger records $0 by design, and exposing a
    # "cost" of zero to API consumers is more confusing than useful — they
    # should look at estimated_savings_vs_external instead.
    cost_actual = usage.get("cost_actual_usd")
    if cost_actual is not None and cost_actual > 0:
        response["cost_actual_usd"] = cost_actual
    return 200, response


def make_handler(config: AppConfig):
    # Pre-warm the embedding model ONCE at server startup.
    # This shared instance is injected into the bridge's search engine
    # so multiple components reuse the same loaded model.
    shared_embeddings = EmbeddingProvider(enabled=config.enable_embeddings)
    if shared_embeddings.is_available():
        shared_embeddings._ensure_model()
        print(json.dumps({"event": "embeddings_loaded", "model": shared_embeddings.model_name, "dim": shared_embeddings.dim, "timestamp": _now_iso()}, ensure_ascii=False))
    else:
        print(json.dumps({"event": "embeddings_unavailable", "fallback": "bm25_only", "timestamp": _now_iso()}, ensure_ascii=False))

    bridge = CodeContextGateway(config)
    executor = AutoExecutor(bridge)

    # Inject shared embedding provider into all components that use one,
    # so the model is loaded exactly once for the server's lifetime.
    if hasattr(bridge, 'search') and hasattr(bridge.search, 'embeddings'):
        bridge.search.embeddings = shared_embeddings
    if hasattr(bridge, 'context') and hasattr(bridge.context, 'search') and hasattr(bridge.context.search, 'embeddings'):
        bridge.context.search.embeddings = shared_embeddings
    if hasattr(bridge, 'summaries') and hasattr(bridge.summaries, 'embeddings'):
        bridge.summaries.embeddings = shared_embeddings

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # keep server quiet for local demo
            return

        def _query(self):
            return parse_qs(urlparse(self.path).query)

        def do_GET(self):
            parsed = urlparse(self.path)
            q = self._query()

            if parsed.path == "/health":
                db_ok = True
                db_error = None
                try:
                    from .db import connect
                    conn = connect(config.db_path)
                    conn.close()
                except Exception as e:
                    db_ok = False
                    db_error = str(e)
                payload = {
                    "ok": db_ok,
                    "service": SERVICE_NAME,
                    "version": SERVICE_VERSION,
                    "db_status": "ok" if db_ok else "error",
                    "backend_readiness": db_ok,
                    "embeddings_available": shared_embeddings.is_available(),
                    "timestamp": _now_iso(),
                }
                if db_error:
                    payload["db_error"] = db_error
                return _json_response(self, 200 if db_ok else 503, payload)

            if parsed.path == "/metrics":
                window = (q.get("window") or [None])[0]
                route_filter = (q.get("route_filter") or [None])[0]
                top_n = int((q.get("top_n") or ["5"])[0])
                definitions = _safe_bool((q.get("definitions") or ["false"])[0])
                report = product_metrics_report(config.db_path, window=window, route_filter=route_filter, top_n=top_n, include_definitions=definitions)
                return _json_response(self, 200, report)

            if parsed.path == "/sales-summary":
                window = (q.get("window") or [None])[0]
                route_filter = (q.get("route_filter") or [None])[0]
                top_n = int((q.get("top_n") or ["5"])[0])
                report = product_metrics_report(config.db_path, window=window, route_filter=route_filter, top_n=top_n, include_definitions=False)
                payload = {
                    "ok": True,
                    "sales_demo_summary": report.get("sales_demo_summary"),
                    "summary": report.get("metrics_report"),
                    "timestamp": _now_iso(),
                }
                return _json_response(self, 200, payload)

            if parsed.path == "/config":
                payload = {
                    "ok": True,
                    "service": SERVICE_NAME,
                    "version": SERVICE_VERSION,
                    "route_policy": DEFAULT_ROUTE_POLICY,
                    "timeouts": DEFAULT_TIMEOUTS,
                    "cost_routing": {
                        "model": DEFAULT_COST_ROUTING["model"],
                        "pricing_model_names": list(DEFAULT_COST_ROUTING["pricing"].keys()),
                        "max_cost_per_request": DEFAULT_COST_ROUTING["max_cost_per_request"],
                        "max_tokens_per_request": DEFAULT_COST_ROUTING["max_tokens_per_request"],
                        "avg_output_tokens": DEFAULT_COST_ROUTING["avg_output_tokens"],
                        "force_external_for_explain_style": DEFAULT_COST_ROUTING["force_external_for_explain_style"],
                        "force_local_for_high_cost": DEFAULT_COST_ROUTING["force_local_for_high_cost"],
                    },
                    "notes": {
                        "secrets": "No secrets are exposed in this view",
                        "db_path": str(config.db_path),
                    },
                    "timestamp": _now_iso(),
                }
                return _json_response(self, 200, payload)

            status, payload = _safe_error("not_found", "endpoint not found", 404)
            return _json_response(self, status, payload)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/route-or-run":
                status, payload = _safe_error("not_found", "endpoint not found", 404)
                return _json_response(self, status, payload)

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(max(0, length)) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                status, payload = _safe_error("validation_error", "malformed JSON body", 422)
                return _json_response(self, status, payload)

            prompt = str(data.get("prompt") or "").strip()
            options = data.get("options") or {}
            dry_run = bool(options.get("dry_run", False))
            try:
                top_k = int(options.get("top_k", 4))
                token_budget = int(options.get("token_budget", 1200))
            except Exception:
                status, payload = _safe_error("validation_error", "options.top_k and options.token_budget must be integers", 422)
                return _json_response(self, status, payload)

            if not prompt:
                status, payload = _safe_error("validation_error", "prompt is required", 422)
                return _json_response(self, status, payload)

            try:
                status, payload = _route_for_api(bridge, executor, prompt, dry_run=dry_run, top_k=top_k, token_budget=token_budget)
                return _json_response(self, status, payload)
            except Exception as exc:
                # Log the actual error to stderr so we can debug server-side crashes
                print(f"[route-or-run error] {exc.__class__.__name__}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                status, payload = _safe_error("internal_error", f"route-or-run failed: {exc.__class__.__name__}", 500)
                return _json_response(self, status, payload)

    return Handler


def run_product_api(root: str = ".", host: str = "127.0.0.1", port: int = 8787):
    _configure_utf8_stdio()
    cfg = AppConfig(root=__import__("pathlib").Path(root))
    server = ThreadingHTTPServer((host, int(port)), make_handler(cfg))
    print(json.dumps({"ok": True, "service": SERVICE_NAME, "version": SERVICE_VERSION, "host": host, "port": int(port), "timestamp": _now_iso()}, ensure_ascii=False))
    server.serve_forever()
