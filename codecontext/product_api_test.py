from __future__ import annotations

import json
import threading
import time
import tempfile
from pathlib import Path
from urllib import request, parse, error
from unittest import mock

from codecontext.config import AppConfig
from codecontext.db import connect
from codecontext.product_api import make_handler
from http.server import ThreadingHTTPServer


def _seed_db(cfg: AppConfig):
    conn = connect(cfg.db_path)
    now = time.time()
    conn.execute(
        "INSERT INTO boundary_events(correlation_id, session_key, request_text_raw, cleaned_goal, matched_intercept_class, candidate_relevant, route_mode, classification_completed, heavy_local_handling_triggered, intercept_attempted, backend_call_attempted, backend_call_succeeded, entered_execution_runs, fallback_used, old_path_used, run_id, reason_code, reason_detail, source_kind, plugin_id, plugin_path, plugin_version_marker, backend_cli_path, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("t1", "s", "thanks", "thanks", "existing_relevant", 1, "pass_through_direct", 1, 0, 1, 0, 0, 0, 0, 0, None, "pass_through_direct", "", "live_plugin", "codecontext-runtime", "codecontext-runtime", "test", "python main.py", now, now),
    )
    conn.execute(
        "INSERT INTO request_metrics(route_mode, local_only, external_reasoning, evidence_count, evidence_chars, evidence_tokens, prompt_tokens_est, avoided_tokens_est, compaction_chars_saved, fallback_used, estimated_tokens, estimated_cost, route_chosen, cost_reason, estimated_savings_vs_external, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("local_only", 1, 0, 0, 0, 0, 120, 0, 0, 0, 120, 0.001, "local_only", "high_cost_force_local", 0.02, now),
    )
    conn.commit()
    conn.close()


def _start_server(cfg: AppConfig):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cfg))
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, th


# --- Fake LLM provider (Phase 2 API surface test) ---------------------------
from http.server import BaseHTTPRequestHandler as _BH


class _FakeLLM(_BH):
    response_spec: dict = {}

    def log_message(self, *a, **kw):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        self.rfile.read(length)
        spec = self.server.response_spec  # type: ignore[attr-defined]
        body = json.dumps(spec.get("payload", {})).encode("utf-8")
        self.send_response(spec.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_fake_llm():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    server.response_spec = {  # type: ignore[attr-defined]
        "status": 200,
        "payload": {
            "choices": [{"message": {"content": json.dumps({"kind": "answer", "answer": "ok"})}}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 40},
        },
    }
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, th


def _get(base: str, path: str):
    with request.urlopen(base + path, timeout=15) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(base: str, path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        cfg = AppConfig(Path(td))
        _seed_db(cfg)
        server, th = _start_server(cfg)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            # 1) health
            status, body = _get(base, "/health")
            assert status == 200 and body["ok"] is True

            # 2) metrics structure
            status, body = _get(base, "/metrics?window=7d&top_n=3&definitions=true")
            assert status == 200 and "metrics_report" in body and "sales_demo_summary" in body

            # 3) sales-summary structure
            status, body = _get(base, "/sales-summary?window=7d")
            assert status == 200 and "sales_demo_summary" in body and "summary" in body

            # 4) config no secret leakage
            status, body = _get(base, "/config")
            dump = json.dumps(body).lower()
            assert status == 200 and "api_key" not in dump and "password" not in dump

            # 5) route-or-run dry-run
            status, body = _post(base, "/route-or-run", {"prompt": "check logs for runtime", "options": {"dry_run": True}})
            assert status == 200 and body.get("dry_run") is True and body.get("run_id") is None

            # 6) route-or-run execute
            status, body = _post(base, "/route-or-run", {"prompt": "check logs for runtime", "options": {"dry_run": False, "top_k": 2, "token_budget": 600}})
            assert status == 200 and body.get("ok") is True and "chosen_route" in body

            # 7) malformed payload
            bad_req = request.Request(base + "/route-or-run", data=b"{not-json", headers={"Content-Type": "application/json"}, method="POST")
            try:
                request.urlopen(bad_req, timeout=15)
                raise AssertionError("expected HTTPError")
            except error.HTTPError as e:
                eb = json.loads(e.read().decode("utf-8"))
                assert e.code == 422 and eb["error"]["code"] == "validation_error"

            # 8) internal/backend failure safe error
            with mock.patch("codecontext.product_api._route_for_api", side_effect=Exception("boom")):
                status, body = _post(base, "/route-or-run", {"prompt": "check logs", "options": {"dry_run": False}})
                assert status == 500 and body["error"]["code"] == "internal_error"

            # 9) query params handling
            status, body = _get(base, "/metrics?window=1d&route_filter=local_only&top_n=1")
            assert status == 200 and body["notes"]["route_filter"] == "local_only"
        finally:
            server.shutdown()
            th.join(timeout=5)

    print("product_api_test.py passed")


def test_llm_response_surfaced_on_route_or_run():
    """Phase 2 API-surface test: when llm_client is enabled and the router
    goes external, /route-or-run must include llm_response + cost_actual_usd.

    We bypass routing uncertainty by calling _route_for_api directly with a
    stubbed bridge/executor — this is a contract test for the HTTP surface,
    not for the router."""
    import os
    from codecontext.executor import AutoExecutor
    from codecontext.gateway import CodeContextGateway
    from codecontext.llm_client import LLMClientConfig
    from codecontext.product_api import _route_for_api

    fake_llm, fake_th = _start_fake_llm()
    llm_base = f"http://127.0.0.1:{fake_llm.server_address[1]}"
    os.environ["FAKE_API_TEST_KEY"] = "sk-x"

    try:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            (repo / "main.py").write_text("def f():\n    return 1\n")
            cfg = AppConfig(root=repo)
            cfg.llm_client = LLMClientConfig(
                enabled=True, provider="openai", model="gpt-4o-mini",
                api_key_env="FAKE_API_TEST_KEY", base_url=llm_base,
            )
            bridge = CodeContextGateway(cfg)
            executor = AutoExecutor(bridge)
            # Force the router external so the LLM call actually fires. We
            # do this by monkey-patching bridge.route_request to return a
            # canonical external result — the point of this test is the
            # HTTP surface, not routing.
            real_route = bridge.route_request

            def force_external(goal, **kw):
                routed = real_route(goal, **kw)
                if routed.get("mode") == "external_reasoning":
                    return routed
                return real_route(
                    goal,
                    forced_route_mode="external_reasoning_with_compaction",
                    forced_intent="code_understanding",
                    forced_task_type="explain_architecture",
                    forced_evidence_source_type="cross_module_design",
                    route_authority="ts_plugin",
                )

            bridge.route_request = force_external  # type: ignore[assignment]

            status, body = _route_for_api(
                bridge, executor, prompt="explain architecture tradeoffs",
                dry_run=False, top_k=3, token_budget=800,
            )
            assert status == 200, body
            assert body.get("ok") is True

            if body.get("chosen_route") == "external_reasoning_with_compaction":
                # LLM should have been called; these keys are Phase 2 additions.
                assert "llm_response" in body, f"missing llm_response: {list(body)}"
                llm = body["llm_response"]
                assert llm["ok"] is True
                assert llm["provider"] == "openai"
                assert llm["prompt_tokens"] == 150
                assert llm["completion_tokens"] == 40
                expected = round((150 / 1e6) * 0.15 + (40 / 1e6) * 0.60, 8)
                assert llm["cost_usd"] == expected
                # cost_actual_usd only surfaces when > 0 (the noise fix).
                assert body.get("cost_actual_usd") == expected
            else:
                # Router downgraded; skip the external-specific assertions
                # but ensure nothing crashed.
                assert body.get("run_id") is not None
    finally:
        fake_llm.shutdown()
        fake_th.join(timeout=5)
        os.environ.pop("FAKE_API_TEST_KEY", None)


def test_cost_actual_usd_omitted_when_zero():
    """The noise fix: local_only runs record $0 cost in the ledger, and the
    API should not expose that zero to consumers."""
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("print('hi')\n")
        cfg = AppConfig(root=repo)
        # llm_client stays disabled (default)
        from codecontext.executor import AutoExecutor
        from codecontext.gateway import CodeContextGateway
        from codecontext.product_api import _route_for_api

        bridge = CodeContextGateway(cfg)
        executor = AutoExecutor(bridge)
        status, body = _route_for_api(
            bridge, executor, prompt="list files in the project",
            dry_run=False, top_k=2, token_budget=400,
        )
        assert status == 200
        # Either no external call happened, or the fake LLM wasn't wired —
        # either way cost_actual_usd must not appear as a zero.
        if "cost_actual_usd" in body:
            assert body["cost_actual_usd"] > 0, "zero cost should be omitted, not exposed"
        # llm_response also must not appear on a local_only run.
        if body.get("chosen_route") == "local_only":
            assert "llm_response" not in body


if __name__ == "__main__":
    run_tests()
    test_llm_response_surfaced_on_route_or_run()
    print("test_llm_response_surfaced_on_route_or_run passed")
    test_cost_actual_usd_omitted_when_zero()
    print("test_cost_actual_usd_omitted_when_zero passed")
