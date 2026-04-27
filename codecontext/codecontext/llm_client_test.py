"""Tests for the Phase 2 LLM client + executor integration.

We run against a real HTTP server (stdlib ThreadingHTTPServer) that
impersonates each of the four providers, so we exercise the actual
urllib path end-to-end — not a mocked urlopen.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from codecontext.config import AppConfig
from codecontext.executor import AutoExecutor
from codecontext.gateway import CodeContextGateway
from codecontext.llm_client import (
    LLMClient,
    LLMClientConfig,
    LLMResponse,
    _lookup_pricing,
    call_for_outbound,
    render_user_message,
)


# ---------------------------------------------------------------------------
# Fake HTTP server
# ---------------------------------------------------------------------------
class _FakeProviderHandler(BaseHTTPRequestHandler):
    # Set by the test harness on the server instance before requests arrive.
    recorded_requests: list[dict] = []
    response_spec: dict = {}

    def log_message(self, *a, **kw):  # silence test noise
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body_raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_raw.decode("utf-8"))
        except Exception:
            body = {"_raw": body_raw.decode("utf-8", errors="replace")}

        rec = {
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
        }
        self.server.recorded_requests.append(rec)  # type: ignore[attr-defined]

        spec = self.server.response_spec  # type: ignore[attr-defined]
        status = spec.get("status", 200)
        payload = spec.get("payload", {})
        out = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _start_fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProviderHandler)
    server.recorded_requests = []  # type: ignore[attr-defined]
    server.response_spec = {"status": 200, "payload": {}}  # type: ignore[attr-defined]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, th


def _stop(server, th):
    server.shutdown()
    th.join(timeout=5)


def _base_url(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# Unit tests — pricing table
# ---------------------------------------------------------------------------
def test_pricing_longest_prefix_match():
    # claude-sonnet-4 should match the 'claude-sonnet-4' key, not 'claude-3-sonnet'.
    ip, op, key = _lookup_pricing("claude-sonnet-4-20250514")
    assert key == "claude-sonnet-4", f"got {key}"
    assert (ip, op) == (3.00, 15.00)

    # gpt-4o-mini should beat gpt-4o.
    ip, op, key = _lookup_pricing("gpt-4o-mini-2024-07-18")
    assert key == "gpt-4o-mini"
    assert (ip, op) == (0.15, 0.60)

    # gpt-4o should beat gpt-4.
    ip, op, key = _lookup_pricing("gpt-4o")
    assert key == "gpt-4o"


def test_pricing_unknown_model_falls_back_to_default():
    ip, op, key = _lookup_pricing("some-random-model-xyz")
    assert key == "default"
    assert (ip, op) == (5.0, 15.0)


def test_pricing_user_override_wins():
    overrides = {"gpt-4o-mini": (99.0, 88.0)}
    ip, op, key = _lookup_pricing("gpt-4o-mini", overrides)
    assert (ip, op) == (99.0, 88.0)


def test_pricing_selfhosted_requires_explicit_opt_in():
    """Open-weight model names (qwen, llama, etc.) used to silently map to
    $0/$0 in the default pricing table. That silently under-billed users
    pointing at paid providers (Fireworks, Together, RunPod, DeepInfra).
    New contract: default pricing is the fallback rate; users who really
    self-host opt in via LLMClientConfig.pricing overrides."""
    # No override → falls back to default rate, not $0.
    ip, op, src = _lookup_pricing("qwen3-32b-fp8")
    assert (ip, op) == (5.0, 15.0), f"expected fallback pricing, got ({ip}, {op})"
    assert "default" in src
    ip, op, src = _lookup_pricing("llama-3.1-70b-instruct")
    assert (ip, op) == (5.0, 15.0), f"expected fallback pricing, got ({ip}, {op})"
    # Explicit opt-in override → $0.
    ip, op, _ = _lookup_pricing("qwen3-32b-fp8", {"qwen3-32b-fp8": (0.0, 0.0)})
    assert (ip, op) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Provider tests — each provider sends the correct shape + parses correctly
# ---------------------------------------------------------------------------
def test_openai_request_shape_and_response_parsing():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "id": "chatcmpl-xyz",
                "choices": [{"message": {"role": "assistant", "content": "hello world"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
            },
        }
        os.environ["FAKE_OPENAI_KEY"] = "sk-test"
        cfg = LLMClientConfig(
            enabled=True, provider="openai", model="gpt-4o-mini",
            api_key_env="FAKE_OPENAI_KEY", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        resp = client.complete(system="you are helpful", user="say hi")

        assert resp.ok, f"error: {resp.error_code} {resp.error_message}"
        assert resp.text == "hello world"
        assert resp.prompt_tokens == 123
        assert resp.completion_tokens == 45
        # gpt-4o-mini priced at (0.15, 0.60) per 1M.
        expected_cost = round((123 / 1e6) * 0.15 + (45 / 1e6) * 0.60, 8)
        assert resp.cost_usd == expected_cost, f"got {resp.cost_usd} expected {expected_cost}"
        assert resp.provider == "openai"
        assert resp.model == "gpt-4o-mini"

        # Request shape: Bearer auth, correct path, messages array.
        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        assert req["path"] == "/chat/completions"
        assert req["headers"].get("authorization") == "Bearer sk-test"
        assert req["body"]["model"] == "gpt-4o-mini"
        assert len(req["body"]["messages"]) == 2
        assert req["body"]["messages"][0]["role"] == "system"
        assert req["body"]["messages"][1]["role"] == "user"
    finally:
        _stop(server, th)
        os.environ.pop("FAKE_OPENAI_KEY", None)


def test_anthropic_request_shape_and_response_parsing():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "id": "msg_abc",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "sonnet reply"}],
                "usage": {"input_tokens": 80, "output_tokens": 20},
            },
        }
        os.environ["FAKE_ANTHROPIC_KEY"] = "sk-ant-test"
        cfg = LLMClientConfig(
            enabled=True, provider="anthropic", model="claude-sonnet-4-20250514",
            api_key_env="FAKE_ANTHROPIC_KEY", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        resp = client.complete(system="sys", user="hi")

        assert resp.ok, f"error: {resp.error_code} {resp.error_message}"
        assert resp.text == "sonnet reply"
        assert resp.prompt_tokens == 80
        assert resp.completion_tokens == 20
        # claude-sonnet-4 priced at (3.00, 15.00) per 1M.
        expected = round((80 / 1e6) * 3.00 + (20 / 1e6) * 15.00, 8)
        assert resp.cost_usd == expected

        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        assert req["path"] == "/v1/messages"
        assert req["headers"].get("x-api-key") == "sk-ant-test"
        assert req["headers"].get("anthropic-version") == "2023-06-01"
        assert req["body"]["system"] == "sys"
        assert req["body"]["messages"][0]["role"] == "user"
    finally:
        _stop(server, th)
        os.environ.pop("FAKE_ANTHROPIC_KEY", None)


def test_ollama_request_shape_and_response_parsing():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "model": "llama3",
                "message": {"role": "assistant", "content": "local model reply"},
                "prompt_eval_count": 40,
                "eval_count": 15,
                "done": True,
            },
        }
        cfg = LLMClientConfig(
            enabled=True, provider="ollama", model="llama3",
            api_key_env="", base_url=_base_url(server),
            # Ollama runs locally → explicitly declare it free. Previously
            # the default pricing table had a blanket `"llama": (0.0, 0.0)`
            # entry that silently zeroed cost for any model name starting
            # with "llama", including paid cloud-hosted llama variants.
            pricing={"llama3": (0.0, 0.0)},
        )
        client = LLMClient(cfg)
        resp = client.complete(system="sys", user="hi")

        assert resp.ok, f"error: {resp.error_code} {resp.error_message}"
        assert resp.text == "local model reply"
        assert resp.prompt_tokens == 40
        assert resp.completion_tokens == 15
        # Explicit pricing override → cost must be 0.
        assert resp.cost_usd == 0.0

        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        assert req["path"] == "/api/chat"
        assert req["body"]["stream"] is False
        assert req["body"]["options"]["num_predict"] > 0
    finally:
        _stop(server, th)


def test_openai_compatible_works_without_api_key():
    """vLLM / LM Studio often run without auth on localhost."""
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "choices": [{"message": {"content": "qwen reply"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }
        # Self-hosted qwen: user declares it free via explicit pricing
        # override. Previously the default pricing table contained
        # `"qwen": (0.0, 0.0)`, which silently zeroed cost for users
        # pointing at paid providers too (Fireworks, Together, RunPod).
        # Now the default is the fallback rate; users who self-host
        # must opt in as shown here.
        cfg = LLMClientConfig(
            enabled=True, provider="openai_compatible", model="qwen3-32b",
            api_key_env="", base_url=_base_url(server),
            pricing={"qwen3-32b": (0.0, 0.0)},
        )
        client = LLMClient(cfg)
        resp = client.complete(system="sys", user="hi")

        assert resp.ok, f"error: {resp.error_code} {resp.error_message}"
        assert resp.text == "qwen reply"
        # qwen is selfhosted — free (because pricing override says so).
        assert resp.cost_usd == 0.0
        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        # No Authorization header should have been sent.
        assert "authorization" not in req["headers"]
    finally:
        _stop(server, th)


def test_openai_compatible_with_api_key_sends_bearer():
    """Same compat shape, but cloud-hosted endpoints (Groq, Together, etc.)
    expect Bearer auth. When api_key_env is set and resolves, we send it."""
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "choices": [{"message": {"content": "groq reply"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        }
        os.environ["FAKE_GROQ_KEY"] = "gsk-test"
        cfg = LLMClientConfig(
            enabled=True, provider="openai_compatible", model="llama-3.1-70b",
            api_key_env="FAKE_GROQ_KEY", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        resp = client.complete(system="sys", user="hi")
        assert resp.ok
        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        assert req["headers"].get("authorization") == "Bearer gsk-test"
    finally:
        _stop(server, th)
        os.environ.pop("FAKE_GROQ_KEY", None)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_http_error_returns_structured_failure():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 429,
            "payload": {"error": {"message": "rate limited", "type": "rate_limit_error"}},
        }
        os.environ["FAKE_KEY"] = "sk"
        cfg = LLMClientConfig(
            enabled=True, provider="openai", model="gpt-4o-mini",
            api_key_env="FAKE_KEY", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        resp = client.complete(system="sys", user="user")
        assert resp.ok is False
        assert resp.error_code == "http_429"
        assert "rate limited" in (resp.error_message or "")
        assert resp.cost_usd == 0.0
    finally:
        _stop(server, th)
        os.environ.pop("FAKE_KEY", None)


def test_missing_api_key_returns_structured_failure():
    # No env var set, provider requires auth.
    cfg = LLMClientConfig(
        enabled=True, provider="openai", model="gpt-4o-mini",
        api_key_env="NONEXISTENT_ENV_VAR_XYZ", base_url="http://127.0.0.1:1",
    )
    client = LLMClient(cfg)
    resp = client.complete(system="sys", user="user")
    assert resp.ok is False
    assert resp.error_code == "config_error", f"got {resp.error_code}"
    assert "api key" in (resp.error_message or "").lower()


def test_unsupported_provider_returns_structured_failure():
    cfg = LLMClientConfig(enabled=True, provider="gemini", model="foo", api_key_env="")
    client = LLMClient(cfg)
    resp = client.complete(system="sys", user="user")
    assert resp.ok is False
    assert resp.error_code == "unsupported_provider"


def test_anthropic_missing_api_key_returns_config_error():
    cfg = LLMClientConfig(
        enabled=True, provider="anthropic", model="claude-sonnet-4",
        api_key_env="NONEXISTENT_ANTHROPIC_KEY_XYZ", base_url="http://127.0.0.1:1",
    )
    client = LLMClient(cfg)
    resp = client.complete(system="sys", user="user")
    assert resp.ok is False
    assert resp.error_code == "config_error"


# ---------------------------------------------------------------------------
# Fallback token estimation when provider omits usage block
# ---------------------------------------------------------------------------
def test_token_estimation_fallback_when_usage_missing():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "choices": [{"message": {"content": "a reply with some tokens in it"}}],
                # no usage block
            },
        }
        cfg = LLMClientConfig(
            enabled=True, provider="openai_compatible", model="qwen",
            api_key_env="", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        resp = client.complete(system="you are a helper", user="hi there")
        assert resp.ok
        # Fallback must produce non-zero token estimates.
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens > 0
    finally:
        _stop(server, th)


# ---------------------------------------------------------------------------
# Metrics overlay — the shape the ledger consumes
# ---------------------------------------------------------------------------
def test_as_metrics_overlay_keys_match_ledger_contract():
    resp = LLMResponse(
        ok=True, text="x", prompt_tokens=100, completion_tokens=50, total_tokens=150,
        cost_usd=0.01, provider="openai", model="gpt-4o-mini", latency_ms=42,
        pricing_source="openai/gpt-4o-mini",
    )
    overlay = resp.as_metrics_overlay()
    # These keys are exactly what usage_ledger.record_request_usage_conn reads.
    assert overlay["prompt_tokens_actual"] == 100
    assert overlay["completion_tokens_actual"] == 50
    assert overlay["cost_actual_usd"] == 0.01
    assert overlay["estimation_method"].startswith("llm_usage:")


# ---------------------------------------------------------------------------
# Backward compatibility — executor with llm_client disabled (default)
# ---------------------------------------------------------------------------
def test_executor_backward_compatible_when_llm_disabled():
    with tempfile.TemporaryDirectory() as td:
        # Work inside a clean dir so the gateway scanner is fast.
        repo = Path(td) / "repo"
        repo.mkdir()
        (repo / "main.py").write_text("print('hi')\n")
        cfg = AppConfig(root=repo)
        assert cfg.llm_client.enabled is False  # default

        bridge = CodeContextGateway(cfg)
        executor = AutoExecutor(bridge)
        assert executor._llm_client is None, "LLM client should not be built when disabled"

        out = executor.start(
            "check logs",
            forced_route_mode="local_only",
            forced_intent="runtime_diagnostics",
            forced_task_type="diagnose_runtime",
            forced_evidence_source_type="telemetry_state",
            route_authority="ts_plugin",
        )
        assert out["mode"] == "local_only"
        assert out.get("llm_response") is None
        assert "run_id" in out
        assert "usage_ledger" in out


# ---------------------------------------------------------------------------
# End-to-end — executor with llm_client enabled, real HTTP roundtrip
# ---------------------------------------------------------------------------
def test_executor_calls_llm_on_external_route_and_records_usage():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "choices": [{"message": {"content": json.dumps({"kind": "answer", "answer": "explained"})}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 60},
            },
        }
        os.environ["FAKE_KEY_E2E"] = "sk-e2e"
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            (repo / "main.py").write_text("def foo():\n    return 1\n")
            cfg = AppConfig(root=repo)
            cfg.llm_client = LLMClientConfig(
                enabled=True, provider="openai", model="gpt-4o-mini",
                api_key_env="FAKE_KEY_E2E", base_url=_base_url(server),
            )
            bridge = CodeContextGateway(cfg)
            executor = AutoExecutor(bridge)
            assert executor._llm_client is not None

            out = executor.start(
                "explain architecture tradeoffs",
                forced_route_mode="external_reasoning_with_compaction",
                forced_intent="code_understanding",
                forced_task_type="explain_architecture",
                forced_evidence_source_type="cross_module_design",
                route_authority="ts_plugin",
            )

            # If the router downgraded to local_only (no evidence), skip the
            # LLM-specific assertions — but this must still not crash.
            if out["mode"] != "external_reasoning":
                assert out.get("llm_response") is None
                return

            llm = out.get("llm_response")
            assert llm is not None, "llm_response must be attached"
            assert llm["ok"] is True
            assert llm["prompt_tokens"] == 200
            assert llm["completion_tokens"] == 60
            # 200 input * 0.15/1M + 60 output * 0.60/1M
            expected = round((200 / 1e6) * 0.15 + (60 / 1e6) * 0.60, 8)
            assert llm["cost_usd"] == expected

            # Ledger should have recorded the actual cost, not a $0 external call.
            ledger = out["usage_ledger"]
            assert ledger["cost_actual_usd"] == expected

            # Fake server should have received exactly one POST to chat/completions.
            reqs = server.recorded_requests  # type: ignore[attr-defined]
            assert len(reqs) == 1
            assert reqs[0]["path"] == "/chat/completions"
    finally:
        _stop(server, th)
        os.environ.pop("FAKE_KEY_E2E", None)


# ---------------------------------------------------------------------------
# render_user_message / call_for_outbound helpers
# ---------------------------------------------------------------------------
def test_render_user_message_is_deterministic_json():
    payload = {"schema_version": "codecontext.outbound.v1", "request": {"goal": "x"}}
    rendered = render_user_message(payload)
    assert json.loads(rendered) == payload


def test_call_for_outbound_passes_payload_through():
    server, th = _start_fake_server()
    try:
        server.response_spec = {  # type: ignore[attr-defined]
            "status": 200,
            "payload": {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        }
        cfg = LLMClientConfig(
            enabled=True, provider="openai_compatible", model="qwen",
            api_key_env="", base_url=_base_url(server),
        )
        client = LLMClient(cfg)
        outbound = {"schema_version": "codecontext.outbound.v1", "request": {"goal": "test"}}
        resp = call_for_outbound(client, outbound)
        assert resp.ok

        req = server.recorded_requests[0]  # type: ignore[attr-defined]
        user_msg = req["body"]["messages"][1]["content"]
        assert json.loads(user_msg) == outbound
        # System prompt should mention the envelope shapes.
        sys_msg = req["body"]["messages"][0]["content"]
        assert "kind" in sys_msg and "needs_more_context" in sys_msg
    finally:
        _stop(server, th)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_tests():
    import inspect
    import sys
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}", flush=True)
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)
            traceback.print_exc()
            failed += 1
    print(f"--- llm_client_test: passed={passed} failed={failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
