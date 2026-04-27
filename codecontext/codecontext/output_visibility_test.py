from __future__ import annotations

import json

from codecontext.output_visibility import format_tool_output, sanitize_payload_for_visibility
from codecontext.cli import main as cli_main


def test_normal_hides_internal_audit_raw():
    payload = {"runtime_operational_debug_audit": {"total_prompts": 42, "likely_local_only": 40, "likely_external": 2}}
    out = format_tool_output(payload, visibility_mode="normal")
    assert out["visibility_mode"] == "normal"
    assert "raw_output" not in out
    assert "debug_summary" not in out
    assert "Audit complete" in out["user_text"]


def test_debug_summary_compact_not_raw_dump():
    payload = {"runtime_operational_debug_audit": {"total_prompts": 42, "likely_local_only": 40, "likely_external": 2, "families": {"runtime_status": 18}}}
    out = format_tool_output(payload, visibility_mode="debug_summary")
    assert out["visibility_mode"] == "debug_summary"
    assert "debug_summary" in out
    assert "raw_output" not in out
    assert out["debug_summary"]["stats"]["total_prompts"] == 42


def test_raw_debug_includes_raw_output():
    payload = {"measurement_source": "execution_runs", "totals": {"requests": 200}}
    out = format_tool_output(payload, visibility_mode="raw_debug")
    assert out["visibility_mode"] == "raw_debug"
    assert "raw_output" in out
    assert "measurement_source" in out["raw_output"]


def test_large_raw_truncates():
    huge = "{" + '"x":"' + ("a" * 50000) + '"}'
    out = format_tool_output(huge, visibility_mode="raw_debug")
    assert "raw_output" in out
    assert "[truncated" in out["raw_output"]


def test_traceback_clean_in_normal_detailed_in_raw():
    tb = "Traceback (most recent call last):\nException: boom"
    normal = format_tool_output(tb, visibility_mode="normal")
    raw = format_tool_output(tb, visibility_mode="raw_debug")
    assert "hidden" in normal["user_text"].lower()
    assert "raw_output" in raw


def test_normal_suppresses_exec_trace_text():
    payload = "Exec completed: rg \"foo\" services/news_service.py:42"
    out = format_tool_output(payload, visibility_mode="normal")
    assert "hidden" in out["user_text"].lower()
    assert "raw_output" not in out


def test_debug_summary_compacts_tool_trace():
    payload = "Exec completed\nRead with lines 10-20\nrg hit services/news_service.py:42"
    out = format_tool_output(payload, visibility_mode="debug_summary")
    assert out["tool_output_class"] in {"structured_summary", "exec_trace", "search_trace", "file_read_trace", "code_navigation_trace"}
    assert "debug_summary" in out
    assert "raw_output" not in out


def test_raw_debug_keeps_full_tool_trace_with_guard():
    payload = "Exec completed\n" + ("services/news_service.py:99\n" * 2000)
    out = format_tool_output(payload, visibility_mode="raw_debug")
    assert "raw_output" in out
    assert "services/news_service.py:99" in out["raw_output"]
    assert ("[truncated" in out["raw_output"]) or (len(out["raw_output"]) <= 12050)


def test_sensitive_detail_keys_masked_in_normal():
    payload = {
        "ok": False,
        "error": "missing_result",
        "reason_detail": "{\"stage\":\"invalid_schema\",\"stderr\":\"Exec completed C:\\\\x.py:9\"}",
        "stderr": "Traceback (most recent call last): ...",
        "stdout": "Read with lines 1-3",
    }
    safe = sanitize_payload_for_visibility(payload, mode="normal")
    assert safe["error"] == "missing_result"
    # reason_detail is internal and removed in normal mode
    assert "reason_detail" not in safe
    assert safe["stderr"] == "[hidden]"
    assert safe["stdout"] == "[hidden]"


def test_backend_invalid_schema_details_masked_in_debug_summary():
    payload = {
        "ok": False,
        "stage": "invalid_schema",
        "error": "missing_result",
        "reason_detail": {
            "stderr": "Exec completed: rg foo services/news_service.py:42",
            "traceback": "Traceback (most recent call last): ...",
        },
    }
    safe = sanitize_payload_for_visibility(payload, mode="debug_summary")
    assert safe["stage"] == "invalid_schema"
    # internal key is preserved only as compact marker in debug_summary mode
    assert safe["reason_detail"] == "[internal omitted]"


def test_string_level_trace_sanitization_normal_and_debug():
    text = "Exec Completed\nRead with lines 10-20\nC:\\repo\\services\\news_service.py:42"
    normal = sanitize_payload_for_visibility(text, mode="normal")
    dbg = sanitize_payload_for_visibility(text, mode="debug_summary")
    assert isinstance(normal, str)
    assert "hidden" in normal.lower()
    assert isinstance(dbg, dict)
    assert dbg["hidden"] is True
    assert "trace_summary" in dbg


def test_machine_mode_emit_path_sanitizes_payload(capsys):
    rc = cli_main([
        "runtime-operational-debug-audit",
        "--root",
        ".",
        "--machine-json",
        "--visibility-mode",
        "normal",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # machine output is still JSON payload, but should not expose raw_output envelope fields
    assert isinstance(out, dict)
    assert "raw_output" not in out


def test_machine_mode_exception_path_masks_traceback(capsys):
    rc = cli_main([
        "auto-show-run",
        "--root",
        ".",
        "--run-id",
        "999999",
        "--machine-json",
        "--visibility-mode",
        "normal",
    ])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out.get("traceback") == "[hidden]"


def test_traceback_masking_in_normal_and_debug_summary():
    payload = {"error": "boom", "traceback": "Traceback (most recent call last): ..."}
    n = sanitize_payload_for_visibility(payload, mode="normal")
    d = sanitize_payload_for_visibility(payload, mode="debug_summary")
    assert n["traceback"] == "[hidden]"
    assert isinstance(d["traceback"], dict)
    assert d["traceback"]["hidden"] is True
