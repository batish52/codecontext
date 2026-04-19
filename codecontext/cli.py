from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .config import AppConfig
from .context_pack import ContextPackBuilder
from .executor import AutoExecutor
from .gateway import CodeContextGateway
from .metrics import Metrics
from .telemetry import benchmark_summary, boundary_compliance_report, integrated_path_benchmark, integrated_path_leakage_audit, persist_boundary_event, product_metrics_report, recent_observed_prompts_audit, runtime_operational_debug_audit
from .patcher import Patcher
from .scanner import ProjectScanner
from .search import SearchEngine
from .summaries import SummaryManager
from .utils import approx_tokens, json_dumps
from .watcher import ProjectWatcher
from .product_api import run_product_api
from .benchmark import run_benchmark, benchmark_report, benchmark_sales_summary, benchmark_leakage_audit
from .db import connect
from .ast_graph import ASTIndexer
from .git_context import GitContext
from .usage_ledger import cron_savings_report_conn, persist_cron_run_summary_conn, usage_ledger_report_conn
from .output_visibility import format_tool_output, safe_format_exception, sanitize_payload_for_visibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeContext — LLM cost optimization gateway")
    parser.add_argument("command", help="command to run")
    parser.add_argument("--root", default=".")
    parser.add_argument("--query")
    parser.add_argument("--goal")
    parser.add_argument("--path")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=1800)
    parser.add_argument("--old-text")
    parser.add_argument("--new-text")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--text")
    parser.add_argument("--patch-id", type=int)
    parser.add_argument("--response-text")
    parser.add_argument("--response-file")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--event-json")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--text-filter")
    parser.add_argument("--covered-class")
    parser.add_argument("--source-kind")
    parser.add_argument("--only-missed", action="store_true")
    parser.add_argument("--window")
    parser.add_argument("--route-filter")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--human-summary", action="store_true")
    parser.add_argument("--definitions", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--dataset")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--session-key")
    parser.add_argument("--cron-run-id")
    parser.add_argument("--request-id")
    parser.add_argument("--provider")
    parser.add_argument("--actual-model-name")
    parser.add_argument("--counterfactual-target-model")
    parser.add_argument("--cron-job-name", default="runtime-operational-debug-soak-audit")
    parser.add_argument("--window-hours", type=float, default=3.0)
    parser.add_argument("--visibility-mode", choices=["normal", "debug_summary", "raw_debug"], default=None)
    parser.add_argument("--raw-debug", action="store_true")
    parser.add_argument("--machine-json", action="store_true", help="emit strict machine contract JSON (no visibility envelope)")
    parser.add_argument("--forced-route-mode")
    parser.add_argument("--forced-intent")
    parser.add_argument("--forced-task-type")
    parser.add_argument("--forced-evidence-source-type")
    parser.add_argument("--route-authority")
    parser.add_argument("--heavy-execution-allowed", action="store_true")
    # --- LLM client (Phase 2) -------------------------------------------
    # All opt-in. If --llm-enabled is omitted, the executor returns the
    # outbound payload verbatim (Phase 1 behaviour).
    parser.add_argument("--llm-enabled", action="store_true",
                        help="enable the LLM client for external_reasoning routes")
    parser.add_argument("--llm-provider", choices=["openai", "anthropic", "ollama", "openai_compatible"],
                        default=None, help="LLM provider (default: openai)")
    parser.add_argument("--llm-model", default=None,
                        help="model name to call (e.g. gpt-4o-mini, claude-sonnet-4, qwen3-32b)")
    parser.add_argument("--llm-api-key-env", default=None,
                        help="name of the env var holding the API key (default: OPENAI_API_KEY)")
    parser.add_argument("--llm-base-url", default=None,
                        help="override endpoint base URL (useful for self-hosted / compat providers)")
    parser.add_argument("--llm-max-tokens", type=int, default=None,
                        help="max_tokens cap for the completion")
    parser.add_argument("--llm-temperature", type=float, default=None,
                        help="sampling temperature")
    parser.add_argument("--llm-timeout-seconds", type=float, default=None,
                        help="HTTP timeout per call")
    return parser


def _configure_utf8_stdio() -> None:
    # Windows console/code-page safety for JSON CLI transport.
    # Prevent UnicodeEncodeError (e.g., cp1252) when printing structured output.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Best-effort only; do not block command execution on stdio reconfigure failure.
        pass


HEAVY_GOAL_PATTERNS = [
    r"\bstress\s*test\b",
    r"\breplay\b",
    r"\bbenchmark\b",
    r"\bload\s*test\b",
    r"\bconcurren(t|cy)\b",
    r"\bvalidation\b.*\bprofile\b",
    r"\bheavy\s*load\b",
]


def _is_heavy_goal(text: str | None) -> bool:
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in HEAVY_GOAL_PATTERNS)


def _current_python_processes() -> int:
    try:
        p = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True, timeout=3)
        out = (p.stdout or "").lower()
        return sum(1 for line in out.splitlines() if "python.exe" in line)
    except Exception:
        return 0


def _heavy_execution_allowed(args: argparse.Namespace) -> bool:
    if args.heavy_execution_allowed:
        return True
    return os.getenv("CODECONTEXT_HEAVY_EXECUTION_ALLOWED", "0").strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    config = AppConfig(Path(args.root))
    # Apply CLI-level LLM client overrides. Each flag is independent — only
    # fields the user set on the command line are touched, the rest keep
    # their dataclass defaults.
    if getattr(args, "llm_enabled", False):
        config.llm_client.enabled = True
    if getattr(args, "llm_provider", None):
        config.llm_client.provider = args.llm_provider
    if getattr(args, "llm_model", None):
        config.llm_client.model = args.llm_model
    if getattr(args, "llm_api_key_env", None):
        config.llm_client.api_key_env = args.llm_api_key_env
    if getattr(args, "llm_base_url", None):
        config.llm_client.base_url = args.llm_base_url
    if getattr(args, "llm_max_tokens", None) is not None:
        config.llm_client.max_tokens = args.llm_max_tokens
    if getattr(args, "llm_temperature", None) is not None:
        config.llm_client.temperature = args.llm_temperature
    if getattr(args, "llm_timeout_seconds", None) is not None:
        config.llm_client.timeout_seconds = args.llm_timeout_seconds
    manager = SummaryManager(config)
    bridge = CodeContextGateway(config)
    executor = AutoExecutor(bridge)
    default_mode = os.getenv("CODECONTEXT_OUTPUT_VISIBILITY", "normal")
    visibility_mode = "raw_debug" if args.raw_debug else (args.visibility_mode or default_mode)
    internal_machine_commands = {
        "auto-start-request",
        "auto-continue-request",
        "auto-show-run",
        "boundary-event-log",
    }
    machine_mode = bool(args.machine_json or (args.command in internal_machine_commands))

    heavy_commands = {"benchmark-run", "benchmark-report", "benchmark-sales-summary", "benchmark-leakage-audit"}
    heavy_goal = _is_heavy_goal(args.goal) or _is_heavy_goal(args.query) or _is_heavy_goal(args.text)
    heavy_requested = (args.command in heavy_commands) or heavy_goal
    heavy_allowed = _heavy_execution_allowed(args)
    subprocess_cap = int(os.getenv("CODECONTEXT_MAX_PYTHON_SUBPROCESSES", "8"))

    def emit(payload):
        # Final response-boundary enforcement: always sanitize payload before rendering.
        safe_payload = sanitize_payload_for_visibility(payload, mode=visibility_mode)
        if machine_mode:
            print(json_dumps(safe_payload))
        else:
            print(json_dumps(format_tool_output(safe_payload, visibility_mode=visibility_mode)))

    try:
        if heavy_requested and not heavy_allowed:
            emit({
                "ok": True,
                "mode": "analysis_only_guard",
                "heavy_execution_allowed": False,
                "message": "Heavy local execution is disabled by default on this workspace. Re-run with --heavy-execution-allowed (or set CODECONTEXT_HEAVY_EXECUTION_ALLOWED=1) to execute stress/replay/benchmark workloads.",
                "command": args.command,
                "goal": args.goal,
            })
            return 0

        if heavy_requested:
            current_py = _current_python_processes()
            if current_py >= subprocess_cap:
                emit({
                    "ok": True,
                    "mode": "deferred_overload_guard",
                    "message": "Heavy task deferred: python subprocess cap reached.",
                    "current_python_processes": current_py,
                    "cap": subprocess_cap,
                    "command": args.command,
                })
                return 0

        if args.command == "scan-project":
            result = manager.index_project(metrics=Metrics())
            emit({"mode": "scan-project", "files": result["files"], "changed": result["changed"], "metrics": result["metrics"]})
            return 0
        if args.command == "index-project":
            emit(manager.index_project(metrics=Metrics()))
            return 0
        if args.command == "refresh-changed-files":
            emit(manager.refresh_changed_files(metrics=Metrics()))
            return 0
        if args.command == "search-project":
            if not args.query:
                parser.error("--query is required")
            emit(SearchEngine(config).search_project(args.query, top_k=args.top_k))
            return 0
        if args.command == "summarize-file":
            if not args.path:
                parser.error("--path is required")
            emit(manager.summarize_file(args.path))
            return 0
        if args.command == "summarize-project":
            emit(manager.summarize_project())
            return 0
        if args.command == "prepare-context-pack":
            if not args.goal:
                parser.error("--goal is required")
            emit(ContextPackBuilder(config).prepare(args.goal, top_k=args.top_k, token_budget=args.token_budget))
            return 0
        if args.command == "apply-patch":
            if not args.path or args.old_text is None or args.new_text is None:
                parser.error("--path, --old-text, and --new-text are required")
            emit(Patcher(config).apply_patch(args.path, args.old_text, args.new_text, dry_run=args.dry_run))
            return 0
        if args.command == "rollback-patch":
            if args.patch_id is None:
                parser.error("--patch-id is required")
            emit(Patcher(config).rollback_patch(args.patch_id))
            return 0
        if args.command == "watch-snapshot":
            watcher = ProjectWatcher(ProjectScanner(config))
            emit(asdict(watcher.snapshot()))
            return 0
        if args.command == "estimate-token-cost":
            if args.text is None:
                parser.error("--text is required")
            emit({"text_length": len(args.text), "estimated_tokens": approx_tokens(args.text)})
            return 0
        if args.command == "route-request":
            if not args.goal:
                parser.error("--goal is required")
            emit(bridge.route_request(args.goal, top_k=args.top_k, token_budget=args.token_budget))
            return 0
        if args.command == "handle-remote-response":
            response_text = args.response_text
            if args.response_file:
                response_text = Path(args.response_file).read_text(encoding="utf-8")
            if response_text is None:
                parser.error("--response-text or --response-file is required")
            emit(bridge.handle_remote_response(response_text, path=args.path, old_text=args.old_text, new_text=args.new_text, dry_run=args.dry_run, top_k=args.top_k, token_budget=args.token_budget))
            return 0
        if args.command == "auto-start-request":
            if not args.goal:
                parser.error("--goal is required")
            emit(executor.start(
                args.goal,
                top_k=args.top_k,
                token_budget=args.token_budget,
                session_key=args.session_key,
                cron_run_id=args.cron_run_id,
                request_id=args.request_id,
                provider=args.provider,
                actual_model_name=args.actual_model_name,
                counterfactual_target_model=args.counterfactual_target_model,
                forced_route_mode=args.forced_route_mode,
                forced_intent=args.forced_intent,
                forced_task_type=args.forced_task_type,
                forced_evidence_source_type=args.forced_evidence_source_type,
                route_authority=args.route_authority,
            ))
            return 0
        if args.command == "auto-continue-request":
            response_text = args.response_text
            if args.response_file:
                response_text = Path(args.response_file).read_text(encoding="utf-8")
            if args.run_id is None or response_text is None:
                parser.error("--run-id and --response-text or --response-file are required")
            emit(executor.continue_with_response(args.run_id, response_text, path=args.path, old_text=args.old_text, new_text=args.new_text, dry_run=args.dry_run, top_k=args.top_k, token_budget=args.token_budget))
            return 0
        if args.command == "auto-show-run":
            if args.run_id is None:
                parser.error("--run-id is required")
            emit(executor.get_run(args.run_id))
            return 0
        if args.command == "benchmark-summary":
            emit(benchmark_summary(config.db_path))
            return 0
        if args.command == "boundary-event-log":
            if not args.event_json:
                parser.error("--event-json is required")
            try:
                persist_boundary_event(config.db_path, json.loads(args.event_json))
                emit({"ok": True})
            except Exception as exc:
                # Telemetry must be best-effort and never break request execution.
                emit({"ok": False, "best_effort": True, "error": str(exc)})
                return 0
            return 0
        if args.command == "boundary-compliance-report":
            emit(boundary_compliance_report(config.db_path))
            return 0
        if args.command == "integrated-path-benchmark":
            emit(integrated_path_benchmark(config.db_path))
            return 0
        if args.command == "integrated-path-leakage-audit":
            emit(integrated_path_leakage_audit(config.db_path))
            return 0
        if args.command == "runtime-operational-debug-audit":
            emit(runtime_operational_debug_audit(config.db_path))
            return 0
        if args.command == "recent-observed-prompts-audit":
            emit(recent_observed_prompts_audit(config.db_path, limit=args.limit, text_filter=args.text_filter, covered_class=args.covered_class, only_missed=args.only_missed, source_kind=args.source_kind))
            return 0
        if args.command == "metrics-report":
            report = product_metrics_report(config.db_path, window=args.window, route_filter=args.route_filter, top_n=args.top_n, include_definitions=args.definitions)
            emit(report.get("sales_demo_summary", "") if args.human_summary else report)
            return 0
        if args.command == "sales-summary":
            report = product_metrics_report(config.db_path, window=args.window, route_filter=args.route_filter, top_n=args.top_n)
            emit(report.get("sales_demo_summary", ""))
            return 0
        if args.command == "serve-api":
            run_product_api(root=args.root, host=args.host, port=args.port)
            emit({"ok": True, "message": "API server started"})
            return 0
        if args.command == "benchmark-run":
            if not args.dataset:
                parser.error("--dataset is required")
            emit(run_benchmark(Path(args.root), dataset=args.dataset, runs=args.runs))
            return 0
        if args.command == "benchmark-report":
            if not args.dataset:
                parser.error("--dataset is required")
            emit(benchmark_report(args.dataset))
            return 0
        if args.command == "benchmark-sales-summary":
            if not args.dataset:
                parser.error("--dataset is required")
            emit(benchmark_sales_summary(args.dataset))
            return 0
        if args.command == "benchmark-leakage-audit":
            if not args.dataset:
                parser.error("--dataset is required")
            emit(benchmark_leakage_audit(args.dataset))
            return 0
        if args.command == "index-ast":
            indexer = ASTIndexer()
            index = indexer.index_project(config.root, config.exclude)
            call_graph = indexer.build_call_graph(index)
            import_graph = indexer.build_import_graph(index)

            conn = connect(config.db_path)
            now_ts = time.time()
            with conn:
                conn.execute("DELETE FROM call_graph")
                conn.execute("DELETE FROM import_graph")

                calls_found = 0
                for caller_key, callee_keys in call_graph.items():
                    caller_path, caller_name = caller_key.split(":", 1) if ":" in caller_key else ("", caller_key)
                    for callee_key in callee_keys:
                        callee_path, callee_name = callee_key.split(":", 1) if ":" in callee_key else (None, callee_key)
                        conn.execute(
                            "INSERT INTO call_graph(caller_path, caller_name, callee_path, callee_name, call_line, updated_at) VALUES(?,?,?,?,?,?)",
                            (caller_path, caller_name, callee_path, callee_name, None, now_ts),
                        )
                        calls_found += 1

                imports_found = 0
                for importer_path, deps in import_graph.items():
                    for imported_path in deps:
                        imported_module = imported_path.replace("/", ".").removesuffix(".py")
                        conn.execute(
                            "INSERT INTO import_graph(importer_path, imported_module, imported_path, updated_at) VALUES(?,?,?,?)",
                            (importer_path, imported_module, imported_path, now_ts),
                        )
                        imports_found += 1

            files_indexed = len(index.get("files", {}))
            functions_found = sum(len(v.get("functions", [])) for v in index.get("files", {}).values())
            emit({
                "ok": True,
                "files_indexed": files_indexed,
                "functions_found": functions_found,
                "calls_found": calls_found,
                "imports_found": imports_found,
                "warnings": indexer.warnings[:20],
            })
            conn.close()
            return 0
        if args.command == "callers-of":
            if not args.query:
                parser.error("--query is required")
            conn = connect(config.db_path)
            rows = conn.execute("SELECT caller_path, caller_name, callee_path, callee_name FROM call_graph").fetchall()
            graph: dict[str, list[str]] = {}
            for r in rows:
                caller = f"{r['caller_path']}:{r['caller_name']}"
                callee = f"{r['callee_path']}:{r['callee_name']}" if r['callee_path'] else r['callee_name']
                graph.setdefault(caller, [])
                if callee not in graph[caller]:
                    graph[caller].append(callee)
            out = ASTIndexer().callers_of(graph, args.query)
            emit({"target": args.query, "callers": out})
            conn.close()
            return 0
        if args.command == "callees-of":
            if not args.query:
                parser.error("--query is required")
            conn = connect(config.db_path)
            rows = conn.execute("SELECT caller_path, caller_name, callee_path, callee_name FROM call_graph").fetchall()
            graph: dict[str, list[str]] = {}
            for r in rows:
                caller = f"{r['caller_path']}:{r['caller_name']}"
                callee = f"{r['callee_path']}:{r['callee_name']}" if r['callee_path'] else r['callee_name']
                graph.setdefault(caller, [])
                if callee not in graph[caller]:
                    graph[caller].append(callee)
            out = ASTIndexer().callees_of(graph, args.query)
            emit({"target": args.query, "callees": out})
            conn.close()
            return 0
        if args.command == "dependents-of":
            if not args.path:
                parser.error("--path is required")
            conn = connect(config.db_path)
            rows = conn.execute("SELECT importer_path, imported_path FROM import_graph WHERE imported_path IS NOT NULL").fetchall()
            graph: dict[str, list[str]] = {}
            for r in rows:
                graph.setdefault(r["importer_path"], []).append(r["imported_path"])
            out = ASTIndexer().dependents_of(graph, args.path)
            emit({"target": args.path, "dependents": out})
            conn.close()
            return 0
        if args.command == "dependencies-of":
            if not args.path:
                parser.error("--path is required")
            conn = connect(config.db_path)
            rows = conn.execute("SELECT importer_path, imported_path FROM import_graph WHERE imported_path IS NOT NULL").fetchall()
            graph: dict[str, list[str]] = {}
            for r in rows:
                graph.setdefault(r["importer_path"], []).append(r["imported_path"])
            out = ASTIndexer().dependencies_of(graph, args.path)
            emit({"target": args.path, "dependencies": out})
            conn.close()
            return 0
        if args.command == "impact-analysis":
            if not args.path:
                parser.error("--path is required")
            conn = connect(config.db_path)
            rows = conn.execute("SELECT importer_path, imported_path FROM import_graph WHERE imported_path IS NOT NULL").fetchall()
            graph: dict[str, list[str]] = {}
            for r in rows:
                graph.setdefault(r["importer_path"], []).append(r["imported_path"])
            idx = ASTIndexer()
            direct = idx.dependents_of(graph, args.path)
            seen = set(direct)
            queue = list(direct)
            while queue:
                cur = queue.pop(0)
                for nxt in idx.dependents_of(graph, cur):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            emit({"target": args.path, "direct_dependents": sorted(direct), "transitive_dependents": sorted(seen)})
            conn.close()
            return 0
        if args.command == "git-recent":
            git_ctx = GitContext(config.root)
            emit(git_ctx.recent_commits(n=max(1, int(args.limit or 10))) if git_ctx.is_available() else [])
            return 0
        if args.command == "git-file-log":
            if not args.path:
                parser.error("--path is required")
            git_ctx = GitContext(config.root)
            emit(git_ctx.file_log(args.path, n=max(1, int(args.limit or 10))) if git_ctx.is_available() else [])
            return 0
        if args.command == "git-diff":
            git_ctx = GitContext(config.root)
            ref = args.query or "HEAD~1"
            emit(git_ctx.diff_summary(ref=ref) if git_ctx.is_available() else [])
            return 0
        if args.command == "git-file-diff":
            if not args.path:
                parser.error("--path is required")
            git_ctx = GitContext(config.root)
            ref = args.query or "HEAD~1"
            emit({"path": args.path, "ref": ref, "diff": git_ctx.file_diff(args.path, ref=ref) if git_ctx.is_available() else ""})
            return 0
        if args.command == "git-blame":
            if not args.path:
                parser.error("--path is required")
            git_ctx = GitContext(config.root)
            start_line = args.top_k if "--top-k" in raw_argv else 1
            end_line = args.token_budget if "--token-budget" in raw_argv else None
            emit(git_ctx.blame_summary(args.path, start_line=start_line, end_line=end_line) if git_ctx.is_available() else [])
            return 0
        if args.command == "git-changed-since":
            git_ctx = GitContext(config.root)
            hours = args.window_hours if "--window-hours" in raw_argv else 24.0
            emit(git_ctx.changed_files_since(hours=hours) if git_ctx.is_available() else [])
            return 0
        if args.command == "git-status":
            git_ctx = GitContext(config.root)
            if not git_ctx.is_available():
                emit({"branch": "", "changes": {"staged": [], "unstaged": [], "untracked": []}})
            else:
                emit({"branch": git_ctx.current_branch(), "changes": git_ctx.uncommitted_changes()})
            return 0
        if args.command == "usage-ledger-report":
            conn = connect(config.db_path)
            emit(usage_ledger_report_conn(conn, limit=args.limit))
            conn.close()
            return 0
        if args.command == "cron-savings-capture":
            conn = connect(config.db_path)
            out = persist_cron_run_summary_conn(
                conn,
                cron_job_name=args.cron_job_name,
                window_seconds=max(1, int(args.window_hours * 3600)),
                cron_run_id=args.cron_run_id,
                notes="captured from CLI",
            )
            conn.commit()
            conn.close()
            emit(out)
            return 0
        if args.command == "cron-savings-report":
            conn = connect(config.db_path)
            emit(cron_savings_report_conn(conn, limit=args.limit))
            conn.close()
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
    except Exception as exc:
        if machine_mode:
            payload = {"ok": False, "error": str(exc), "exception_type": exc.__class__.__name__, "traceback": repr(exc)}
            print(json_dumps(sanitize_payload_for_visibility(payload, mode=visibility_mode)))
        else:
            print(json_dumps(safe_format_exception(exc, visibility_mode=visibility_mode)))
        return 1
