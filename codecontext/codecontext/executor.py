from __future__ import annotations

import json
import time
from dataclasses import asdict

from .db import connect
from .gateway import CodeContextGateway
from .llm_client import LLMClient, call_for_outbound
from .telemetry import benchmark_summary, persist_route_metrics_conn
from .usage_ledger import record_request_usage_conn


class AutoExecutor:
    def __init__(self, bridge: CodeContextGateway):
        self.bridge = bridge
        self.config = bridge.config
        # Lazily construct the LLM client only when the config enables it.
        # This keeps Phase 1 behaviour byte-identical for anyone not opting in.
        self._llm_client: LLMClient | None = None
        llm_cfg = getattr(self.config, "llm_client", None)
        if llm_cfg is not None and getattr(llm_cfg, "enabled", False):
            self._llm_client = LLMClient(llm_cfg)

    def start(
        self,
        goal: str,
        top_k: int = 6,
        token_budget: int = 1800,
        *,
        session_key: str | None = None,
        cron_run_id: str | None = None,
        request_id: str | None = None,
        provider: str | None = None,
        actual_model_name: str | None = None,
        counterfactual_target_model: str | None = None,
        forced_route_mode: str | None = None,
        forced_intent: str | None = None,
        forced_task_type: str | None = None,
        forced_evidence_source_type: str | None = None,
        route_authority: str | None = None,
    ) -> dict:
        routed = self.bridge.route_request(
            goal,
            top_k=top_k,
            token_budget=token_budget,
            forced_route_mode=forced_route_mode,
            forced_intent=forced_intent,
            forced_task_type=forced_task_type,
            forced_evidence_source_type=forced_evidence_source_type,
            route_authority=route_authority,
        )

        # --- External LLM call (opt-in) -----------------------------------
        # Only fire when (a) the client is configured + enabled, (b) the
        # router actually decided to go external, and (c) an outbound
        # payload was produced. If any of those fail we fall through to the
        # Phase 1 behaviour — return the outbound payload verbatim.
        llm_response = None
        llm_reason_detail = (routed.get("debug") or {}).get("external_escalation_reason")
        outbound_payload = routed.get("outbound_payload")
        if (
            self._llm_client is not None
            and routed.get("mode") == "external_reasoning"
            and outbound_payload is not None
        ):
            llm_response = call_for_outbound(self._llm_client, outbound_payload)
            # Overlay the real token counts + cost onto the metrics dict so
            # they get persisted by persist_route_metrics_conn AND picked up
            # by record_request_usage_conn below. We also override the
            # caller-supplied provider/model so the ledger records the
            # provider that actually ran the request, not the counterfactual
            # default.
            metrics_mut = routed.get("metrics") or {}
            if llm_response.ok:
                metrics_mut.update(llm_response.as_metrics_overlay())
            else:
                # Record the failure cost-neutrally but surface the reason.
                metrics_mut.setdefault("prompt_tokens_actual", 0)
                metrics_mut.setdefault("completion_tokens_actual", 0)
                metrics_mut.setdefault("cost_actual_usd", 0.0)
                metrics_mut["estimation_method"] = f"llm_error:{llm_response.error_code}"
                # Append error detail so it lands in the ledger's reason_detail.
                suffix = f"llm_error:{llm_response.error_code}:{llm_response.error_message or ''}"
                llm_reason_detail = f"{llm_reason_detail}|{suffix}" if llm_reason_detail else suffix
            routed["metrics"] = metrics_mut
            # Attach the raw response to the routed dict so callers (and
            # consumers of execution_runs.result_json) can see what the
            # model said without a second DB read.
            routed["llm_response"] = {
                "ok": llm_response.ok,
                "text": llm_response.text,
                "provider": llm_response.provider,
                "model": llm_response.model,
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "cost_usd": llm_response.cost_usd,
                "latency_ms": llm_response.latency_ms,
                "pricing_source": llm_response.pricing_source,
                "error_code": llm_response.error_code,
                "error_message": llm_response.error_message,
            }
            # If the caller didn't pin provider/model explicitly, use what
            # actually ran so the pricing snapshot in the ledger is correct.
            if provider is None and llm_response.ok:
                provider = llm_response.provider
            if actual_model_name is None and llm_response.ok:
                actual_model_name = llm_response.model

        now = time.time()
        conn = connect(self.config.db_path)
        try:
            # Bug #26: previously `metrics_json` was set to
            # `routed.get("metrics", {})` while a richer fallback
            # (with route_mode/local_only/etc.) was used a few lines
            # later for `persist_route_metrics_conn` and the usage
            # ledger. When the routed dict didn't carry a "metrics"
            # key, `execution_runs.metrics_json` got `"{}"` while
            # `request_metrics` got the enriched payload — same run,
            # divergent storage. Build the fallback once and use it
            # everywhere so the two tables agree.
            metrics = routed.get("metrics") or {
                "route_mode": routed["mode"],
                "local_only": int(routed["mode"] == "local_only"),
                "external_reasoning": int(routed["mode"] == "external_reasoning"),
                "evidence_count": 0,
                "evidence_chars": 0,
                "evidence_tokens": 0,
                "prompt_tokens_est": 0,
                "avoided_tokens_est": 0,
                "compaction_chars_saved": 0,
                "fallback_used": 0,
            }
            with conn:
                cur = conn.execute(
                    "INSERT INTO execution_runs(goal, route_mode, intent, outbound_json, result_json, metrics_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        goal,
                        routed["mode"],
                        (routed.get("route") or {}).get("intent"),
                        json.dumps(routed.get("outbound_payload"), ensure_ascii=False) if routed.get("outbound_payload") else None,
                        json.dumps(routed, ensure_ascii=False),
                        json.dumps(metrics, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                run_id = cur.lastrowid
                persist_route_metrics_conn(conn, metrics)
                ledger_row = record_request_usage_conn(
                    conn,
                    goal=goal,
                    route_mode=routed["mode"],
                    run_id=run_id,
                    session_key=session_key,
                    cron_run_id=cron_run_id,
                    request_id=request_id,
                    provider=provider,
                    actual_model_name=actual_model_name,
                    counterfactual_target_model=counterfactual_target_model,
                    metrics=metrics,
                    fallback_used=bool((routed.get("debug") or {}).get("fallback_used") or metrics.get("fallback_used")),
                    reason_code=(routed.get("debug") or {}).get("route_action"),
                    reason_detail=llm_reason_detail,
                    intent=(routed.get("route") or {}).get("intent"),
                )
            return {
                "run_id": run_id,
                "mode": routed["mode"],
                "result": routed,
                "usage_ledger": ledger_row,
                "llm_response": routed.get("llm_response"),
            }
        finally:
            conn.close()

    def continue_with_response(
        self,
        run_id: int,
        response_text: str,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        dry_run: bool = False,
        top_k: int = 6,
        token_budget: int = 1800,
    ) -> dict:
        outcome = self.bridge.handle_remote_response(
            response_text=response_text,
            path=path,
            old_text=old_text,
            new_text=new_text,
            dry_run=dry_run,
            top_k=top_k,
            token_budget=token_budget,
        )
        conn = connect(self.config.db_path)
        now = time.time()
        try:
            with conn:
                conn.execute(
                    "UPDATE execution_runs SET result_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(outcome, ensure_ascii=False), now, run_id),
                )
            return {"run_id": run_id, "outcome": outcome}
        finally:
            conn.close()

    def get_run(self, run_id: int) -> dict:
        conn = connect(self.config.db_path)
        try:
            row = conn.execute(
                "SELECT id, goal, route_mode, intent, outbound_json, result_json, metrics_json, created_at, updated_at FROM execution_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"run_id not found: {run_id}")
            return {
                "id": row["id"],
                "goal": row["goal"],
                "route_mode": row["route_mode"],
                "intent": row["intent"] if "intent" in row.keys() else None,
                "outbound_payload": json.loads(row["outbound_json"]) if row["outbound_json"] else None,
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
                "benchmark": benchmark_summary(self.config.db_path),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()
