from __future__ import annotations

from dataclasses import asdict
import re

from .config import AppConfig
from .context_pack import ContextPackBuilder
from .outbound_schema import build_outbound_request
from .patcher import Patcher
from .telemetry import build_route_metrics, runtime_diagnostics_summary
from .response_handler import ResponseHandler
from .router import RequestRouter, RoutedRequest
from .routing_contract import TS_INTERCEPT_TO_CANONICAL, normalize_route_mode
from .search import SearchEngine
from .summaries import SummaryManager
from .ranking import wants_exact_symbol_body


class CodeContextGateway:
    """Core routing gateway — classifies prompts and decides local vs external execution."""

    def _explain_subsystem_key(self, path: str | None) -> str:
        path = (path or "").replace("\\", "/")
        if path.startswith("codecontext-runtime/") or path.startswith("codecontext-runtime/"):
            return "runtime_plugin"
        if path.startswith("codecontext/"):
            return "codecontext_core"
        if path in {"requirements.txt", "MEMORY.md"} or path.startswith("memory/"):
            return "meta_support"
        return (path.split("/")[0] if path else "") or "unknown"

    def _support_signal_is_strong(self, evidence: dict | None) -> bool:
        if not evidence:
            return False
        return evidence.get("evidence_type") in {"semantic_chunk", "exact_symbol_body"}

    def _is_mechanism_style_wording(self, lowered: str) -> bool:
        return any(phrase in lowered for phrase in {
            "explain how",
            "how does",
            "walk me through",
            "how is",
        })

    def _explain_style_local_only_decision(self, goal: str, local_search: dict, context_pack: dict, routed) -> dict:
        lowered = (goal or "").lower()
        explanation_oriented = self._is_mechanism_style_wording(lowered) or any(term in lowered for term in {"explain", "how"})
        non_trivial_reasoning = any(term in lowered for term in {
            "architecture", "tradeoff", "trade-off", "design", "compare", "comparison", "debug", "bug", "root cause", "root-cause", "why"
        })
        chosen_core = context_pack.get("debug", {}).get("chosen_core_evidence") or {}
        chosen_support = context_pack.get("debug", {}).get("chosen_support_evidence") or {}
        support_debug = context_pack.get("debug", {}).get("support_choice_debug", [])
        evidence = context_pack.get("evidence", [])
        evidence_paths = [e.get("path") for e in evidence if e.get("path")]
        meaningful_subsystems = {
            self._explain_subsystem_key(p)
            for p in evidence_paths
            if self._explain_subsystem_key(p) not in {"meta_support", "unknown"}
        }
        cross_subsystem = len(meaningful_subsystems) > 1
        mechanism_runtime_family = any(term in lowered for term in {"runtime", "interceptor", "plugin"}) and self._is_mechanism_style_wording(lowered)
        selected_subsystems = {
            self._explain_subsystem_key(chosen_core.get("path")),
            self._explain_subsystem_key(chosen_support.get("path")),
        } - {"meta_support", "unknown", ""}
        if mechanism_runtime_family and selected_subsystems and selected_subsystems <= {"runtime_plugin", "codecontext_core"}:
            cross_subsystem = False
        same_path_only = bool(chosen_core.get("path")) and all((e.get("path") == chosen_core.get("path")) for e in evidence)
        same_file_multi_chunk = bool(chosen_core.get("path")) and same_path_only and len(evidence) >= 2
        support_signal_strong = self._support_signal_is_strong(chosen_support)
        same_mechanism_metadata_support = (
            bool(chosen_support.get("path"))
            and chosen_support.get("evidence_type") == "lexical_hit"
            and self._explain_subsystem_key(chosen_support.get("path")) == "runtime_plugin"
            and self._explain_subsystem_key(chosen_core.get("path")) == "runtime_plugin"
            and self._is_mechanism_style_wording(lowered)
            and any(term in lowered for term in {"runtime", "interceptor", "plugin", "work"})
        )
        core_alone_sufficient = bool(chosen_core.get("path")) and same_file_multi_chunk and not any(term in lowered for term in {"tradeoff", "trade-off", "design", "debug", "bug", "compare", "comparison", "why"})
        runtime_mechanism_core_alone_sufficient = (
            bool(chosen_core.get("path"))
            and self._explain_subsystem_key(chosen_core.get("path")) == "runtime_plugin"
            and self._is_mechanism_style_wording(lowered)
            and any(term in lowered for term in {"runtime", "interceptor", "plugin", "work"})
            and not any(term in lowered for term in {"tradeoff", "trade-off", "design", "debug", "bug", "compare", "comparison", "why", "architecture"})
        )
        searchengine_mechanism_core_alone_sufficient = (
            bool(chosen_core.get("path"))
            and (chosen_core.get("path") or "").replace('\\', '/').startswith('codecontext/search.py')
            and self._is_mechanism_style_wording(lowered)
            and 'searchengine' in lowered
            and any(term in lowered for term in {'bm25', 'symbol retrieval'})
            and not any(term in lowered for term in {"tradeoff", "trade-off", "design", "debug", "bug", "compare", "comparison", "why", "architecture"})
        )
        two_axis_override_used = routed.task_type in {"explain_mechanism", "explain_architecture"}
        preferred_evidence_type = (
            "cross_module_design" if routed.task_type == "explain_architecture" else
            "code_mechanism_family" if routed.task_type == "explain_mechanism" else
            None
        )
        confidence = 0.0
        if explanation_oriented:
            confidence += 0.15
        if routed.intent == "code_understanding":
            confidence += 0.15
        if two_axis_override_used:
            confidence += 0.1
        if routed.task_type == "explain_mechanism":
            confidence += 0.1
        if chosen_core.get("path") and chosen_core.get("evidence_type") in {"semantic_chunk", "lexical_hit", "exact_symbol_body"}:
            confidence += 0.25
        if routed.evidence_source_type == preferred_evidence_type and preferred_evidence_type is not None:
            confidence += 0.05
        if support_signal_strong or same_mechanism_metadata_support:
            confidence += 0.1
        elif core_alone_sufficient or runtime_mechanism_core_alone_sufficient or searchengine_mechanism_core_alone_sufficient:
            confidence += 0.1
        elif chosen_core.get("path") and not same_path_only:
            confidence += 0.05
        if not context_pack.get("debug", {}).get("fallback_used"):
            confidence += 0.1
        confidence = round(min(confidence, 1.0), 3)

        if routed.task_type == "explain_architecture":
            eligible = False
            reason = "2-axis explain override: architecture/tradeoff explain requests default to external reasoning"
        else:
            mechanism_like = routed.task_type == "explain_mechanism" or (not two_axis_override_used and routed.intent == "code_understanding")
            mechanism_evidence_ok = (
                routed.evidence_source_type == "code_mechanism_family"
                or preferred_evidence_type == "code_mechanism_family"
            )
            eligible = (
                mechanism_like
                and explanation_oriented
                and not non_trivial_reasoning
                and not cross_subsystem
                and mechanism_evidence_ok
                and (support_signal_strong or same_mechanism_metadata_support or core_alone_sufficient or runtime_mechanism_core_alone_sufficient or searchengine_mechanism_core_alone_sufficient)
                and bool(chosen_core.get("path"))
                and not context_pack.get("debug", {}).get("fallback_used")
                and confidence >= 0.8
            )
            if eligible:
                reason = "2-axis explain override: mechanism-style request has sufficient local code-mechanism evidence"
            elif non_trivial_reasoning:
                reason = "explain-style request asks for higher-order reasoning or judgment"
            elif not mechanism_evidence_ok:
                reason = "2-axis explain override: mechanism-style request lacked preferred code-mechanism evidence"
            elif not support_signal_strong and not same_mechanism_metadata_support and not core_alone_sufficient and not runtime_mechanism_core_alone_sufficient and not searchengine_mechanism_core_alone_sufficient:
                reason = "explain-style support evidence was too weak or non-mechanistic for local-only answer"
            elif same_path_only and not core_alone_sufficient:
                reason = "explain-style evidence stayed too concentrated in one file for local-only answer"
            elif cross_subsystem:
                reason = "explain-style request spans multiple subsystems"
            else:
                reason = "explain-style confidence was insufficient for local-only answer"
        return {
            "eligible": eligible,
            "reason": reason,
            "confidence": confidence,
            "selected_core_evidence": chosen_core,
            "selected_support_evidence": chosen_support,
            "support_choice_debug": support_debug,
            "preferred_evidence_source_type": preferred_evidence_type,
            "two_axis_override_used": two_axis_override_used,
            "external_escalation_reason": None if eligible else reason,
        }

    def _exact_body_local_only_decision(self, goal: str, local_search: dict, context_pack: dict) -> dict:
        lowered = (goal or "").lower()
        retrieval_only = wants_exact_symbol_body(goal) and not any(term in lowered for term in {
            "explain", "why", "how", "compare", "comparison", "refactor", "rewrite", "modify", "edit", "change", "fix"
        })
        helper_context_requested = any(term in lowered for term in {"helper", "helpers", "nearby", "adjacent", "surrounding", "context"})
        results = local_search.get("results", [])
        exact_candidates = [r for r in results if r.get("best_exact_strength", 0) >= 2]
        chosen = context_pack.get("debug", {}).get("chosen_core_evidence") or {}
        exact_details = context_pack.get("debug", {}).get("exact_body_details") or {}
        confidence = 0.0
        if chosen.get("evidence_type") == "exact_symbol_body":
            confidence += 0.7
        if len(exact_candidates) == 1:
            confidence += 0.2
        if not context_pack.get("debug", {}).get("fallback_used"):
            confidence += 0.1
        confidence = round(min(confidence, 1.0), 3)
        eligible = (
            retrieval_only
            and not helper_context_requested
            and chosen.get("evidence_type") == "exact_symbol_body"
            and len(exact_candidates) == 1
            and not context_pack.get("debug", {}).get("fallback_used")
            and bool(chosen.get("path"))
        )
        reason = "single strong exact symbol-body match with retrieval-only request" if eligible else (
            "non-trivial exact-body request requires external reasoning"
            if helper_context_requested or not retrieval_only else
            "exact-body confidence or uniqueness was insufficient for local-only answer"
        )
        return {
            "eligible": eligible,
            "reason": reason,
            "confidence": confidence,
            "symbol_candidates_considered": [
                {
                    "path": r.get("path"),
                    "best_exact_strength": r.get("best_exact_strength", 0),
                    "score": r.get("score", 0),
                    "match_count": len(r.get("matches", [])),
                }
                for r in exact_candidates[:5]
            ],
            "selected_body": {
                "path": chosen.get("path"),
                "start_line": chosen.get("start_line"),
                "end_line": chosen.get("end_line"),
                "snippet_text": next((e.get("snippet_text") for e in context_pack.get("evidence", []) if e.get("path") == chosen.get("path") and e.get("start_line") == chosen.get("start_line")), None),
            } if chosen.get("evidence_type") == "exact_symbol_body" else None,
            "exact_body_details": exact_details,
            "external_escalation_reason": None if eligible else reason,
        }

    def _log_error_local_only_decision(self, goal: str, local_search: dict, context_pack: dict, routed) -> dict:
        lowered = (goal or "").lower()
        chosen_core = context_pack.get("debug", {}).get("chosen_core_evidence") or {}
        chosen_support = context_pack.get("debug", {}).get("chosen_support_evidence") or {}
        support_debug = context_pack.get("debug", {}).get("support_choice_debug", [])
        evidence = context_pack.get("evidence", [])
        if not chosen_core and evidence:
            chosen_core = {
                "path": evidence[0].get("path"),
                "start_line": evidence[0].get("start_line"),
                "end_line": evidence[0].get("end_line"),
                "evidence_type": evidence[0].get("evidence_type"),
            }
        evidence_types = {e.get("evidence_type") for e in evidence}
        same_path_only = bool(chosen_core.get("path")) and all((e.get("path") == chosen_core.get("path")) for e in evidence)
        local_log_wording = any(term in lowered for term in {"log", "logs", "stderr", "stdout"})
        inspect_logs_like = routed.task_type == "inspect_logs"
        debug_root_cause_like = routed.task_type == "debug_root_cause"
        two_axis_override_used = inspect_logs_like or debug_root_cause_like
        preferred_evidence_type = (
            "runtime_logs" if inspect_logs_like else
            "traceback" if debug_root_cause_like else
            None
        )
        strong_log_evidence = (
            chosen_core.get("path")
            and chosen_core.get("evidence_type") in {"semantic_chunk", "lexical_hit", "exact_symbol_body"}
            and any(t in evidence_types for t in {"semantic_chunk", "lexical_hit", "exact_symbol_body"})
        )
        traceback_evidence_present = (
            any(term in lowered for term in {"traceback", "exception", "stack", "stacktrace"})
            or chosen_core.get("evidence_type") == "exact_symbol_body"
            or chosen_support.get("evidence_type") == "exact_symbol_body"
        )
        trivially_answerable_from_local = (
            debug_root_cause_like
            and traceback_evidence_present
            and bool(chosen_core.get("path"))
            and same_path_only
            and not context_pack.get("debug", {}).get("fallback_used")
            and not any(term in lowered for term in {"across modules", "systemic", "architecture", "tradeoff", "design", "multiple subsystems"})
        )
        confidence = 0.0
        if routed.intent in {"bug_hunt", "runtime_diagnostics"}:
            confidence += 0.2
        if two_axis_override_used:
            confidence += 0.15
        if routed.evidence_source_type == preferred_evidence_type and preferred_evidence_type is not None:
            confidence += 0.1
        if strong_log_evidence:
            confidence += 0.25
        if bool(chosen_core.get("path")):
            confidence += 0.1
        if not context_pack.get("debug", {}).get("fallback_used"):
            confidence += 0.1
        if inspect_logs_like and local_log_wording:
            confidence += 0.1
        if debug_root_cause_like and traceback_evidence_present:
            confidence += 0.1
        confidence = round(min(confidence, 1.0), 3)

        if inspect_logs_like:
            eligible = (
                local_log_wording
                and strong_log_evidence
                and bool(chosen_core.get("path"))
                and confidence >= 0.75
            )
            if eligible:
                reason = "2-axis log/error override: inspect-logs request has sufficient local runtime-log evidence"
            elif not local_log_wording:
                reason = "2-axis log/error override: inspect-logs request lacked explicit local log wording"
            elif not strong_log_evidence:
                reason = "2-axis log/error override: inspect-logs request lacked sufficient local runtime-log evidence"
            else:
                reason = "log/error confidence was insufficient for local-only answer"
        elif debug_root_cause_like:
            eligible = trivially_answerable_from_local and confidence >= 0.9
            if eligible:
                reason = "2-axis log/error override: root-cause request is trivially answerable from local traceback evidence"
            elif not traceback_evidence_present:
                reason = "2-axis log/error override: root-cause request lacked preferred traceback evidence"
            else:
                reason = "2-axis log/error override: root-cause requests default to external reasoning unless trivially answerable from local traceback alone"
        else:
            eligible = False
            reason = "log/error 2-axis override not applicable"

        return {
            "eligible": eligible,
            "reason": reason,
            "confidence": confidence,
            "selected_core_evidence": chosen_core,
            "selected_support_evidence": chosen_support,
            "support_choice_debug": support_debug,
            "preferred_evidence_source_type": preferred_evidence_type,
            "two_axis_override_used": two_axis_override_used,
            "external_escalation_reason": None if eligible else reason,
        }

    def _runtime_operational_debug_decision(self, goal: str, local_search: dict, context_pack: dict, routed) -> dict:
        lowered = (goal or "").lower()
        evidence = context_pack.get("evidence", [])
        chosen_core = context_pack.get("debug", {}).get("chosen_core_evidence") or {}
        chosen_support = context_pack.get("debug", {}).get("chosen_support_evidence") or {}
        support_debug = context_pack.get("debug", {}).get("support_choice_debug", [])
        if not chosen_core and evidence:
            chosen_core = {
                "path": evidence[0].get("path"),
                "start_line": evidence[0].get("start_line"),
                "end_line": evidence[0].get("end_line"),
                "evidence_type": evidence[0].get("evidence_type"),
            }

        status_terms = {"is the plugin working", "status", "health", "still running", "alive"}
        fallback_terms = {"did fallback happen", "fallback"}
        route_mix_terms = {"local vs external", "how much was local vs external"}
        logs_terms = {"check logs", "what logs", "which logs", "inspect logs", "stderr", "stdout", "logs"}
        external_only_terms = {"traceback", "exception", "stack", "stacktrace", "why", "root cause", "cause", "diagnos"}

        is_status_prompt = any(term in lowered for term in status_terms)
        is_fallback_prompt = any(term in lowered for term in fallback_terms)
        is_route_mix_prompt = any(term in lowered for term in route_mix_terms)
        is_logs_prompt = any(term in lowered for term in logs_terms)
        is_external_only_prompt = any(term in lowered for term in external_only_terms)

        operational_prompt = is_status_prompt or is_fallback_prompt or is_route_mix_prompt or is_logs_prompt or is_external_only_prompt

        preferred_evidence_source = (
            "traceback" if any(term in lowered for term in {"traceback", "exception", "stack", "stacktrace"}) else
            "runtime_logs" if is_logs_prompt else
            "runtime_execution_trace" if is_fallback_prompt or is_route_mix_prompt else
            "runtime_process_state" if is_status_prompt else
            routed.evidence_source_type
        )
        evidence_available = bool(evidence) or bool(chosen_core.get("path"))
        recognized_task_type = "runtime_operational_debug" if operational_prompt else routed.task_type
        confidence = 0.0
        if operational_prompt:
            confidence += 0.35
        if preferred_evidence_source in {"runtime_process_state", "runtime_execution_trace", "runtime_logs", "traceback"}:
            confidence += 0.2
        if evidence_available:
            confidence += 0.15
        if not context_pack.get("debug", {}).get("fallback_used"):
            confidence += 0.1
        if is_status_prompt or is_fallback_prompt or is_route_mix_prompt or is_logs_prompt:
            confidence += 0.1
        confidence = round(min(confidence, 1.0), 3)

        if not operational_prompt:
            eligible = False
            reason = "runtime operational override not applicable"
            route_mode = None
            route_action = None
        elif is_external_only_prompt:
            eligible = False
            reason = "runtime operational debug defaults to external reasoning for traceback interpretation or causal diagnosis"
            route_mode = "external_reasoning"
            route_action = "runtime_operational_debug_external"
        elif is_status_prompt:
            eligible = True
            reason = "runtime operational override: status check can be answered from local runtime process-state evidence"
            route_mode = "local_only"
            route_action = "runtime_operational_status_local"
        elif is_fallback_prompt:
            eligible = True
            reason = "runtime operational override: fallback check can be answered from local execution-trace evidence"
            route_mode = "local_only"
            route_action = "runtime_operational_status_local"
        elif is_route_mix_prompt:
            eligible = True
            reason = "runtime operational override: route-mix summary can be answered from local execution-trace evidence"
            route_mode = "local_only"
            route_action = "runtime_operational_status_local"
        elif is_logs_prompt:
            eligible = True
            reason = "runtime operational override: direct log lookup/navigation can be answered from local runtime-log evidence"
            route_mode = "local_only"
            route_action = "runtime_operational_status_local"
        else:
            eligible = False
            reason = "runtime operational debug defaults to external reasoning outside the proven local-only subset"
            route_mode = "external_reasoning"
            route_action = "runtime_operational_debug_external"

        return {
            "recognized_task_type": recognized_task_type,
            "eligible": eligible,
            "route_mode": route_mode,
            "route_action": route_action,
            "reason": reason,
            "confidence": confidence,
            "selected_core_evidence": chosen_core,
            "selected_support_evidence": chosen_support,
            "support_choice_debug": support_debug,
            "preferred_evidence_source": preferred_evidence_source,
            "override_used": operational_prompt,
            "external_escalation_reason": None if eligible and route_mode == "local_only" else reason,
        }

    def _estimate_external_escalation_cost(self, goal: str, context_pack: dict) -> float:
        prompt_chars = len(goal or "") + len(str(context_pack.get("project_summary") or ""))
        prompt_tokens_est = max(1, prompt_chars // 4)
        # Keep estimate aligned with telemetry's simple economics.
        return round((prompt_tokens_est / 1000.0) * 0.005 + 0.0001, 6)

    def _is_product_workflow_prompt(self, goal: str) -> bool:
        lowered = (goal or "").lower()
        return bool(re.search(r"\b(phase\s+\d+|product\s+api|codecontext|codecontext|plugin\s+runtime|route\s+mode|intercept\s+pipeline|backend\s+flow)\b", lowered))

    def _escalation_cost_discipline(self, goal: str, routed, context_pack: dict) -> dict:
        est_extra_cost = self._estimate_external_escalation_cost(goal, context_pack)
        is_product_workflow = self._is_product_workflow_prompt(goal)
        is_plugin_runtime = routed.intent in {"runtime_diagnostics", "code_understanding"} and any(t in (goal or "").lower() for t in ["codecontext", "codecontext", "plugin", "runtime", "intercept", "route mode"])

        allow = True
        reason = "escalation_allowed"
        blocked_by_cost_policy = False

        if est_extra_cost > self.config.max_escalation_cost_per_request:
            blocked_by_cost_policy = True
            allow = False
            reason = "blocked_high_estimated_escalation_cost"

        if allow and is_plugin_runtime and self.config.prefer_local_for_plugin_runtime:
            allow = False
            reason = "blocked_prefer_local_for_plugin_runtime"

        if allow and is_product_workflow and self.config.prefer_local_for_phase_workflow:
            allow = False
            reason = "blocked_prefer_local_for_phase_workflow"

        if not self.config.allow_external_for_explain_style and routed.task_type in {"explain_mechanism", "explain_architecture"}:
            allow = False
            reason = "blocked_external_disabled_for_explain_style"

        if not self.config.allow_external_for_product_workflow and is_product_workflow:
            allow = False
            reason = "blocked_external_disabled_for_product_workflow"

        return {
            "escalation_allowed": allow,
            "escalation_reason": reason,
            "escalation_blocked_by_cost_policy": blocked_by_cost_policy,
            "estimated_extra_cost_of_escalation": est_extra_cost,
        }

    def __init__(self, config: AppConfig):
        self.config = config
        self.summaries = SummaryManager(config)
        self.search = SearchEngine(config)
        self.context = ContextPackBuilder(config)
        self.patcher = Patcher(config)
        self.router = RequestRouter()
        self.responses = ResponseHandler(self.patcher, self.summaries)

    def scan_project(self) -> dict:
        return self.summaries.index_project()

    def index_project(self) -> dict:
        return self.summaries.index_project()

    def refresh_changed_files(self) -> dict:
        return self.summaries.refresh_changed_files()

    def search_project(self, query: str, top_k: int = 8) -> dict:
        return self.search.search_project(query, top_k=top_k)

    def summarize_file(self, path: str) -> dict:
        return self.summaries.summarize_file(path)

    def summarize_project(self) -> dict:
        return self.summaries.summarize_project()

    def prepare_context_pack(self, goal: str, top_k: int = 6, token_budget: int = 1800) -> dict:
        return self.context.prepare(goal, top_k=top_k, token_budget=token_budget)

    def apply_patch(self, path: str, old_text: str, new_text: str, dry_run: bool = False) -> dict:
        return self.patcher.apply_patch(path, old_text, new_text, dry_run=dry_run)

    def rollback_patch(self, patch_id: int) -> dict:
        return self.patcher.rollback_patch(patch_id)

    def route_request(
        self,
        goal: str,
        top_k: int = 6,
        token_budget: int = 1800,
        *,
        forced_route_mode: str | None = None,
        forced_intent: str | None = None,
        forced_task_type: str | None = None,
        forced_evidence_source_type: str | None = None,
        route_authority: str | None = None,
    ) -> dict:
        forced_route_mode = normalize_route_mode(forced_route_mode)
        if forced_route_mode:
            canonical_from_class = TS_INTERCEPT_TO_CANONICAL.get((forced_intent or "").strip())
            effective_intent = forced_intent or (canonical_from_class[0] if canonical_from_class else "code_understanding")
            effective_task_type = forced_task_type or (canonical_from_class[1] if canonical_from_class else "explain_mechanism")
            effective_evidence_source = forced_evidence_source_type or (canonical_from_class[2] if canonical_from_class else "code_mechanism_family")
            routed = RoutedRequest(
                user_goal=goal,
                intent=effective_intent,
                task_type=effective_task_type,
                evidence_source_type=effective_evidence_source,
                task_type_confidence=0.99,
                evidence_source_confidence=0.99,
                local_only_possible=forced_route_mode in {"local_only", "local_try_then_fallback"},
                external_reasoning_recommended=forced_route_mode in {"external_reasoning_with_compaction", "local_try_then_fallback"},
                should_refresh_first=effective_intent != "runtime_diagnostics",
            )
        else:
            # Fallback-only policy mode for direct CLI/backend usage.
            routed = self.router.classify(goal)
        refresh = self.summaries.refresh_changed_files() if routed.should_refresh_first else None

        forced_external = forced_route_mode == "external_reasoning_with_compaction"
        forced_local_only = forced_route_mode == "local_only"
        forced_local_try = forced_route_mode == "local_try_then_fallback"
        route_authority_effective = route_authority or ("ts_plugin" if forced_route_mode else "python_fallback_router")

        if routed.intent == "runtime_diagnostics" and not forced_external:
            diagnostics = runtime_diagnostics_summary(self.config.db_path, goal)
            empty_context_pack = {"debug": {}, "evidence": []}
            operational_decision = self._runtime_operational_debug_decision(goal, {}, empty_context_pack, routed)
            if routed.task_type == "inspect_logs":
                local_search = self.search.search_project(goal, top_k=top_k)
                context_pack = self.context.prepare(goal, top_k=top_k, token_budget=token_budget)
                evidence_count = context_pack["debug"].get("evidence_count", len(context_pack.get("evidence", [])))
                log_error_decision = self._log_error_local_only_decision(goal, local_search, context_pack, routed)
                if log_error_decision["eligible"]:
                    result = {
                        "mode": "local_only",
                        "route": asdict(routed),
                        "refresh": refresh,
                        "diagnostics": diagnostics,
                        "project_summary": context_pack["project_summary"],
                        "evidence": context_pack["evidence"],
                        "answer_strategy": "answer locally from sufficient runtime-log evidence for a simple inspect-logs request",
                        "debug": {
                            "route_action": "log_error_local_only",
                            "route_intent": routed.intent,
                            "task_type": routed.task_type,
                            "evidence_source_type": routed.evidence_source_type,
                            "task_type_confidence": routed.task_type_confidence,
                            "evidence_source_confidence": routed.evidence_source_confidence,
                            "evidence_count": evidence_count,
                            "fallback_used": context_pack["debug"].get("fallback_used", False),
                            "runtime_operational_override_used": operational_decision["override_used"],
                            "runtime_operational_preferred_evidence_source": operational_decision["preferred_evidence_source"],
                            "runtime_operational_local_only_reason": operational_decision["reason"] if operational_decision["route_mode"] == "local_only" else None,
                            "log_error_two_axis_override_used": log_error_decision["two_axis_override_used"],
                            "log_error_preferred_evidence_source_type": log_error_decision["preferred_evidence_source_type"],
                            "log_error_local_only_eligible": True,
                            "log_error_local_only_reason": log_error_decision["reason"],
                            "log_error_confidence": log_error_decision["confidence"],
                            "selected_core_evidence": log_error_decision["selected_core_evidence"],
                            "selected_support_evidence": log_error_decision["selected_support_evidence"],
                            "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                            "chosen_support_evidence": context_pack["debug"].get("chosen_support_evidence"),
                            "support_choice_debug": log_error_decision["support_choice_debug"],
                            "external_escalation_reason": None,
                        },
                    }
                    result["metrics"] = build_route_metrics(result)
                    return result
            result = {
                "mode": "local_only",
                "route": {**asdict(routed), "task_type": operational_decision["recognized_task_type"]} if operational_decision["override_used"] and operational_decision["route_mode"] == "local_only" else asdict(routed),
                "refresh": refresh,
                "diagnostics": diagnostics,
                "answer_strategy": "answer from telemetry, debug metadata, and run-state without generic repo retrieval",
                "debug": {
                    "route_action": operational_decision["route_action"] if operational_decision["override_used"] and operational_decision["route_mode"] == "local_only" else "runtime_diagnostics_local_only",
                    "task_type": operational_decision["recognized_task_type"] if operational_decision["override_used"] and operational_decision["route_mode"] == "local_only" else routed.task_type,
                    "evidence_source_type": operational_decision["preferred_evidence_source"] if operational_decision["override_used"] and operational_decision["route_mode"] == "local_only" else routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": 0,
                    "fallback_used": False,
                    "evidence_suppressed": True,
                    "suppression_reason": "runtime diagnostics should use telemetry/log/run-state evidence first",
                    "runtime_operational_override_used": operational_decision["override_used"],
                    "runtime_operational_preferred_evidence_source": operational_decision["preferred_evidence_source"],
                    "runtime_operational_local_only_reason": operational_decision["reason"] if operational_decision["route_mode"] == "local_only" else None,
                    "external_escalation_reason": None,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        local_search = self.search.search_project(goal, top_k=top_k)
        project_summary = self.summaries.summarize_project()

        if (forced_local_only or (routed.local_only_possible and not routed.external_reasoning_recommended)) and not forced_external:
            result = {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "project_summary": project_summary,
                "search": local_search,
                "answer_strategy": "answer from local cache/search without external model",
                "debug": {
                    "route_action": "local_only_direct",
                    "route_intent": routed.intent,
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        context_pack = self.context.prepare(goal, top_k=top_k, token_budget=token_budget)
        evidence_count = context_pack["debug"].get("evidence_count", len(context_pack.get("evidence", [])))
        operational_decision = self._runtime_operational_debug_decision(goal, local_search, context_pack, routed)
        explain_style_decision = self._explain_style_local_only_decision(goal, local_search, context_pack, routed)
        log_error_decision = self._log_error_local_only_decision(goal, local_search, context_pack, routed)
        exact_body_decision = self._exact_body_local_only_decision(goal, local_search, context_pack)

        if not forced_external and operational_decision["override_used"] and operational_decision["route_mode"] == "local_only":
            result = {
                "mode": "local_only",
                "route": {**asdict(routed), "task_type": operational_decision["recognized_task_type"]},
                "refresh": refresh,
                "project_summary": context_pack["project_summary"],
                "evidence": context_pack["evidence"],
                "answer_strategy": "answer locally from runtime process/trace/log evidence for an operational status request",
                "debug": {
                    "route_action": operational_decision["route_action"],
                    "route_intent": routed.intent,
                    "task_type": operational_decision["recognized_task_type"],
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": evidence_count,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "runtime_operational_override_used": operational_decision["override_used"],
                    "runtime_operational_preferred_evidence_source": operational_decision["preferred_evidence_source"],
                    "runtime_operational_local_only_reason": operational_decision["reason"],
                    "selected_core_evidence": operational_decision["selected_core_evidence"],
                    "selected_support_evidence": operational_decision["selected_support_evidence"],
                    "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                    "chosen_support_evidence": context_pack["debug"].get("chosen_support_evidence"),
                    "support_choice_debug": operational_decision["support_choice_debug"],
                    "external_escalation_reason": None,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        if not forced_external and explain_style_decision["eligible"]:
            result = {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "project_summary": context_pack["project_summary"],
                "evidence": context_pack["evidence"],
                "answer_strategy": "answer locally from strong core/support evidence for a simple explain-style request",
                "debug": {
                    "route_action": "explain_style_local_only",
                    "route_intent": routed.intent,
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": evidence_count,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "explain_style_two_axis_override_used": explain_style_decision["two_axis_override_used"],
                    "explain_style_preferred_evidence_source_type": explain_style_decision["preferred_evidence_source_type"],
                    "explain_style_local_only_eligible": True,
                    "explain_style_local_only_reason": explain_style_decision["reason"],
                    "explain_style_confidence": explain_style_decision["confidence"],
                    "selected_core_evidence": explain_style_decision["selected_core_evidence"],
                    "selected_support_evidence": explain_style_decision["selected_support_evidence"],
                    "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                    "chosen_support_evidence": context_pack["debug"].get("chosen_support_evidence"),
                    "support_choice_debug": explain_style_decision["support_choice_debug"],
                    "external_escalation_reason": None,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        if not forced_external and log_error_decision["eligible"]:
            result = {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "project_summary": context_pack["project_summary"],
                "evidence": context_pack["evidence"],
                "answer_strategy": "answer locally from sufficient log/traceback evidence for a simple log-error request",
                "debug": {
                    "route_action": "log_error_local_only",
                    "route_intent": routed.intent,
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": evidence_count,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "log_error_two_axis_override_used": log_error_decision["two_axis_override_used"],
                    "log_error_preferred_evidence_source_type": log_error_decision["preferred_evidence_source_type"],
                    "log_error_local_only_eligible": True,
                    "log_error_local_only_reason": log_error_decision["reason"],
                    "log_error_confidence": log_error_decision["confidence"],
                    "selected_core_evidence": log_error_decision["selected_core_evidence"],
                    "selected_support_evidence": log_error_decision["selected_support_evidence"],
                    "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                    "chosen_support_evidence": context_pack["debug"].get("chosen_support_evidence"),
                    "support_choice_debug": log_error_decision["support_choice_debug"],
                    "external_escalation_reason": None,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        if not forced_external and exact_body_decision["eligible"]:
            result = {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "exact_body_answer": {
                    "path": exact_body_decision["selected_body"]["path"],
                    "start_line": exact_body_decision["selected_body"]["start_line"],
                    "end_line": exact_body_decision["selected_body"]["end_line"],
                    "snippet_text": exact_body_decision["selected_body"]["snippet_text"],
                },
                "answer_strategy": "answer locally from one strong exact symbol-body match without external reasoning",
                "debug": {
                    "route_action": "exact_body_local_only",
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": evidence_count,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "exact_body_local_only_eligible": True,
                    "exact_body_local_only_reason": exact_body_decision["reason"],
                    "exact_body_confidence": exact_body_decision["confidence"],
                    "symbol_candidates_considered": exact_body_decision["symbol_candidates_considered"],
                    "exact_body_details": exact_body_decision["exact_body_details"],
                    "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                    "external_escalation_reason": None,
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        if evidence_count == 0 and routed.local_only_possible:
            return {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "project_summary": project_summary,
                "search": local_search,
                "answer_strategy": "downgraded to local_only because external_reasoning had zero evidence",
                "debug": {
                    "route_action": "downgraded_local_only",
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": 0,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "initial_empty_reason": context_pack["debug"].get("initial_empty_reason"),
                    "fallback_rule": context_pack["debug"].get("fallback_rule"),
                    "escalation_allowed": False,
                    "escalation_reason": "blocked_zero_evidence_local_possible",
                    "escalation_blocked_by_cost_policy": False,
                    "estimated_extra_cost_of_escalation": self._estimate_external_escalation_cost(goal, context_pack),
                    "local_retry_attempted": False,
                },
            }

        escalation = self._escalation_cost_discipline(goal, routed, context_pack)
        if not forced_external and not escalation["escalation_allowed"]:
            result = {
                "mode": "local_only",
                "route": asdict(routed),
                "refresh": refresh,
                "project_summary": context_pack["project_summary"],
                "evidence": context_pack.get("evidence", []),
                "search": local_search,
                "answer_strategy": "local_only enforced by escalation cost-discipline policy",
                "debug": {
                    "route_action": "escalation_blocked_local_only",
                    "task_type": routed.task_type,
                    "evidence_source_type": routed.evidence_source_type,
                    "task_type_confidence": routed.task_type_confidence,
                    "evidence_source_confidence": routed.evidence_source_confidence,
                    "evidence_count": evidence_count,
                    "fallback_used": context_pack["debug"].get("fallback_used", False),
                    "escalation_allowed": False,
                    "escalation_reason": escalation["escalation_reason"],
                    "escalation_blocked_by_cost_policy": escalation["escalation_blocked_by_cost_policy"],
                    "estimated_extra_cost_of_escalation": escalation["estimated_extra_cost_of_escalation"],
                    "local_retry_attempted": True,
                    "external_escalation_reason": escalation["escalation_reason"],
                },
            }
            result["metrics"] = build_route_metrics(result)
            return result

        if evidence_count == 0 and not routed.local_only_possible:
            route_action = "external_reasoning_without_evidence_justified"
        else:
            route_action = "external_reasoning_with_evidence" if evidence_count > 0 else "downgraded_local_only"

        outbound = build_outbound_request(
            goal=goal,
            intent=routed.intent,
            token_budget=token_budget,
            project_summary=context_pack["project_summary"],
            evidence=context_pack["evidence"],
            policy=context_pack["policy"],
            response_format=self.responses.patch_format_instructions(),
            debug={
                **context_pack["debug"],
                "route_action": operational_decision["route_action"] if operational_decision["override_used"] and operational_decision["route_mode"] == "external_reasoning" else route_action,
                "task_type": operational_decision["recognized_task_type"] if operational_decision["override_used"] else routed.task_type,
                "evidence_source_type": routed.evidence_source_type,
                "task_type_confidence": routed.task_type_confidence,
                "evidence_source_confidence": routed.evidence_source_confidence,
                "route_authority": route_authority_effective,
                "forced_route_mode": forced_route_mode,
            },
        )
        result = {
            "mode": "external_reasoning",
            "route": {**asdict(routed), "task_type": operational_decision["recognized_task_type"]} if operational_decision["override_used"] and operational_decision["route_mode"] == "external_reasoning" else asdict(routed),
            "refresh": refresh,
            "outbound_payload": outbound,
            "search_preview": local_search,
            "debug": {
                "route_intent": routed.intent,
                "task_type": routed.task_type,
                "evidence_source_type": routed.evidence_source_type,
                "task_type_confidence": routed.task_type_confidence,
                "evidence_source_confidence": routed.evidence_source_confidence,
                "evidence_count": evidence_count,
                "evidence_count_before_prune": context_pack["debug"].get("evidence_count_before_prune", evidence_count),
                "evidence_count_after_prune": context_pack["debug"].get("evidence_count_after_prune", evidence_count),
                "compaction_chars_saved": context_pack["debug"].get("compaction_chars_saved", 0),
                "fallback_used": context_pack["debug"].get("fallback_used", False),
                "initial_empty_reason": context_pack["debug"].get("initial_empty_reason"),
                "fallback_rule": context_pack["debug"].get("fallback_rule"),
                "route_action": operational_decision["route_action"] if operational_decision["override_used"] and operational_decision["route_mode"] == "external_reasoning" else route_action,
                "runtime_operational_override_used": operational_decision["override_used"],
                "runtime_operational_preferred_evidence_source": operational_decision["preferred_evidence_source"],
                "runtime_operational_local_only_reason": operational_decision["reason"] if operational_decision["route_mode"] == "local_only" else None,
                "support_slot_reserved": context_pack["debug"].get("support_slot_reserved", False),
                "chosen_support_survived": context_pack["debug"].get("chosen_support_survived", False),
                "support_retention_reason": context_pack["debug"].get("support_retention_reason"),
                "support_retention_exception_used": context_pack["debug"].get("support_retention_exception_used", False),
                "retained_items": context_pack["debug"].get("retained_items", []),
                "pruned_items": context_pack["debug"].get("pruned_items", []),
                "chosen_core_evidence": context_pack["debug"].get("chosen_core_evidence"),
                "chosen_support_evidence": context_pack["debug"].get("chosen_support_evidence"),
                "support_choice_debug": context_pack["debug"].get("support_choice_debug", []),
                "explain_style_two_axis_override_used": explain_style_decision["two_axis_override_used"],
                "explain_style_preferred_evidence_source_type": explain_style_decision["preferred_evidence_source_type"],
                "explain_style_local_only_eligible": explain_style_decision["eligible"],
                "explain_style_local_only_reason": explain_style_decision["reason"],
                "explain_style_confidence": explain_style_decision["confidence"],
                "log_error_two_axis_override_used": log_error_decision["two_axis_override_used"],
                "log_error_preferred_evidence_source_type": log_error_decision["preferred_evidence_source_type"],
                "log_error_local_only_eligible": log_error_decision["eligible"],
                "log_error_local_only_reason": log_error_decision["reason"],
                "log_error_confidence": log_error_decision["confidence"],
                "selected_core_evidence": log_error_decision["selected_core_evidence"] or explain_style_decision["selected_core_evidence"],
                "selected_support_evidence": log_error_decision["selected_support_evidence"] or explain_style_decision["selected_support_evidence"],
                "exact_body_local_only_eligible": exact_body_decision["eligible"],
                "exact_body_local_only_reason": exact_body_decision["reason"],
                "exact_body_confidence": exact_body_decision["confidence"],
                "symbol_candidates_considered": exact_body_decision["symbol_candidates_considered"],
                "exact_body_details": context_pack["debug"].get("exact_body_details"),
                "escalation_allowed": escalation["escalation_allowed"],
                "escalation_reason": escalation["escalation_reason"],
                "escalation_blocked_by_cost_policy": escalation["escalation_blocked_by_cost_policy"],
                "estimated_extra_cost_of_escalation": escalation["estimated_extra_cost_of_escalation"],
                "local_retry_attempted": False,
                "external_escalation_reason": (
                    operational_decision["external_escalation_reason"] if operational_decision["override_used"] and operational_decision["route_mode"] == "external_reasoning" else
                    explain_style_decision["external_escalation_reason"] if routed.intent == "code_understanding" else
                    log_error_decision["external_escalation_reason"] if routed.intent == "bug_hunt" else
                    exact_body_decision["external_escalation_reason"] if routed.task_type == "retrieve_exact" else
                    explain_style_decision["external_escalation_reason"] or log_error_decision["external_escalation_reason"] or exact_body_decision["external_escalation_reason"]
                ),
                "route_authority": route_authority_effective,
                "forced_route_mode": forced_route_mode,
                "forced_local_try": forced_local_try,
            },
        }
        result["metrics"] = build_route_metrics(result)
        return result

    def handle_remote_response(
        self,
        response_text: str,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        dry_run: bool = False,
        top_k: int = 6,
        token_budget: int = 1800,
    ) -> dict:
        decision = self.responses.classify_response(response_text)
        if decision.payload is not None:
            result = self.responses.handle_structured_payload(decision.payload)
            if result.get("kind") == "needs_more_context":
                followup_goal = result.get("request") or "Provide narrower context"
                followup = self.route_request(followup_goal, top_k=top_k, token_budget=token_budget)
                result["followup"] = followup
            result["decision"] = asdict(decision)
            return result
        if decision.kind == "patch_instruction":
            if not path or old_text is None or new_text is None:
                return {
                    "kind": "patch_instruction_detected_but_missing_fields",
                    "decision": asdict(decision),
                    "response_text": response_text,
                }
            return self.responses.handle_patch_payload(path, old_text, new_text, dry_run=dry_run)
        if decision.kind == "needs_more_context":
            followup = self.route_request(response_text, top_k=top_k, token_budget=token_budget)
            return {
                "kind": "needs_more_context",
                "decision": asdict(decision),
                "next_step": "ran local retrieval again with narrower follow-up query",
                "followup": followup,
            }
        return {
            "kind": "answer",
            "decision": asdict(decision),
            "response_text": response_text,
        }
