from __future__ import annotations

from typing import Literal

CanonicalRouteMode = Literal[
    "pass_through_direct",
    "local_only",
    "external_reasoning_with_compaction",
    "local_try_then_fallback",
]

CanonicalIntent = Literal[
    "runtime_diagnostics",
    "code_understanding",
    "code_edit",
    "config_lookup",
    "project_summary",
    "bug_hunt",
    "symbol_lookup",
    "code_review",
    "metrics_analysis",
]

# Bug #19: keep this Literal in lockstep with `task_type` values that
# `RequestRouter.classify()` actually emits. Previously the router
# assigned `"debug_root_cause"` and `"compare_or_refactor"` but neither
# was listed here, so any type-narrowing consumer would silently miss
# them. The full set is the union of router emissions and the older
# canonical names that downstream consumers branch on.
CanonicalTaskType = Literal[
    "diagnose_runtime",
    "inspect_logs",
    "explain_mechanism",
    "explain_architecture",
    "edit_code",
    "retrieve_exact",
    "review_engine",
    "analyze_metrics",
    "lookup_config",
    "navigate_repo",
    "debug_root_cause",
    "compare_or_refactor",
]

CanonicalEvidenceSourceType = Literal[
    "telemetry_state",
    "runtime_logs",
    "code_mechanism_family",
    "cross_module_design",
    "traceback",
    "symbol_body",
    "config_manifest",
    "repo_structure",
]

# Bug #17: include both the legacy "symbol_or_exact_body" boundary
# class name AND the normalized "exact_body" form that
# normalize_boundary_class rewrites it to. Prior to this fix, downstream
# code that received an already-normalized class would get a None back
# from this dict because only the legacy key was present.
TS_INTERCEPT_TO_CANONICAL: dict[str, tuple[CanonicalIntent, CanonicalTaskType, CanonicalEvidenceSourceType]] = {
    "runtime_diagnostics": ("runtime_diagnostics", "diagnose_runtime", "telemetry_state"),
    "plugin_runtime_inspection": ("runtime_diagnostics", "inspect_logs", "runtime_logs"),
    "config_manifest_lookup": ("config_lookup", "lookup_config", "config_manifest"),
    "repo_navigation": ("project_summary", "navigate_repo", "repo_structure"),
    "simple_edit_intent": ("code_edit", "edit_code", "code_mechanism_family"),
    "log_error_triage": ("bug_hunt", "inspect_logs", "traceback"),
    "code_review_or_engine_audit": ("code_review", "review_engine", "cross_module_design"),
    "product_metrics_or_cost_analysis": ("metrics_analysis", "analyze_metrics", "telemetry_state"),
    "symbol_or_exact_body": ("symbol_lookup", "retrieve_exact", "symbol_body"),
    "exact_body": ("symbol_lookup", "retrieve_exact", "symbol_body"),
    "explain_style": ("code_understanding", "explain_mechanism", "code_mechanism_family"),
    "existing_relevant": ("code_understanding", "explain_mechanism", "code_mechanism_family"),
}


def normalize_route_mode(v: str | None) -> CanonicalRouteMode | None:
    if v in {"pass_through_direct", "local_only", "external_reasoning_with_compaction", "local_try_then_fallback"}:
        return v  # type: ignore[return-value]
    return None
