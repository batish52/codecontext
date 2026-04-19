from __future__ import annotations

from dataclasses import dataclass


LOCAL_ONLY_HINTS = {
    "rollback", "diff", "scan", "index", "manifest",
    "list", "callers", "callees", "dependents", "dependencies", "impact",
    "blame", "grep",
}

EDIT_HINTS = {"patch", "edit", "change", "update", "modify", "rewrite", "refactor", "fix"}

REASONING_HINTS = {
    "why", "design", "architecture", "plan", "strategy", "compare", "decide",
    "best", "tradeoff", "reason", "ambiguous", "synthesize", "explain",
    "implement", "build", "create", "write", "generate", "suggest",
    "improve", "optimize", "recommend", "review", "analyze", "think",
    "help", "should", "could", "would",
    "properly", "correctly", "better", "alternative",
}

DIAGNOSTIC_HINTS = {
    "telemetry", "metrics", "metric", "logs", "log", "debug", "runtime", "run", "state",
    "fallback", "external", "local", "working", "status", "plugin", "intercept", "interceptor",
}

CONFIG_HINTS = {"config", "setting", "settings", "env", "manifest", "package", "json", ".env", "toml", "yaml", "yml"}
NAVIGATION_HINTS = {"repo", "project", "tree", "structure", "layout", "files", "folders", "folder", "directory", "directories", "navigate"}
TRIAGE_HINTS = {"error", "exception", "traceback", "bug", "issue", "failure", "failing", "broken", "stacktrace", "stack", "log", "logs"}


@dataclass(slots=True)
class RoutedRequest:
    user_goal: str
    intent: str
    task_type: str
    evidence_source_type: str
    task_type_confidence: float
    evidence_source_confidence: float
    local_only_possible: bool
    external_reasoning_recommended: bool
    should_refresh_first: bool


class RequestRouter:
    """Fallback-only router for direct backend/CLI usage.

    Accepts forced routing overrides from upstream integrations.
    """
    def _classify_two_axis(self, lowered: str, terms: set[str], *, diagnostic_request: bool, config_request: bool, navigation_request: bool, triage_request: bool, simple_edit_request: bool, explain_request: bool) -> tuple[str, float, str, float]:
        exact_request = bool({"exact", "body", "defined", "definition"} & terms)
        architecture_request = bool({"architecture", "tradeoff", "trade-off", "design"} & terms)
        root_cause_request = bool({"why", "debug", "bug", "root", "cause", "traceback", "exception"} & terms)
        logs_request = bool({"logs", "log", "traceback", "stacktrace", "stack"} & terms)
        compare_refactor_request = bool({"compare", "comparison", "refactor", "rewrite"} & terms)
        mechanism_request = explain_request and not architecture_request and not root_cause_request and not compare_refactor_request
        runtime_operational_request = bool({"run_live_engine", "live", "engine", "stderr", "stdout", "process"} & terms)

        if exact_request:
            task_type, task_conf = "retrieve_exact", 0.97
        elif diagnostic_request and logs_request:
            task_type, task_conf = "inspect_logs", 0.95
        elif diagnostic_request:
            task_type, task_conf = "diagnose_runtime", 0.96
        elif triage_request and root_cause_request:
            task_type, task_conf = "debug_root_cause", 0.92
        elif simple_edit_request:
            task_type, task_conf = "edit_code", 0.94
        elif compare_refactor_request:
            task_type, task_conf = "compare_or_refactor", 0.91
        elif architecture_request:
            task_type, task_conf = "explain_architecture", 0.92
        elif mechanism_request or runtime_operational_request:
            task_type, task_conf = "explain_mechanism", 0.86 if runtime_operational_request else 0.93
        else:
            task_type, task_conf = "explain_mechanism", 0.62

        if task_type == "retrieve_exact":
            evidence_type, evidence_conf = "symbol_body", 0.97
        elif task_type == "diagnose_runtime":
            evidence_type, evidence_conf = "telemetry_state", 0.96
        elif task_type == "inspect_logs":
            evidence_type, evidence_conf = "runtime_logs", 0.95
        elif task_type == "debug_root_cause":
            evidence_type, evidence_conf = ("traceback", 0.94) if logs_request else ("cross_module_design", 0.76)
        elif config_request:
            evidence_type, evidence_conf = "config_manifest", 0.95
        elif navigation_request:
            evidence_type, evidence_conf = "repo_structure", 0.93
        elif task_type == "explain_architecture":
            evidence_type, evidence_conf = "cross_module_design", 0.9
        elif task_type == "explain_mechanism":
            evidence_type, evidence_conf = "code_mechanism_family", 0.9
        else:
            evidence_type, evidence_conf = "code_mechanism_family", 0.7

        return task_type, task_conf, evidence_type, evidence_conf

    def classify(self, goal: str) -> RoutedRequest:
        force_external = False
        stripped_goal = goal.lstrip()
        for prefix in ("!llm", "@llm"):
            if stripped_goal.lower().startswith(prefix):
                stripped_goal = stripped_goal[len(prefix):].lstrip()
                force_external = True
                break

        goal = stripped_goal or goal
        lowered = goal.lower()
        terms = set(lowered.replace("/", " ").replace("-", " ").replace("?", " ").replace(":", " ").replace("(", " ").replace(")", " ").split())
        reasoning_phrases = {"how to", "what if"}
        has_reasoning_phrase = any(phrase in lowered for phrase in reasoning_phrases)
        config_request = bool(terms & CONFIG_HINTS)
        navigation_request = bool(terms & NAVIGATION_HINTS)
        triage_request = bool(terms & TRIAGE_HINTS)
        simple_edit_request = any(term in EDIT_HINTS for term in terms)
        diagnostic_request = (
            bool({"fallback", "telemetry", "metrics", "metric", "debug", "status", "working"} & terms)
            or ({"local", "external"} <= terms and ("how" in terms or "much" in terms))
            or (("check" in terms or "show" in terms) and bool({"logs", "log", "run", "state"} & terms) and not config_request and not navigation_request and not simple_edit_request)
            or (("plugin" in terms or "runtime" in terms) and bool({"working", "status", "telemetry", "fallback", "debug"} & terms))
            or (bool({"intercept", "interceptor"} & terms) and bool({"working", "status", "telemetry", "fallback", "debug"} & terms))
        ) and not bool(terms & REASONING_HINTS) and not has_reasoning_phrase

        explain_request = bool(terms & {"explain", "how", "why", "overview", "architecture", "walk", "through"}) and not bool(terms & {"exact", "body", "defined", "definition"})

        if diagnostic_request:
            intent = "runtime_diagnostics"
        elif simple_edit_request:
            intent = "code_edit"
        elif config_request:
            intent = "config_lookup"
        elif triage_request:
            intent = "bug_hunt"
        elif explain_request:
            intent = "code_understanding"
        elif navigation_request or any(term in {"summary", "summarize", "overview", "architecture"} for term in terms):
            intent = "project_summary"
        elif any(term in {"function", "class", "method", "symbol"} for term in terms):
            intent = "symbol_lookup"
        else:
            intent = "code_understanding"

        task_type, task_type_confidence, evidence_source_type, evidence_source_confidence = self._classify_two_axis(
            lowered,
            terms,
            diagnostic_request=diagnostic_request,
            config_request=config_request,
            navigation_request=navigation_request,
            triage_request=triage_request,
            simple_edit_request=simple_edit_request,
            explain_request=explain_request,
        )

        local_only = diagnostic_request or (
            bool(terms & LOCAL_ONLY_HINTS)
            and not bool(terms & REASONING_HINTS)
            and not has_reasoning_phrase
            and len(terms) <= 8
        )
        if config_request and not bool(terms & REASONING_HINTS):
            local_only = True
        if navigation_request and bool({"show", "list", "find", "where", "files", "folders", "structure", "tree"} & terms):
            local_only = True

        external_needed = False if diagnostic_request else (
            bool(terms & REASONING_HINTS) or has_reasoning_phrase or intent in {"code_edit", "bug_hunt"}
        )

        # LOCAL_ONLY_HINTS match takes priority — these are definitively local operations
        if local_only and bool(terms & LOCAL_ONLY_HINTS):
            external_needed = False

        if intent in {"symbol_lookup", "config_lookup", "project_summary", "runtime_diagnostics"} and local_only:
            external_needed = False
        if {"where", "defined"} & terms:
            local_only = True
            external_needed = False

        if force_external:
            local_only = False
            external_needed = True

        return RoutedRequest(
            user_goal=goal,
            intent=intent,
            task_type=task_type,
            evidence_source_type=evidence_source_type,
            task_type_confidence=task_type_confidence,
            evidence_source_confidence=evidence_source_confidence,
            local_only_possible=local_only,
            external_reasoning_recommended=external_needed,
            should_refresh_first=not diagnostic_request,
        )
