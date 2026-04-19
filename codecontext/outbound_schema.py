from __future__ import annotations


def build_outbound_request(
    *,
    goal: str,
    intent: str,
    token_budget: int,
    project_summary: dict,
    evidence: list[dict],
    policy: dict,
    response_format: dict,
    debug: dict,
) -> dict:
    return {
        "schema_version": "codecontext.outbound.v1",
        "request": {
            "goal": goal,
            "intent": intent,
            "task_type": debug.get("task_type"),
            "evidence_source_type": debug.get("evidence_source_type"),
            "task_type_confidence": debug.get("task_type_confidence"),
            "evidence_source_confidence": debug.get("evidence_source_confidence"),
            "token_budget": token_budget,
        },
        "repository_context": {
            "project_summary": project_summary,
            "evidence": evidence,
        },
        "constraints": {
            "use_only_provided_context": True,
            "avoid_full_files": True,
            "ask_for_more_context_if_needed": True,
            "policy": policy,
        },
        "debug": {
            "evidence_count": debug.get("evidence_count", len(evidence)),
            "fallback_used": debug.get("fallback_used", False),
            "initial_empty_reason": debug.get("initial_empty_reason"),
            "fallback_rule": debug.get("fallback_rule"),
            "route_action": debug.get("route_action", "external_reasoning"),
            "task_type": debug.get("task_type"),
            "evidence_source_type": debug.get("evidence_source_type"),
            "task_type_confidence": debug.get("task_type_confidence"),
            "evidence_source_confidence": debug.get("evidence_source_confidence"),
            "chosen_core_evidence": debug.get("chosen_core_evidence"),
            "chosen_support_evidence": debug.get("chosen_support_evidence"),
            "support_choice_debug": debug.get("support_choice_debug", []),
            "exact_body_debug": debug.get("exact_body_debug"),
            "exact_body_details": debug.get("exact_body_details"),
        },
        "expected_response": response_format,
    }
