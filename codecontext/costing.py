from __future__ import annotations

from .utils import approx_tokens


DEFAULT_INPUT_COST_PER_1K = 0.005


def estimate_text_tokens(text: str) -> int:
    return approx_tokens(text)


def estimate_evidence_tokens(evidence: list[dict]) -> int:
    return sum(approx_tokens(item.get("snippet_text", "")) for item in evidence)


def estimate_project_summary_tokens(project_summary: dict) -> int:
    return approx_tokens(str(project_summary))


def estimate_request_cost_usd(tokens: int, cost_per_1k: float = DEFAULT_INPUT_COST_PER_1K) -> float:
    return round((tokens / 1000.0) * cost_per_1k, 6)
