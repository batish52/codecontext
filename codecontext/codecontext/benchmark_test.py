from __future__ import annotations

import json
import tempfile
from pathlib import Path

from codecontext.benchmark import _aggregate, run_benchmark, benchmark_sales_summary, benchmark_leakage_audit, DATASET_DIR


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # create tiny dataset
        ds = DATASET_DIR / "unit_test_dataset.json"
        ds.parent.mkdir(parents=True, exist_ok=True)
        ds.write_text(json.dumps([
            {"prompt": "check logs for the codecontext runtime", "type": "plugin_runtime"},
            {"prompt": "what is the weather tomorrow", "type": "irrelevant"},
            {"prompt": "explain architecture tradeoffs across microservices", "type": "explain_style"}
        ]), encoding="utf-8")

        def fake_eval(prompt: str):
            p = prompt.lower()
            if "check logs" in p:
                return {"result": {"mode": "local_only", "debug": {"escalation_allowed": False, "escalation_reason": "blocked_prefer_local_for_plugin_runtime", "escalation_blocked_by_cost_policy": False, "estimated_extra_cost_of_escalation": 0.004, "local_retry_attempted": True}}, "metrics": {"estimated_tokens": 100, "estimated_cost": 0.001, "estimated_savings_vs_external": 0.01}}
            return {"result": {"mode": "external_reasoning", "debug": {"escalation_allowed": True, "escalation_reason": "escalation_allowed", "escalation_blocked_by_cost_policy": False, "estimated_extra_cost_of_escalation": 0.01, "local_retry_attempted": False}}, "metrics": {"estimated_tokens": 600, "estimated_cost": 0.02, "estimated_savings_vs_external": 0.0}}

        out = run_benchmark(root, dataset="unit_test_dataset", runs=1, evaluator=fake_eval)
        assert out["dataset"] == "unit_test_dataset"
        assert len(out["records"]) == 3

        agg = _aggregate(out["records"])
        assert agg["total_requests"] == 3
        assert agg["route_counts"]["pass_through_direct"] == 1
        assert agg["route_counts"]["local_only"] == 1
        assert agg["route_counts"]["external_reasoning_with_compaction"] == 1
        assert agg["estimated_savings_absolute"] >= 0

        sales = benchmark_sales_summary("unit_test_dataset")
        assert "sales_summary" in sales
        assert "handled locally" in sales["sales_summary"]

        leak = benchmark_leakage_audit("unit_test_dataset")
        assert "suspicious_escalations" in leak
        assert "high_cost_outliers" in leak

    print("benchmark_test.py passed")


if __name__ == "__main__":
    run_tests()
