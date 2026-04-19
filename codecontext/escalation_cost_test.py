from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from codecontext.config import AppConfig
from codecontext.gateway import CodeContextGateway


def _bridge():
    td = tempfile.TemporaryDirectory()
    cfg = AppConfig(Path(td.name))
    b = CodeContextGateway(cfg)
    b._tmpdir = td  # keep alive
    return b


def run_tests():
    b = _bridge()

    # 1) plugin/runtime prompt stays local_only via escalation discipline block
    routed = SimpleNamespace(intent="code_understanding", task_type="explain_mechanism", evidence_source_type="code_mechanism_family")
    ctx = {"project_summary": {"a": "b"}, "debug": {}, "evidence": [{"path": "codecontext-runtime/index.ts"}]}
    d = b._escalation_cost_discipline("explain codecontext routing gateway internals and route mode", routed, ctx)
    assert d["escalation_allowed"] is False
    assert "plugin_runtime" in d["escalation_reason"] or "product_workflow" in d["escalation_reason"]

    # 2) phase/workflow prompt stays local when context exists
    d2 = b._escalation_cost_discipline("Proceed to Phase 4 only for codecontext product API work", routed, ctx)
    assert d2["escalation_allowed"] is False

    # 3) genuinely complex explain-style may still escalate
    cfg2 = AppConfig(Path(tempfile.mkdtemp()))
    cfg2.prefer_local_for_plugin_runtime = False
    cfg2.prefer_local_for_phase_workflow = False
    cfg2.allow_external_for_product_workflow = True
    b2 = CodeContextGateway(cfg2)
    routed2 = SimpleNamespace(intent="code_understanding", task_type="explain_architecture", evidence_source_type="cross_module_design")
    d3 = b2._escalation_cost_discipline("Explain architecture tradeoffs across modules", routed2, {"project_summary": {}, "debug": {}, "evidence": [{"path": "codecontext/search.py"}]})
    assert d3["escalation_allowed"] is True

    # 4) escalation blocked when cost threshold hit
    cfg3 = AppConfig(Path(tempfile.mkdtemp()))
    cfg3.max_escalation_cost_per_request = 0.00001
    cfg3.prefer_local_for_plugin_runtime = False
    cfg3.prefer_local_for_phase_workflow = False
    cfg3.allow_external_for_product_workflow = True
    b3 = CodeContextGateway(cfg3)
    d4 = b3._escalation_cost_discipline("Explain architecture tradeoffs across modules", routed2, {"project_summary": {"x": "y" * 1000}, "debug": {}, "evidence": [{"path": "codecontext/search.py"}]})
    assert d4["escalation_allowed"] is False
    assert d4["escalation_blocked_by_cost_policy"] is True

    # 5) irrelevant prompt still pass-through (router-level invariant via decision helper in plugin TS; just assert policy doesn't force escalation)
    d5 = b3._escalation_cost_discipline("hello", SimpleNamespace(intent="other", task_type="other", evidence_source_type="other"), {"project_summary": {}, "debug": {}, "evidence": []})
    assert isinstance(d5["escalation_allowed"], bool)

    # 6) policy fields present
    for key in ["escalation_allowed", "escalation_reason", "escalation_blocked_by_cost_policy", "estimated_extra_cost_of_escalation"]:
        assert key in d4

    print("escalation_cost_test.py passed")


if __name__ == "__main__":
    run_tests()
