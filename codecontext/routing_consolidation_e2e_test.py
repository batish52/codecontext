from pathlib import Path

from codecontext.config import AppConfig
from codecontext.gateway import CodeContextGateway


def _mk_bridge() -> CodeContextGateway:
    cfg = AppConfig(root=Path('.'))
    return CodeContextGateway(cfg)


def test_intercepted_local_route_forced_local_only():
    bridge = _mk_bridge()
    out = bridge.route_request(
        'check logs',
        forced_route_mode='local_only',
        forced_intent='runtime_diagnostics',
        forced_task_type='diagnose_runtime',
        forced_evidence_source_type='telemetry_state',
        route_authority='ts_plugin',
    )
    assert out['mode'] == 'local_only'


def test_intercepted_external_route_forced_external():
    bridge = _mk_bridge()
    out = bridge.route_request(
        'check logs',
        forced_route_mode='external_reasoning_with_compaction',
        forced_intent='runtime_diagnostics',
        forced_task_type='diagnose_runtime',
        forced_evidence_source_type='telemetry_state',
        route_authority='ts_plugin',
    )
    assert out['mode'] == 'external_reasoning'
    debug = (out.get('debug') or {})
    assert debug.get('route_authority') == 'ts_plugin'


def test_intercepted_local_try_route_contract():
    bridge = _mk_bridge()
    out = bridge.route_request(
        'explain architecture tradeoffs in routing',
        forced_route_mode='local_try_then_fallback',
        forced_intent='code_understanding',
        forced_task_type='explain_architecture',
        forced_evidence_source_type='cross_module_design',
        route_authority='ts_plugin',
    )
    assert out['mode'] in {'local_only', 'external_reasoning'}
    debug = (out.get('debug') or {})
    if out['mode'] == 'external_reasoning':
        assert debug.get('forced_local_try') is True


def test_direct_cli_backend_fallback_route_uses_python_policy():
    bridge = _mk_bridge()
    out = bridge.route_request('explain architecture tradeoffs in routing')
    assert out['mode'] in {'local_only', 'external_reasoning'}
    if out['mode'] == 'external_reasoning':
        debug = (out.get('debug') or {})
        assert debug.get('route_authority') == 'python_fallback_router'


def test_telemetry_route_authority_visible_in_result_debug():
    bridge = _mk_bridge()
    forced = bridge.route_request(
        'check logs',
        forced_route_mode='external_reasoning_with_compaction',
        forced_intent='runtime_diagnostics',
        forced_task_type='diagnose_runtime',
        forced_evidence_source_type='telemetry_state',
        route_authority='ts_plugin',
    )
    f_debug = (forced.get('debug') or {})
    assert f_debug.get('route_authority') == 'ts_plugin'
