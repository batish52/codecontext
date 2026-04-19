from pathlib import Path

from codecontext.config import AppConfig
from codecontext.gateway import CodeContextGateway


def test_forced_external_route_bypasses_python_local_routing():
    cfg = AppConfig(root=Path('.'))
    bridge = CodeContextGateway(cfg)

    result = bridge.route_request(
        'check logs',
        forced_route_mode='external_reasoning_with_compaction',
        forced_intent='runtime_diagnostics',
        forced_task_type='runtime_diagnostics',
        forced_evidence_source_type='runtime_diagnostics',
        route_authority='ts_plugin',
    )

    assert result['mode'] == 'external_reasoning'
    assert (result.get('debug') or {}).get('route_authority') == 'ts_plugin'


def test_without_forced_route_uses_python_router_default():
    cfg = AppConfig(root=Path('.'))
    bridge = CodeContextGateway(cfg)

    result = bridge.route_request('check logs')
    assert result['mode'] in {'local_only', 'external_reasoning'}
