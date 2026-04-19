# Local-First Product API (Phase 4)

Minimal backend API surface for demos/pilots.

## What this is
A thin HTTP wrapper over the existing local-first engine that exposes health, metrics, sales summary, config, and route-or-run execution.

## Why it saves money
- routes irrelevant prompts to pass-through
- keeps many relevant prompts local-only
- compacts context for external reasoning
- tracks estimated tokens/cost/savings in telemetry

## Start API
```powershell
python main.py serve-api --root . --host 127.0.0.1 --port 8787
```

## Endpoints

### GET /health
Returns service/db readiness and timestamp.

### GET /metrics
Returns corrected machine-readable metrics report.

Query params:
- `window` (e.g. `1d`, `7d`)
- `route_filter`
- `top_n`
- `definitions` (`true|false`)

Example:
```powershell
curl "http://127.0.0.1:8787/metrics?window=7d&top_n=5&definitions=true"
```

### GET /sales-summary
Returns concise summary string + machine summary payload.

Example:
```powershell
curl "http://127.0.0.1:8787/sales-summary?window=7d"
```

### GET /config
Safe product/demo config view (no secrets).

### POST /route-or-run
Body:
```json
{
  "prompt": "check logs for the local-first runtime",
  "session_key": "optional",
  "options": {
    "dry_run": true,
    "top_k": 4,
    "token_budget": 1200
  }
}
```

Response fields include:
- `cleaned_prompt`
- `intercept_class`
- `candidate_relevant`
- `chosen_route`
- `class_reason`
- `relevance_reason`
- `route_reason`
- `cost_estimate`
- `cost_threshold_hit`
- `cost_decision_reason`
- `estimated_savings_vs_external`
- `trace_id`
- `run_id` (when execution occurs)

## Known limitations
- no auth/tenant isolation (local demo mode)
- cost estimates are approximate
- route metadata is based on existing local_first route payloads

## Production-readiness checklist
- [x] structured logging
- [x] DB-backed telemetry
- [x] schema migrations additive
- [x] timeout/fallback handling
- [x] safe error responses (no stack traces)
- [x] no secret leakage in `/config`
- [ ] auth / RBAC
- [ ] multi-tenant partitioning
- [ ] hard rate limits
