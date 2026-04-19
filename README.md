# CodeContext

**Reduce your LLM costs by 40–70% automatically.**

CodeContext is a drop-in gateway that sits between your code and the LLM API. It routes trivial prompts to a local model (or skips the call entirely), compacts the context it does send, and tracks exactly how much money it saved you — per request, per day, per model. Typical savings of 40–70% depend on workload mix; the built-in benchmark harness reports the number for your workload specifically.

```
your app  ─────►  CodeContext gateway  ─────►  OpenAI / Anthropic / Ollama / any OpenAI-compatible endpoint
                       │
                       ├── routes trivial prompts locally            (0 external tokens)
                       ├── strips irrelevant files from context     (−40–80% input tokens)
                       ├── caps per-request cost                     (hard spend rail)
                       └── logs real tokens + real $ to SQLite       (auditable savings)
```

---

## Why it exists

If you're paying OpenAI or Anthropic for code-related work, three things are usually true:

1. **You're sending too much context.** Most requests don't need the whole repo — they need 3–5 files. CodeContext ranks chunks with BM25 + embeddings and packs only what matters into a token budget you set.
2. **Many prompts don't need a frontier model.** "Format this dict" doesn't need GPT-4. The gateway classifies every request and routes cheap ones to a local model or skips the external call entirely.
3. **You don't actually know what you're spending.** Provider dashboards are lagged and aggregate. CodeContext writes every call to a local SQLite ledger with real token counts and real dollar cost.

It's built for engineers shipping AI-assisted tools, not for dashboards people.

---

## Install

```bash
pip install promptrouter
```

Optional extras:

```bash
pip install promptrouter[embeddings]   # semantic retrieval (sentence-transformers)
pip install promptrouter[tokens]       # precise token counts (tiktoken)
pip install promptrouter[all]          # both of the above
```

CodeContext itself has **one** required dependency (`pathspec`). Everything else is lazy-loaded and optional. The gateway speaks to OpenAI and Anthropic over plain HTTP — no SDK install needed.

Requires Python 3.10+.

---

## Quickstart (60 seconds)

```bash
# 1. From the root of the project you want to index:
cd ~/code/my-project

# 2. Drop in a config (see example at .codecontext.toml in this repo):
cat > .codecontext.toml <<'EOF'
[llm_client]
enabled     = true
provider    = "openai"
model       = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
EOF

# 3. Set your API key:
export OPENAI_API_KEY="sk-..."

# 4. Index the project:
codecontext index-project --root .

# 5. Ask a question. CodeContext builds a minimal context pack, routes the
#    request, calls the model, and logs the real cost:
codecontext auto-start-request --root . \
    --goal "where do we validate user email addresses?" \
    --top-k 6 --token-budget 1500

# 6. See what you actually spent and saved:
codecontext sales-summary --window 7d
```

You'll get something like:

```
7-day window: 128 requests, 41 routed local, 87 to external.
External tokens sent: 94,312 (vs 312,447 naive baseline, −70%).
Spend: $0.72. Estimated savings vs full-context baseline: $2.31.
```

---

## How it works (the short version)

Every request passes through five stages:

1. **Scan & index** — Walks your project (respecting `.gitignore`), extracts symbols, builds a BM25 index and optional semantic embeddings.
2. **Classify** — Decides whether the prompt needs external reasoning at all, or can be answered by a local model / canned route. This is where the biggest savings come from.
3. **Pack context** — For prompts that do go external, assembles the top-K most relevant chunks into a token budget you control. Novelty penalty avoids duplicate content. Symbol body bonus keeps full function definitions together.
4. **Call the model** — Speaks OpenAI, Anthropic, Ollama, or any OpenAI-compatible endpoint over HTTP. Caps per-request cost at `max_escalation_cost_per_request` USD (default $0.08).
5. **Log & learn** — Writes real token counts and dollar cost to `.codecontext/data/codecontext.db`. Every report in the CLI is backed by this ledger — not estimates, not dashboards.

---

## Configuration

CodeContext looks for `.codecontext.toml` in the project root. A minimal file:

```toml
[llm_client]
enabled     = true
provider    = "openai"            # openai | anthropic | ollama | openai_compatible
model       = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

See [`.codecontext.toml`](./.codecontext.toml) in this repo for a fully commented example covering OpenAI, Anthropic, Ollama, and self-hosted / OpenAI-compatible endpoints (vLLM, LM Studio, Groq, Together, DeepSeek, RunPod).

**Secrets rule:** API keys live in environment variables, not in config files. `api_key_env` names the variable; CodeContext reads it at call time. Keys never touch the config file, git history, or crash dumps.

### Key config knobs

| Field | Default | What it does |
|---|---|---|
| `default_context_budget_tokens` | `4000` | Max tokens CodeContext packs into context for external calls. Lower = cheaper, less context. |
| `max_escalation_cost_per_request` | `0.08` | Hard per-request spend cap in USD. Above this, the request is routed locally or blocked. |
| `enable_embeddings` | `true` | Use semantic retrieval on top of BM25. Requires `[embeddings]` extra. Falls back gracefully if missing. |
| `llm_client.enabled` | `false` | Gateway returns the outbound payload without calling the model. Useful for dry runs / CI. |
| `llm_client.max_tokens` | `1024` | Max output tokens. |
| `llm_client.temperature` | `0.2` | Low default — CodeContext is typically used for code reasoning, not creative generation. |

---

## CLI reference

All commands take `--root <path>` (defaults to `.`) and emit JSON to stdout.

### Indexing

```bash
codecontext scan-project                    # walk the tree, report what's indexable
codecontext index-project                   # build the full index
codecontext refresh-changed-files           # fast re-index of modified files
```

### Retrieval

```bash
codecontext search-project --query "rate limiter" --top-k 8
codecontext prepare-context-pack --goal "add retries to the HTTP client" --top-k 6 --token-budget 1500
```

### Request execution (the main flow)

```bash
# End-to-end: classify, pack context, call model, log cost.
codecontext auto-start-request --goal "why is /login slow?" --top-k 6 --token-budget 1500

# Manual flow: pack context, get outbound payload, call your own client,
# then feed the response back in.
codecontext route-request --goal "..."
codecontext handle-remote-response --response-file reply.txt
```

### Reporting

```bash
codecontext sales-summary --window 7d                # human-readable savings summary
codecontext metrics-report --window 7d --top-n 10    # machine-readable metrics
codecontext usage-ledger-report --window 30d         # every call, every token, every dollar
codecontext benchmark-report                         # run the built-in savings benchmark
```

### Patching (experimental)

```bash
codecontext apply-patch --path src/foo.py \
    --old-text "def broken(): pass" \
    --new-text "def fixed(): return 1" \
    --dry-run

codecontext rollback-patch --patch-id 42
```

---

## HTTP API

For integration into apps, run the gateway as a local HTTP server:

```bash
codecontext serve-api --host 127.0.0.1 --port 8787
```

Then:

```bash
# Health
curl http://127.0.0.1:8787/health

# Route or run a prompt
curl -X POST http://127.0.0.1:8787/route-or-run \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "check logs for the local-first runtime",
        "options": { "top_k": 4, "token_budget": 1200 }
    }'

# 7-day savings summary
curl "http://127.0.0.1:8787/sales-summary?window=7d"
```

See [`codecontext/PRODUCT_API.md`](./codecontext/PRODUCT_API.md) for the full endpoint reference, request/response fields, and production-readiness checklist (auth, rate limits, and tenant isolation are currently the user's responsibility — the server is a local-demo surface, not a multi-tenant SaaS).

---

## Python API

For finer-grained use inside your own code:

```python
from pathlib import Path
from codecontext.config import AppConfig
from codecontext.gateway import CodeContextGateway
from codecontext.executor import AutoExecutor

config  = AppConfig(Path("."))
gateway = CodeContextGateway(config)
executor = AutoExecutor(gateway)

# Index once:
from codecontext.summaries import SummaryManager
from codecontext.metrics import Metrics
SummaryManager(config).index_project(metrics=Metrics())

# Then route requests through the gateway:
result = executor.start(
    goal="where is the rate limiter implemented?",
    top_k=6,
    token_budget=1500,
)
print(result["chosen_route"], result["cost_estimate"])
```

The gateway returns a structured dict: `chosen_route`, `class_reason`, `cost_estimate`, `estimated_savings_vs_external`, `run_id`, and more. Every field is documented in `codecontext/PRODUCT_API.md`.

---

## What "40–70% savings" actually means

CodeContext ships with a benchmark harness that compares its routing + packing against a naive baseline ("send the whole file / whole prompt to the frontier model") on a fixed dataset of code-reasoning tasks. On that dataset, the reduction in external tokens is 40–70% depending on task mix — heavily local-intent workloads score higher, heavy reasoning workloads score lower.

**Your number will vary.** To measure it on your own workload:

```bash
codecontext benchmark-run --dataset default --runs 3
codecontext benchmark-sales-summary
```

The ledger behind every report is real: real tokens reported by the provider, real dollar cost from the published price tables. No extrapolation, no estimates pretending to be measurements.

---

## Licence

MIT — see [LICENSE](./LICENSE). You can use CodeContext freely in commercial and private projects. Paid tiers bundle support, priority fixes, and proprietary benchmark datasets; the code itself is and will remain MIT.

---

## Support & links

- **Issues:** [github.com/batish52/codecontext/issues](https://github.com/batish52/codecontext/issues)
- **Changelog:** [CHANGELOG.md](./CHANGELOG.md)
- **Sister project:** [`llm-costlog`](https://pypi.org/project/llm-costlog/) — the lightweight cost-logging library that funnels into CodeContext.
