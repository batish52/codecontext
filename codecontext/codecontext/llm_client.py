"""LLM client for CodeContext's external routing path.

This module is the thin adapter that turns CodeContext's outbound payload
(produced by ``outbound_schema.build_outbound_request``) into an actual LLM
call and returns a normalized ``LLMResponse`` containing the model's text,
real token counts, real $ cost, and provider metadata.

Design goals:
  * Zero hard dependencies. We speak each provider's HTTP API directly using
    ``urllib``. If the user has ``openai`` / ``anthropic`` SDKs installed we
    still don't touch them — everything is one code path and easy to mock.
  * Opt-in. If ``AppConfig.llm_client.enabled`` is false (the default), the
    executor returns the outbound payload as before (Phase 1 behaviour) and
    no LLM call is made. This preserves backward compatibility.
  * Provider-portable. Four providers are supported out of the box:
      - ``openai``             — https://api.openai.com/v1/chat/completions
      - ``anthropic``          — https://api.anthropic.com/v1/messages
      - ``ollama``             — http://<host>/api/chat
      - ``openai_compatible``  — any endpoint that speaks the OpenAI
        chat/completions shape (vLLM, LM Studio, Groq, Together, Fireworks,
        DeepSeek, RunPod-hosted Qwen, etc.)
  * Honest cost accounting. We read token counts from the provider's own
    usage block when present, fall back to our approximator otherwise, and
    price them from a built-in table that can be overridden in config.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .costing import estimate_text_tokens


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# USD per 1M tokens, (input, output). Values are deliberately conservative and
# easy to override via LLMClientConfig.pricing. Models we don't recognise fall
# through to a permissive $5 / $15 default — the same default usage_ledger
# uses — so cost is never silently zero for external calls.
#
# Matching is prefix-based on lowercased model name, longest-prefix wins.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI — https://openai.com/api/pricing
    "gpt-5":                (1.25,  10.00),
    "gpt-4.1":              (2.00,   8.00),
    "gpt-4o-mini":          (0.15,   0.60),
    "gpt-4o":               (2.50,  10.00),
    "gpt-4-turbo":          (10.00, 30.00),
    "gpt-4":                (30.00, 60.00),
    "gpt-3.5-turbo":        (0.50,   1.50),
    "o1-mini":              (1.10,   4.40),
    "o1":                   (15.00, 60.00),
    "o3-mini":              (1.10,   4.40),
    "o3":                   (2.00,   8.00),
    # Anthropic — https://www.anthropic.com/pricing
    "claude-opus-4":        (15.00, 75.00),
    "claude-sonnet-4":      (3.00,  15.00),
    "claude-haiku-4":       (1.00,   5.00),
    "claude-3-5-sonnet":    (3.00,  15.00),
    "claude-3-5-haiku":     (0.80,   4.00),
    "claude-3-opus":        (15.00, 75.00),
    "claude-3-sonnet":      (3.00,  15.00),
    "claude-3-haiku":       (0.25,   1.25),
    # Open-weight models are NOT listed here with $0 defaults. The
    # previous behaviour — `"qwen": (0.0, 0.0)`, `"llama": (0.0, 0.0)`,
    # etc. — silently recorded $0 cost even when users pointed
    # openai_compatible at paid providers (DeepInfra, Fireworks,
    # Together.ai, Groq, RunPod GPU rental). Users who truly self-host
    # should add an explicit (0.0, 0.0) entry for their specific
    # model via LLMClientConfig.pricing; users on paid providers get
    # accurate fallback pricing instead of silent $0.
}

_FALLBACK_PRICING = (5.0, 15.0)


class _LLMConfigError(Exception):
    """Raised when the client is misconfigured (e.g. missing API key).

    Distinct from parse/HTTP errors so callers can tell setup failures
    apart from provider-side failures in the usage ledger.
    """


def _lookup_pricing(model: str, overrides: dict[str, tuple[float, float]] | None = None) -> tuple[float, float, str]:
    """Return (input_per_1m, output_per_1m, match_key) for a model name."""
    if not model:
        return (*_FALLBACK_PRICING, "default")
    lo = model.lower()
    tables: list[dict[str, tuple[float, float]]] = []
    if overrides:
        tables.append({k.lower(): v for k, v in overrides.items()})
    tables.append(_DEFAULT_PRICING)
    best_key: str | None = None
    best_val: tuple[float, float] | None = None
    for table in tables:
        for key, val in table.items():
            if lo.startswith(key) and (best_key is None or len(key) > len(best_key)):
                best_key, best_val = key, val
        if best_key:
            return (best_val[0], best_val[1], best_key)  # type: ignore[index]
    return (*_FALLBACK_PRICING, "default")


# ---------------------------------------------------------------------------
# Config + response types
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LLMClientConfig:
    """Configuration for the external LLM call.

    The client is opt-in: when ``enabled=False`` the executor skips the LLM
    call entirely and returns the outbound payload (Phase 1 behaviour).

    ``api_key_env`` names an environment variable that holds the key, rather
    than storing the key directly. This keeps secrets out of config files
    and crash dumps — a soft rule that matches the rest of the codebase's
    secret-handling posture.
    """

    enabled: bool = False
    provider: str = "openai"                 # openai | anthropic | ollama | openai_compatible
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None              # override default endpoint
    timeout_seconds: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)
    pricing: dict[str, tuple[float, float]] = field(default_factory=dict)
    anthropic_version: str = "2023-06-01"

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None


@dataclass(slots=True)
class LLMResponse:
    """Normalized output across all providers.

    ``prompt_tokens`` / ``completion_tokens`` / ``cost_usd`` are shaped to
    drop straight into the usage-ledger metrics dict. ``text`` is the
    assistant-message content the gateway's response_handler will classify.
    """

    ok: bool
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    provider: str
    model: str
    latency_ms: int
    pricing_source: str
    error_code: str | None = None
    error_message: str | None = None
    raw: dict | None = None

    def as_metrics_overlay(self) -> dict[str, Any]:
        """Return the fields the usage ledger reads from the metrics dict."""
        return {
            "prompt_tokens_actual": self.prompt_tokens,
            "completion_tokens_actual": self.completion_tokens,
            "cost_actual_usd": self.cost_usd,
            "estimation_method": f"llm_usage:{self.provider}/{self.model}",
        }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------
class LLMClient:
    """Calls a configured LLM and returns a normalized LLMResponse.

    Usage::

        client = LLMClient(config.llm_client)
        resp = client.complete(system=..., user=...)
        if resp.ok:
            response_text = resp.text
    """

    def __init__(self, config: LLMClientConfig):
        self.config = config

    # -- public API ---------------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        cfg = self.config
        model = model or cfg.model
        max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        temperature = temperature if temperature is not None else cfg.temperature
        provider = (cfg.provider or "openai").lower().strip()

        started = time.time()
        try:
            if provider == "openai":
                raw = self._call_openai_compatible(
                    system=system, user=user, model=model,
                    max_tokens=max_tokens, temperature=temperature,
                    base_url=cfg.base_url or "https://api.openai.com/v1",
                    requires_auth=True,
                )
                text, pt, ct = self._parse_openai_response(raw)
            elif provider == "openai_compatible":
                raw = self._call_openai_compatible(
                    system=system, user=user, model=model,
                    max_tokens=max_tokens, temperature=temperature,
                    base_url=cfg.base_url or "http://localhost:8000/v1",
                    requires_auth=bool(cfg.resolve_api_key()),
                )
                text, pt, ct = self._parse_openai_response(raw)
            elif provider == "anthropic":
                raw = self._call_anthropic(
                    system=system, user=user, model=model,
                    max_tokens=max_tokens, temperature=temperature,
                )
                text, pt, ct = self._parse_anthropic_response(raw)
            elif provider == "ollama":
                raw = self._call_ollama(
                    system=system, user=user, model=model,
                    max_tokens=max_tokens, temperature=temperature,
                )
                text, pt, ct = self._parse_ollama_response(raw)
            else:
                return self._err(
                    provider, model, started,
                    code="unsupported_provider",
                    message=f"provider '{provider}' is not supported",
                )
        except urllib.error.HTTPError as e:
            body = _safe_read_error_body(e)
            return self._err(provider, model, started, code=f"http_{e.code}", message=body[:500])
        except urllib.error.URLError as e:
            return self._err(provider, model, started, code="url_error", message=str(e.reason)[:500])
        except TimeoutError as e:
            return self._err(provider, model, started, code="timeout", message=str(e)[:500])
        except _LLMConfigError as e:
            return self._err(provider, model, started, code="config_error", message=str(e)[:500])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            return self._err(provider, model, started, code="parse_error", message=str(e)[:500])

        # Fallback token estimation ONLY when the provider didn't report
        # usage at all. A legitimate `prompt_tokens: 0` from Anthropic can
        # mean all input was served from the prompt cache — overwriting
        # that with a fresh full-prompt estimate would bill the user as if
        # caching never happened. `pt`/`ct` arrive as `None` from the
        # parsers when the usage field was absent, and as `int` (possibly
        # 0) when it was genuinely reported.
        if pt is None:
            pt = estimate_text_tokens(system) + estimate_text_tokens(user)
        if ct is None:
            ct = estimate_text_tokens(text or "")

        in_price, out_price, price_source = _lookup_pricing(model, self.config.pricing)
        cost_usd = round((pt / 1_000_000.0) * in_price + (ct / 1_000_000.0) * out_price, 8)
        latency_ms = int((time.time() - started) * 1000)

        return LLMResponse(
            ok=True,
            text=text or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            cost_usd=cost_usd,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            pricing_source=f"{provider}/{price_source}",
            raw=raw,
        )

    # Model-name prefixes that require OpenAI's newer `max_completion_tokens`
    # parameter instead of the legacy `max_tokens`. Sending `max_tokens`
    # to these models returns HTTP 400. They also only accept the default
    # temperature (1.0), so we omit that field entirely for them.
    _OPENAI_REASONING_PREFIXES: tuple[str, ...] = ("o1", "o3", "gpt-5")

    @classmethod
    def _is_openai_reasoning_model(cls, model: str) -> bool:
        lo = (model or "").lower()
        # Match "o1", "o1-mini", "o3", "o3-mini", "gpt-5", "gpt-5-turbo"
        # etc. but not e.g. "o11y-model" — require a token boundary.
        for pref in cls._OPENAI_REASONING_PREFIXES:
            if lo == pref or lo.startswith(pref + "-") or lo.startswith(pref + "."):
                return True
        return False

    # -- provider calls -----------------------------------------------------
    def _call_openai_compatible(
        self,
        *, system: str, user: str, model: str,
        max_tokens: int, temperature: float,
        base_url: str, requires_auth: bool,
    ) -> dict:
        url = base_url.rstrip("/") + "/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Reasoning models (o1/o3/gpt-5 family) require
        # `max_completion_tokens` and reject `temperature != 1.0`.
        # Everything else still uses `max_tokens` + `temperature`.
        if self._is_openai_reasoning_model(model):
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        api_key = self.config.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif requires_auth:
            raise _LLMConfigError(
                f"api key not found in env var '{self.config.api_key_env}' (required for provider '{self.config.provider}')"
            )
        return _http_post_json(url, body, headers, self.config.timeout_seconds)

    def _call_anthropic(
        self,
        *, system: str, user: str, model: str,
        max_tokens: int, temperature: float,
    ) -> dict:
        base = (self.config.base_url or "https://api.anthropic.com").rstrip("/")
        url = base + "/v1/messages"
        api_key = self.config.resolve_api_key()
        if not api_key:
            raise _LLMConfigError(f"api key not found in env var '{self.config.api_key_env}'")
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self.config.anthropic_version,
            **self.config.extra_headers,
        }
        return _http_post_json(url, body, headers, self.config.timeout_seconds)

    def _call_ollama(
        self,
        *, system: str, user: str, model: str,
        max_tokens: int, temperature: float,
    ) -> dict:
        base = (self.config.base_url or "http://localhost:11434").rstrip("/")
        url = base + "/api/chat"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        return _http_post_json(url, body, headers, self.config.timeout_seconds)

    # -- response parsing ---------------------------------------------------
    # Each parser returns (text, prompt_tokens_or_None, completion_tokens_or_None).
    # `None` means "the provider didn't report this count" — the caller
    # will fall back to approximation. `0` means "the provider explicitly
    # reported zero" (e.g. Anthropic prompt caching can report 0 input
    # tokens) — the caller trusts it and bills accordingly.
    @staticmethod
    def _parse_openai_response(raw: dict) -> tuple[str, int | None, int | None]:
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some compat endpoints return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return text, None, None
        pt_raw = usage.get("prompt_tokens")
        ct_raw = usage.get("completion_tokens")
        pt = int(pt_raw) if isinstance(pt_raw, (int, float)) else None
        ct = int(ct_raw) if isinstance(ct_raw, (int, float)) else None
        return text, pt, ct

    @staticmethod
    def _parse_anthropic_response(raw: dict) -> tuple[str, int | None, int | None]:
        # Anthropic returns content as a list of blocks — we concatenate text blocks.
        parts = raw.get("content") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return text, None, None
        pt_raw = usage.get("input_tokens")
        ct_raw = usage.get("output_tokens")
        pt = int(pt_raw) if isinstance(pt_raw, (int, float)) else None
        ct = int(ct_raw) if isinstance(ct_raw, (int, float)) else None
        return text, pt, ct

    @staticmethod
    def _parse_ollama_response(raw: dict) -> tuple[str, int | None, int | None]:
        msg = raw.get("message") or {}
        text = msg.get("content") or raw.get("response", "") or ""
        pt_raw = raw.get("prompt_eval_count")
        ct_raw = raw.get("eval_count")
        pt = int(pt_raw) if isinstance(pt_raw, (int, float)) else None
        ct = int(ct_raw) if isinstance(ct_raw, (int, float)) else None
        return text, pt, ct

    # -- helpers ------------------------------------------------------------
    def _err(self, provider: str, model: str, started: float, *, code: str, message: str) -> LLMResponse:
        return LLMResponse(
            ok=False,
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            provider=provider,
            model=model,
            latency_ms=int((time.time() - started) * 1000),
            pricing_source="error",
            error_code=code,
            error_message=message,
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a code assistant working strictly from the repository context "
    "provided in the user message. The user message is a JSON envelope that "
    "follows the 'codecontext.outbound.v1' schema; read its 'repository_context', "
    "'constraints', and 'expected_response' fields carefully and respond in one "
    "of these JSON envelope shapes:\n"
    '  {"kind":"answer","answer":"..."}\n'
    '  {"kind":"patch_instruction","patch":{"path":"...","old_text":"...","new_text":"...","dry_run":false}}\n'
    '  {"kind":"needs_more_context","request":"..."}\n'
    "Do not invent files or APIs that are not present in the evidence. If the "
    "evidence is insufficient, prefer 'needs_more_context' over guessing."
)


def render_user_message(outbound_payload: dict) -> str:
    """Serialize the outbound payload as the user message."""
    return json.dumps(outbound_payload, ensure_ascii=False)


def call_for_outbound(client: LLMClient, outbound_payload: dict, *, system_prompt: str | None = None) -> LLMResponse:
    """Run a CodeContext outbound payload through the configured LLM."""
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    return client.complete(system=system, user=render_user_message(outbound_payload))


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------
def _http_post_json(url: str, body: dict, headers: dict[str, str], timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    if not payload:
        return {}
    return json.loads(payload)


def _safe_read_error_body(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read()
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return f"HTTP {e.code}"
