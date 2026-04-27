from __future__ import annotations

import json
from dataclasses import dataclass

from .patcher import Patcher
from .summaries import SummaryManager


@dataclass(slots=True)
class RemoteResponseDecision:
    kind: str
    reason: str
    payload: dict | None = None


class ResponseHandler:
    def __init__(self, patcher: Patcher, summaries: SummaryManager):
        self.patcher = patcher
        self.summaries = summaries

    def _parse_json_envelope(self, response_text: str) -> dict | None:
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict) and isinstance(data.get("kind"), str):
            return data
        return None

    def classify_response(self, response_text: str) -> RemoteResponseDecision:
        envelope = self._parse_json_envelope(response_text)
        if envelope:
            kind = envelope.get("kind")
            if kind == "needs_more_context":
                return RemoteResponseDecision(kind="needs_more_context", reason="structured response requested more context", payload=envelope)
            if kind == "patch_instruction":
                return RemoteResponseDecision(kind="patch_instruction", reason="structured patch payload detected", payload=envelope)
            if kind == "answer":
                return RemoteResponseDecision(kind="answer", reason="structured answer payload detected", payload=envelope)

        low = response_text.lower()
        if "need more context" in low or "need more snippet" in low or "need more file" in low:
            return RemoteResponseDecision(kind="needs_more_context", reason="model requested additional scoped context")
        # Bug #16: previously this checked `"old_text"`/`"new_text"`
        # against the original response_text (case-sensitive) but
        # `"path"` against `low` (lowercased). A response with
        # `OLD_TEXT`/`NEW_TEXT` would miss; one with `PATH` would hit.
        # Use lowercased text for all three so the heuristic is
        # consistently case-insensitive.
        if "old_text" in low and "new_text" in low and "path" in low:
            return RemoteResponseDecision(kind="patch_instruction", reason="response looks like a patch payload")
        return RemoteResponseDecision(kind="answer", reason="response is plain answer/explanation")

    def patch_format_instructions(self) -> dict:
        return {
            "required_remote_patch_format": {
                "kind": "patch_instruction",
                "patch": {
                    "path": "relative/path/to/file.py",
                    "old_text": "exact old text to replace",
                    "new_text": "replacement text",
                    "dry_run": False,
                },
            },
            "alternate_answer_format": {
                "kind": "answer",
                "answer": "plain explanation or result text",
            },
            "alternate_more_context_format": {
                "kind": "needs_more_context",
                "request": "narrow follow-up retrieval request",
            },
        }

    def handle_patch_payload(
        self,
        path: str,
        old_text: str,
        new_text: str,
        dry_run: bool = False,
    ) -> dict:
        patch_result = self.patcher.apply_patch(path, old_text, new_text, dry_run=dry_run)
        refresh_result = self.summaries.refresh_changed_files()
        return {
            "kind": "patch_applied",
            "patch": patch_result,
            "refresh": refresh_result,
        }

    def handle_structured_payload(self, payload: dict) -> dict:
        kind = payload.get("kind")
        if kind == "answer":
            return {
                "kind": "answer",
                "response_text": payload.get("answer", ""),
                "structured": True,
            }
        if kind == "needs_more_context":
            return {
                "kind": "needs_more_context",
                "request": payload.get("request", ""),
                "structured": True,
            }
        if kind == "patch_instruction":
            patch = payload.get("patch") or {}
            path = patch.get("path")
            old_text = patch.get("old_text")
            new_text = patch.get("new_text")
            dry_run = bool(patch.get("dry_run", False))
            if not path or old_text is None or new_text is None:
                return {
                    "kind": "patch_instruction_invalid",
                    "structured": True,
                    "payload": payload,
                }
            result = self.handle_patch_payload(path, old_text, new_text, dry_run=dry_run)
            result["structured"] = True
            return result
        return {
            "kind": "unknown_structured_response",
            "structured": True,
            "payload": payload,
        }
