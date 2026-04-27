from __future__ import annotations

import json
import time
from pathlib import Path


class Metrics:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.values: dict[str, float | int | str] = {
            "files_scanned": 0,
            "files_changed": 0,
            "files_sent_remote": 0,
            "chunks_retrieved": 0,
            "chunks_sent": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "raw_bytes_avoided": 0,
            "estimated_prompt_tokens": 0,
            "estimated_saved_tokens": 0,
        }

    def inc(self, key: str, amount: int = 1) -> None:
        self.values[key] = int(self.values.get(key, 0)) + amount

    def set(self, key: str, value) -> None:
        self.values[key] = value

    def finish(self) -> dict:
        self.values["latency_ms"] = round((time.perf_counter() - self.started_at) * 1000, 2)
        return self.values

    def append_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.finish(), ensure_ascii=False) + "\n")
