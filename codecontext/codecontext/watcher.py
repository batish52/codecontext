from __future__ import annotations

import time
from dataclasses import dataclass

from .scanner import ProjectScanner


@dataclass(slots=True)
class WatchSnapshot:
    file_count: int
    generated_at: float
    paths: list[str]


class ProjectWatcher:
    """Phase 2 scaffold for event-driven refresh.

    Current MVP: lightweight polling snapshot, suitable for cron/manual refresh.
    Later swap with watchdog/watchfiles without changing callers.
    """

    def __init__(self, scanner: ProjectScanner):
        self.scanner = scanner

    def snapshot(self) -> WatchSnapshot:
        records = self.scanner.scan()
        return WatchSnapshot(
            file_count=len(records),
            generated_at=time.time(),
            paths=[r.path for r in records],
        )

    def diff_paths(self, before: WatchSnapshot, after: WatchSnapshot) -> dict:
        a = set(before.paths)
        b = set(after.paths)
        return {
            "added": sorted(b - a),
            "removed": sorted(a - b),
            "unchanged_count": len(a & b),
        }
