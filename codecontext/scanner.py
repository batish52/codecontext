from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from pathspec import PathSpec

from .config import AppConfig, IMPORTANT_EXTENSIONS, IMPORTANT_FILE_NAMES
from .utils import is_binary_bytes, sha256_file


@dataclass(slots=True)
class FileRecord:
    path: str
    size: int
    mtime: float
    extension: str
    sha256: str
    is_binary: bool
    importance: int


class ProjectScanner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.include_spec = PathSpec.from_lines("gitwildmatch", config.include)
        self.exclude_spec = PathSpec.from_lines("gitwildmatch", config.exclude)

    def should_include(self, rel_path: str) -> bool:
        rel = rel_path.replace("\\", "/")
        if self.exclude_spec.match_file(rel):
            return False
        return self.include_spec.match_file(rel)

    def score_importance(self, path: Path) -> int:
        score = IMPORTANT_EXTENSIONS.get(path.suffix.lower(), 1)
        score += IMPORTANT_FILE_NAMES.get(path.name.lower(), 0)
        lower = path.as_posix().lower()
        for marker, boost in (("test", 3), ("config", 3), ("schema", 3), ("migration", 2), ("doc", 2), ("service", 2)):
            if marker in lower:
                score += boost
        return score

    def scan(self) -> list[FileRecord]:
        records: list[FileRecord] = []
        for path in self.config.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.config.root).as_posix()
            if not self.should_include(rel):
                continue
            stat = path.stat()
            with path.open("rb") as f:
                head = f.read(4096)
            is_binary = is_binary_bytes(head)
            records.append(
                FileRecord(
                    path=rel,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    extension=path.suffix.lower(),
                    sha256=sha256_file(path),
                    is_binary=is_binary,
                    importance=self.score_importance(path),
                )
            )
        return sorted(records, key=lambda r: (-r.importance, r.path))

    def write_manifest(self, records: list[FileRecord]) -> None:
        payload = {"generated_at": time.time(), "files": [asdict(r) for r in records]}
        self.config.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
