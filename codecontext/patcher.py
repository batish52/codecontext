from __future__ import annotations

import difflib
import shutil
import time
from pathlib import Path

from .config import AppConfig
from .db import connect


class Patcher:
    def __init__(self, config: AppConfig):
        self.config = config

    def apply_patch(self, rel_path: str, old_text: str, new_text: str, dry_run: bool = False) -> dict:
        path = (self.config.root / rel_path).resolve()
        if not path.exists():
            raise FileNotFoundError(rel_path)
        original = path.read_text(encoding="utf-8")
        if old_text not in original:
            raise ValueError("old_text not found exactly")
        updated = original.replace(old_text, new_text, 1)
        diff_text = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=rel_path,
                tofile=rel_path,
                lineterm="",
            )
        )
        backup = self.config.backup_dir / f"{path.name}.{int(time.time())}.bak"
        if not dry_run:
            shutil.copy2(path, backup)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            tmp.replace(path)
        conn = connect(self.config.db_path)
        with conn:
            conn.execute(
                "INSERT INTO patches(path, backup_path, diff_text, created_at, dry_run) VALUES(?,?,?,?,?)",
                (rel_path, str(backup), diff_text, time.time(), int(dry_run)),
            )
        patch_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()[0]
        return {"path": rel_path, "backup": str(backup), "dry_run": dry_run, "diff": diff_text, "patch_id": patch_id}

    def rollback_patch(self, patch_id: int) -> dict:
        conn = connect(self.config.db_path)
        row = conn.execute(
            "SELECT id, path, backup_path, dry_run FROM patches WHERE id = ?",
            (patch_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"patch_id not found: {patch_id}")
        if int(row["dry_run"]):
            raise ValueError("cannot rollback a dry-run patch")
        target = self.config.root / row["path"]
        backup = Path(row["backup_path"])
        if not backup.exists():
            raise FileNotFoundError(str(backup))
        tmp = target.with_suffix(target.suffix + ".rollback.tmp")
        tmp.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.replace(target)
        return {"rolled_back": True, "patch_id": patch_id, "path": row["path"], "backup": str(backup)}
