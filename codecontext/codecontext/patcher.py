from __future__ import annotations

import difflib
import os
import shutil
import time
from pathlib import Path

from .config import AppConfig
from .db import connect


class Patcher:
    def __init__(self, config: AppConfig):
        self.config = config
        self._backup_seq = 0

    def _next_backup_path(self, path: Path) -> Path:
        # Use monotonic_ns + an in-process sequence so two patches to the
        # same file within one wall-clock second don't collide and
        # silently overwrite each other's backups.
        self._backup_seq += 1
        stamp = f"{time.time_ns()}.{self._backup_seq}"
        return self.config.backup_dir / f"{path.name}.{stamp}.bak"

    def apply_patch(self, rel_path: str, old_text: str, new_text: str, dry_run: bool = False) -> dict:
        path = (self.config.root / rel_path).resolve()
        if not path.exists():
            raise FileNotFoundError(rel_path)
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ValueError("old_text not found exactly")
        if occurrences > 1:
            # Refuse to guess which occurrence to patch. The caller must
            # make old_text unique (e.g. by including surrounding
            # context) so the patch is unambiguous.
            raise ValueError(
                f"old_text matches {occurrences} times in {rel_path}; "
                "expand old_text with surrounding context so it matches exactly once"
            )
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
        backup = self._next_backup_path(path) if not dry_run else self.config.backup_dir / f"{path.name}.dry-run"
        if not dry_run:
            shutil.copy2(path, backup)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            # Preserve the original file's permission bits (0755 script
            # stays 0755 after patch). tmp.write_text creates a new inode
            # with umask-default perms; shutil.copymode mirrors the
            # original file's mode onto the tmp file before atomic rename.
            try:
                shutil.copymode(path, tmp)
            except OSError:
                # Best-effort on exotic filesystems; we still complete
                # the patch rather than failing.
                pass
            tmp.replace(path)
        conn = connect(self.config.db_path)
        with conn:
            conn.execute(
                "INSERT INTO patches(path, backup_path, diff_text, created_at, dry_run) VALUES(?,?,?,?,?)",
                (rel_path, str(backup) if not dry_run else "", diff_text, time.time(), int(dry_run)),
            )
        patch_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()[0]
        return {"path": rel_path, "backup": str(backup) if not dry_run else "", "dry_run": dry_run, "diff": diff_text, "patch_id": patch_id}

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
        try:
            shutil.copymode(target, tmp)
        except OSError:
            pass
        tmp.replace(target)
        return {"rolled_back": True, "patch_id": patch_id, "path": row["path"], "backup": str(backup)}
