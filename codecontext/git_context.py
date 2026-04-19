from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess


class GitContext:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.available = False

        git_check = self._run_git(["--version"])
        if git_check is None or git_check.returncode != 0:
            return

        repo_check = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if repo_check is None or repo_check.returncode != 0:
            return

        if (repo_check.stdout or "").strip().lower() == "true":
            self.available = True

    def _run_git(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None

    def _parse_log_lines(self, text: str) -> list[dict]:
        out: list[dict] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            out.append(
                {
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                }
            )
        return out

    def _iso_to_epoch(self, iso_text: str) -> float | None:
        try:
            ts = datetime.fromisoformat(iso_text.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()
        except Exception:
            return None

    def is_available(self) -> bool:
        return self.available

    def recent_commits(self, n: int = 10) -> list[dict]:
        if not self.available:
            return []
        p = self._run_git(["log", "--oneline", "--format=%H|%an|%aI|%s", "-n", str(max(1, int(n)))])
        if p is None or p.returncode != 0:
            return []
        return self._parse_log_lines(p.stdout or "")

    def file_log(self, path: str, n: int = 10) -> list[dict]:
        if not self.available:
            return []
        p = self._run_git(["log", "--oneline", "--format=%H|%an|%aI|%s", "-n", str(max(1, int(n))), "--", path])
        if p is None or p.returncode != 0:
            return []
        return self._parse_log_lines(p.stdout or "")

    def diff_summary(self, ref: str = "HEAD~1") -> list[dict]:
        if not self.available:
            return []
        p = self._run_git(["diff", "--stat", "--name-status", ref])
        if p is None or p.returncode != 0:
            return []

        out: list[dict] = []
        for line in (p.stdout or "").splitlines():
            raw = line.strip()
            if not raw or raw.startswith(" "):
                continue
            parts = raw.split("\t")
            if len(parts) >= 2 and parts[0] in {"A", "M", "D", "R", "C", "T", "U"}:
                status = parts[0][0]
                path = parts[-1].strip()
            else:
                continue

            insertions = 0
            deletions = 0
            if "|" in raw:
                right = raw.split("|", 1)[1]
                insertions = right.count("+")
                deletions = right.count("-")

            out.append(
                {
                    "path": path,
                    "status": status,
                    "insertions": insertions,
                    "deletions": deletions,
                }
            )
        return out

    def file_diff(self, path: str, ref: str = "HEAD~1") -> str:
        if not self.available:
            return ""
        p = self._run_git(["diff", ref, "--", path])
        if p is None or p.returncode != 0:
            return ""
        return (p.stdout or "")[:5000]

    def blame_summary(self, path: str, start_line: int = 1, end_line: int | None = None) -> list[dict]:
        if not self.available:
            return []

        fpath = self.root / path
        if end_line is None:
            try:
                end_line = len(fpath.read_text(encoding="utf-8", errors="ignore").splitlines())
            except Exception:
                end_line = start_line

        start = max(1, int(start_line))
        end = max(start, int(end_line or start))
        p = self._run_git(["blame", "-L", f"{start},{end}", "--porcelain", path])
        if p is None or p.returncode != 0:
            return []

        results: list[dict] = []
        commit_hash = ""
        author = ""
        author_time = ""
        current_line = start - 1
        for line in (p.stdout or "").splitlines():
            if not line:
                continue
            if line.startswith("\t"):
                current_line += 1
                results.append(
                    {
                        "line": current_line,
                        "author": author,
                        "date": author_time,
                        "commit_hash": commit_hash,
                        "content": line[1:],
                    }
                )
                continue

            parts = line.split()
            if len(parts) >= 4 and len(parts[0]) >= 7 and all(ch in "0123456789abcdef" for ch in parts[0].lower()[:7]):
                commit_hash = parts[0]
            elif line.startswith("author "):
                author = line[len("author "):]
            elif line.startswith("author-time "):
                try:
                    ts = datetime.fromtimestamp(int(line[len("author-time "):]), tz=timezone.utc)
                    author_time = ts.isoformat()
                except Exception:
                    author_time = ""
        return results

    def changed_files_since(self, hours: float = 24.0) -> list[dict]:
        if not self.available:
            return []
        p = self._run_git(["log", f"--since={float(hours)} hours ago", "--name-only", "--format="])
        if p is None or p.returncode != 0:
            return []

        seen: set[str] = set()
        out: list[dict] = []
        threshold = datetime.now(timezone.utc).timestamp() - (float(hours) * 3600.0)
        for path in (p.stdout or "").splitlines():
            pth = path.strip()
            if not pth or pth in seen:
                continue
            seen.add(pth)
            last = self.file_log(pth, n=1)
            changed_at = last[0]["date"] if last else ""
            changed_epoch = self._iso_to_epoch(changed_at) if changed_at else None
            if changed_epoch is not None and changed_epoch < threshold:
                continue
            out.append({"path": pth})
        return out

    def current_branch(self) -> str:
        if not self.available:
            return ""
        p = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if p is None or p.returncode != 0:
            return ""
        return (p.stdout or "").strip()

    def uncommitted_changes(self) -> dict:
        out = {"staged": [], "unstaged": [], "untracked": []}
        if not self.available:
            return out

        p = self._run_git(["status", "--porcelain"])
        if p is None or p.returncode != 0:
            return out

        for line in (p.stdout or "").splitlines():
            if len(line) < 3:
                continue
            x = line[0]
            y = line[1]
            path = line[3:].strip()
            if x == "?" and y == "?":
                out["untracked"].append(path)
                continue
            if x != " ":
                out["staged"].append(path)
            if y != " ":
                out["unstaged"].append(path)

        for key in out:
            out[key] = sorted(dict.fromkeys(out[key]))
        return out
