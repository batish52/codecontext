from __future__ import annotations

import json
import time
from pathlib import Path

from .config import AppConfig
from .db import connect
from .embeddings import EmbeddingProvider
from .extractors import summarize_text
from .metrics import Metrics
from .redaction import looks_secret_path, redact_text
from .scanner import FileRecord, ProjectScanner
from .utils import read_text_safely


class SummaryManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embeddings = EmbeddingProvider(enabled=config.enable_embeddings)

    def index_project(self, metrics: Metrics | None = None) -> dict:
        metrics = metrics or Metrics()
        scanner = ProjectScanner(self.config)
        records = scanner.scan()
        scanner.write_manifest(records)
        conn = connect(self.config.db_path)
        changed = 0
        now = time.time()
        schema_row = conn.execute("SELECT value FROM project_state WHERE key = 'index_schema_version'").fetchone()
        schema_mismatch = (not schema_row) or (schema_row["value"] != self.config.index_schema_version)
        with conn:
            live_paths = {record.path for record in records}
            stale_rows = conn.execute("SELECT path FROM files").fetchall()
            for stale in stale_rows:
                if stale["path"] not in live_paths:
                    conn.execute("DELETE FROM files WHERE path = ?", (stale["path"],))
                    conn.execute("DELETE FROM embeddings WHERE path = ?", (stale["path"],))
                    conn.execute("DELETE FROM chunk_embeddings WHERE path = ?", (stale["path"],))
                    conn.execute("DELETE FROM symbols WHERE path = ?", (stale["path"],))
            for record in records:
                metrics.inc("files_scanned")
                existing = conn.execute(
                    "SELECT sha256, summary_tiny FROM files WHERE path = ?",
                    (record.path,),
                ).fetchone()
                if existing and existing["sha256"] == record.sha256 and not schema_mismatch:
                    metrics.inc("cache_hits")
                    continue
                metrics.inc("cache_misses")
                changed += 1
                embedding_text = ""
                chunks = []
                if record.is_binary or record.size > self.config.max_file_bytes:
                    tiny = f"{record.path}: binary or oversized file ({record.size} bytes)"
                    detailed = tiny
                    symbol_summary = ""
                    symbols_json = "[]"
                else:
                    path = self.config.root / record.path
                    text = read_text_safely(path, self.config.max_file_bytes)
                    if looks_secret_path(path):
                        text = redact_text(text)
                    extracted = summarize_text(path, text)
                    tiny = extracted.tiny_summary
                    detailed = extracted.detailed_summary
                    symbol_summary = extracted.symbol_summary
                    symbols_json = json.dumps(extracted.symbols, ensure_ascii=False)
                    embedding_text = "\n".join(filter(None, [record.path, tiny, detailed, symbol_summary]))
                    chunks = extracted.snippets
                    conn.execute("DELETE FROM symbols WHERE path = ?", (record.path,))
                    for sym in extracted.symbols:
                        conn.execute(
                            "INSERT INTO symbols(path, symbol_name, symbol_type, line, doc, updated_at) VALUES(?,?,?,?,?,?)",
                            (
                                record.path,
                                sym.get("name", ""),
                                sym.get("type", ""),
                                sym.get("line"),
                                f"end_line={sym.get('end_line')} || {sym.get('doc', '')}",
                                now,
                            ),
                        )
                conn.execute(
                    """
                    INSERT INTO files(path, sha256, size, mtime, extension, is_binary, importance,
                                      summary_tiny, summary_detailed, summary_symbols, summary_change,
                                      symbols_json, last_indexed_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(path) DO UPDATE SET
                      sha256=excluded.sha256,
                      size=excluded.size,
                      mtime=excluded.mtime,
                      extension=excluded.extension,
                      is_binary=excluded.is_binary,
                      importance=excluded.importance,
                      summary_tiny=excluded.summary_tiny,
                      summary_detailed=excluded.summary_detailed,
                      summary_symbols=excluded.summary_symbols,
                      summary_change=excluded.summary_change,
                      symbols_json=excluded.symbols_json,
                      last_indexed_at=excluded.last_indexed_at
                    """,
                    (
                        record.path,
                        record.sha256,
                        record.size,
                        record.mtime,
                        record.extension,
                        int(record.is_binary),
                        record.importance,
                        tiny,
                        detailed,
                        symbol_summary,
                        "updated or newly indexed",
                        symbols_json,
                        now,
                    ),
                )
                if self.embeddings.is_available() and embedding_text:
                    emb = self.embeddings.embed_one(embedding_text)
                    conn.execute(
                        """
                        INSERT INTO embeddings(path, model, dim, vector_json, text_sha256, updated_at)
                        VALUES(?,?,?,?,?,?)
                        ON CONFLICT(path) DO UPDATE SET
                          model=excluded.model,
                          dim=excluded.dim,
                          vector_json=excluded.vector_json,
                          text_sha256=excluded.text_sha256,
                          updated_at=excluded.updated_at
                        """,
                        (
                            record.path,
                            self.embeddings.model_name,
                            self.embeddings.dim,
                            self.embeddings.to_json(emb.vector),
                            record.sha256,
                            now,
                        ),
                    )
                    conn.execute("DELETE FROM chunk_embeddings WHERE path = ?", (record.path,))
                    for chunk in chunks[:64]:
                        chunk_text = chunk.get("text", "")
                        if not chunk_text.strip():
                            continue
                        cemb = self.embeddings.embed_one(chunk_text)
                        conn.execute(
                            """
                            INSERT INTO chunk_embeddings(path, start_line, end_line, chunk_hash, model, dim, vector_json, chunk_text, updated_at)
                            VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                record.path,
                                int(chunk.get("start_line", 1)),
                                int(chunk.get("end_line", 1)),
                                f"{record.sha256}:{chunk.get('start_line', 1)}:{chunk.get('end_line', 1)}",
                                self.embeddings.model_name,
                                self.embeddings.dim,
                                self.embeddings.to_json(cemb.vector),
                                chunk_text,
                                now,
                            ),
                        )
        metrics.set("files_changed", changed)
        project_summary = self.build_project_summary(conn)
        with conn:
            conn.execute(
                "INSERT INTO project_state(key, value, updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                ("project_summary", json.dumps(project_summary, ensure_ascii=False), now),
            )
            conn.execute(
                "INSERT INTO project_state(key, value, updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                ("index_schema_version", self.config.index_schema_version, now),
            )
        metrics.append_jsonl(self.config.metrics_path)
        return {"files": len(records), "changed": changed, "project_summary": project_summary, "metrics": metrics.finish()}

    def refresh_changed_files(self, metrics: Metrics | None = None) -> dict:
        return self.index_project(metrics=metrics)

    def summarize_file(self, rel_path: str) -> dict:
        conn = connect(self.config.db_path)
        row = conn.execute(
            "SELECT path, summary_tiny, summary_detailed, summary_symbols, summary_change FROM files WHERE path = ?",
            (rel_path,),
        ).fetchone()
        if row:
            return dict(row)
        result = self.index_project()
        row = conn.execute(
            "SELECT path, summary_tiny, summary_detailed, summary_symbols, summary_change FROM files WHERE path = ?",
            (rel_path,),
        ).fetchone()
        return {"reindexed": True, "index_result": result, "file": dict(row) if row else None}

    def summarize_project(self) -> dict:
        conn = connect(self.config.db_path)
        row = conn.execute("SELECT value FROM project_state WHERE key = 'project_summary'").fetchone()
        if row:
            return json.loads(row["value"])
        return self.index_project()["project_summary"]

    def build_project_summary(self, conn) -> dict:
        rows = conn.execute(
            "SELECT path, importance, summary_tiny, summary_symbols FROM files ORDER BY importance DESC, size DESC LIMIT 20"
        ).fetchall()
        important_files = [r["path"] for r in rows]
        key_modules = [
            {"path": r["path"], "why": r["summary_tiny"], "symbols": r["summary_symbols"]}
            for r in rows[:10]
        ]
        conventions = []
        for r in rows:
            path = r["path"].lower()
            if path.endswith("requirements.txt"):
                conventions.append("Python project with pinned requirements")
            if path.endswith("run_dev.ps1"):
                conventions.append("PowerShell-based local dev entrypoint")
            if "/tests/" in path or "test" in path:
                conventions.append("Test files present in repo")
        return {
            "architecture_summary": "Local-first context middleware for coding/file workflows with cached summaries and selective context packing.",
            "important_files": important_files,
            "key_modules": key_modules,
            "known_conventions": sorted(set(conventions)),
            "recent_edits": [r["path"] for r in rows[:8]],
            "unresolved_issues": ["Phase 2 features not yet implemented", "No embeddings/vector retrieval in MVP"],
        }
