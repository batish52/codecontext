from __future__ import annotations

import hashlib
import re
import time

from .bm25 import BM25Scorer
from .config import AppConfig
from .db import connect
from .embeddings import EmbeddingProvider
from .ranking import entrypoint_penalty, filetype_intent_boost, memory_file_penalty, wants_exact_symbol_body
from .redaction import looks_secret_path, redact_text
from .reranker import Reranker
from .git_context import GitContext
from .utils import approx_tokens, clamp_text, read_text_safely


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
SYMBOL_LIKE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
END_LINE_RE = re.compile(r"end_line=(\d+)")
GIT_HINT_RE = re.compile(r"\b(changed|recent|blame|commit|diff|git|who modified|last change)\b", re.IGNORECASE)
FILE_IN_QUERY_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:py|ts|js|json|md|yaml|yml|toml|ini)")


class SearchEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.reranker = Reranker(enabled=config.enable_reranker)
        self.embeddings = EmbeddingProvider(enabled=config.enable_embeddings)

    def _terms(self, text: str) -> list[str]:
        return [t.lower() for t in WORD_RE.findall(text)]

    def _query_symbol_candidates(self, query: str) -> list[str]:
        lowered = query.lower()
        found = [m.group(0).lower() for m in SYMBOL_LIKE_RE.finditer(lowered)]
        return list(dict.fromkeys(found))

    def _symbol_match_strength(self, query_symbol: str, symbol_name: str) -> int:
        if query_symbol == symbol_name:
            return 3
        if query_symbol.endswith(symbol_name) and query_symbol.count('.') == symbol_name.count('.'):
            return 2
        if '.' in query_symbol and symbol_name == query_symbol.rsplit('.', 1)[-1]:
            return 1
        return 0

    def _expand_line_window(self, path, start_line: int, end_line: int) -> dict:
        if not path.exists():
            return {"line": start_line, "end_line": end_line, "text": ""}
        lines = read_text_safely(path, self.config.max_file_bytes).splitlines()
        max_lines = self.config.max_merged_chunk_lines
        start = max(1, start_line)
        end = max(start, end_line)
        if (end - start + 1) > max_lines:
            end = start + max_lines - 1
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        return {
            "line": start,
            "end_line": end,
            "text": "\n".join(lines[start_idx:end_idx]),
        }

    def _merge_adjacent_matches(self, matches: list[dict], path) -> list[dict]:
        if not matches:
            return []
        matches = sorted(matches, key=lambda m: (m.get("line", 1), m.get("end_line", m.get("line", 1))))
        merged = [dict(matches[0])]
        gap = self.config.adjacent_chunk_merge_gap
        max_lines = self.config.max_merged_chunk_lines
        for match in matches[1:]:
            last = merged[-1]
            proposed_end = max(last.get("end_line", last["line"]), match.get("end_line", match["line"]))
            if match["line"] <= last.get("end_line", last["line"]) + gap and (proposed_end - last["line"] + 1) <= max_lines:
                window = self._expand_line_window(path, last["line"], proposed_end)
                last["end_line"] = window["end_line"]
                last["text"] = window["text"]
                last["semantic_score"] = max(last.get("semantic_score", 0.0), match.get("semantic_score", 0.0))
            else:
                merged.append(dict(match))
        return merged[: self.config.max_chunks_per_file]

    def _symbol_candidates(self, conn, qterms: list[str], query: str) -> dict[str, dict]:
        if not qterms:
            return {}
        exact_body = wants_exact_symbol_body(query)
        query_symbols = self._query_symbol_candidates(query)
        rows = conn.execute("SELECT path, symbol_name, symbol_type, line, doc FROM symbols").fetchall()
        scored: dict[str, dict] = {}
        for row in rows:
            symbol_name = (row["symbol_name"] or "").lower()
            exact_strength = max((self._symbol_match_strength(s, symbol_name) for s in query_symbols), default=0)
            exact_symbol_hits = 1 if exact_strength > 0 else 0
            generic_hits = sum(1 for t in qterms if len(t) >= 4 and t in symbol_name)
            if exact_strength == 0:
                generic_hits = sum(1 for t in qterms if len(t) >= 6 and t in symbol_name and t not in {"exact", "function", "method", "helper", "logic", "body"})
            hits = generic_hits + (exact_strength * 4)
            if hits <= 0:
                continue
            path = row["path"]
            full_path = self.config.root / path
            line = int(row["line"] or 1)
            doc = row["doc"] or ""
            end_match = END_LINE_RE.search(doc)
            end_line = int(end_match.group(1)) if end_match else line
            if exact_body and end_line >= line:
                window = self._expand_line_window(full_path, line, end_line)
                symbol_bonus = hits * self.config.exact_symbol_body_bonus
                if exact_strength >= 2:
                    symbol_bonus += self.config.exact_symbol_body_bonus * 3
                elif exact_symbol_hits:
                    symbol_bonus += self.config.exact_symbol_body_bonus
            else:
                radius = self.config.symbol_context_radius
                window = self._expand_line_window(full_path, max(1, line - radius), line + radius)
                symbol_bonus = hits * 4.0
            entry = scored.setdefault(path, {"path": path, "symbol_score": 0.0, "matches": [], "best_exact_strength": 0})
            entry["symbol_score"] += symbol_bonus
            entry["best_exact_strength"] = max(entry.get("best_exact_strength", 0), exact_strength)
            entry["matches"].append(
                {
                    "line": window["line"],
                    "end_line": window["end_line"],
                    "text": window["text"],
                    "semantic_score": 0.0,
                }
            )
        return scored

    def _bm25_candidates(self, rows, qterms: list[str], query: str) -> list[dict]:
        docs = []
        doc_rows = []
        for row in rows:
            if row["is_binary"]:
                continue
            doc_terms = self._terms(
                " ".join(filter(None, [row["path"], row["summary_tiny"], row["summary_detailed"], row["summary_symbols"]]))
            )
            docs.append(doc_terms)
            doc_rows.append(row)
        scorer = BM25Scorer(docs)
        scored = []
        for row, doc_terms in zip(doc_rows, docs):
            bm25_score = scorer.score(qterms, doc_terms)
            path_bonus = sum(4 for t in qterms if t in row["path"].lower())
            symbol_bonus = sum(2 for t in qterms if t in (row["summary_symbols"] or "").lower())
            intent_boost = filetype_intent_boost(row["path"], query)
            lexical_score = bm25_score + path_bonus + symbol_bonus + int(row["importance"]) + intent_boost
            if lexical_score <= 0:
                continue
            scored.append(
                {
                    "path": row["path"],
                    "summary": row["summary_tiny"],
                    "lexical_score": round(lexical_score, 6),
                    "bm25_score": round(bm25_score, 6),
                    "path_bonus": path_bonus,
                    "symbol_bonus": symbol_bonus,
                    "intent_boost": intent_boost,
                    "symbol_score": 0.0,
                    "semantic_score": 0.0,
                    "chunk_semantic_score": 0.0,
                    "importance": int(row["importance"]),
                    "matches": [],
                }
            )
        return scored

    def _semantic_file_candidates(self, conn, query: str) -> dict[str, dict]:
        if not self.embeddings.is_available():
            return {}
        qvec = self.embeddings.embed_one(query).vector
        rows = conn.execute(
            "SELECT e.path, e.vector_json, f.summary_tiny, f.importance FROM embeddings e JOIN files f ON f.path = e.path WHERE f.is_binary = 0"
        ).fetchall()
        scored: dict[str, dict] = {}
        for row in rows:
            vec = self.embeddings.from_json(row["vector_json"])
            similarity = self.embeddings.cosine(qvec, vec)
            if similarity <= 0.015:
                continue
            scored[row["path"]] = {
                "path": row["path"],
                "summary": row["summary_tiny"],
                "lexical_score": 0.0,
                "bm25_score": 0.0,
                "path_bonus": 0.0,
                "symbol_bonus": 0.0,
                "intent_boost": 0.0,
                "symbol_score": 0.0,
                "semantic_score": round(similarity, 6),
                "chunk_semantic_score": 0.0,
                "importance": int(row["importance"]),
                "matches": [],
            }
        return scored

    def _semantic_chunk_candidates(self, conn, query: str) -> dict[str, dict]:
        if not self.embeddings.is_available():
            return {}
        qvec = self.embeddings.embed_one(query).vector
        rows = conn.execute("SELECT path, start_line, end_line, vector_json FROM chunk_embeddings").fetchall()
        per_path: dict[str, list[dict]] = {}
        for row in rows:
            vec = self.embeddings.from_json(row["vector_json"])
            similarity = self.embeddings.cosine(qvec, vec)
            if similarity <= 0.02:
                continue
            full_path = self.config.root / row["path"]
            window = self._expand_line_window(full_path, int(row["start_line"]), int(row["end_line"]))
            per_path.setdefault(row["path"], []).append(
                {
                    "line": window["line"],
                    "end_line": window["end_line"],
                    "text": window["text"],
                    "semantic_score": round(similarity, 6),
                }
            )
        best: dict[str, dict] = {}
        for path, matches in per_path.items():
            matches.sort(key=lambda m: (-m["semantic_score"], m["line"]))
            merged = self._merge_adjacent_matches(matches[: self.config.max_chunk_candidates], self.config.root / path)
            best_score = max(m.get("semantic_score", 0.0) for m in merged) if merged else 0.0
            best[path] = {"path": path, "chunk_semantic_score": round(best_score, 6), "matches": merged}
        items = sorted(best.values(), key=lambda x: (-x["chunk_semantic_score"], x["path"]))[: self.config.max_chunk_candidates]
        return {item["path"]: item for item in items}

    def _overlap_ratio(self, a: dict, b: dict) -> float:
        a1, a2 = a.get("line", 1), a.get("end_line", a.get("line", 1))
        b1, b2 = b.get("line", 1), b.get("end_line", b.get("line", 1))
        overlap = max(0, min(a2, b2) - max(a1, b1) + 1)
        denom = max(1, min(a2 - a1 + 1, b2 - b1 + 1))
        return overlap / denom

    def _wants_git_context(self, query: str) -> bool:
        return bool(GIT_HINT_RE.search(query or ""))

    def _query_file_hint(self, query: str) -> str | None:
        m = FILE_IN_QUERY_RE.search(query or "")
        if not m:
            return None
        return m.group(0).replace('\\', '/')

    def _get_or_create_chunk_embedding(self, conn, path: str, start_line: int, end_line: int, chunk_text: str) -> list[float] | None:
        if not self.embeddings.is_available() or not chunk_text.strip():
            return None
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        row = conn.execute(
            "SELECT vector_json FROM chunk_embeddings WHERE path = ? AND start_line = ? AND end_line = ? AND chunk_hash = ? LIMIT 1",
            (path, int(start_line), int(end_line), chunk_hash),
        ).fetchone()
        if row and row["vector_json"]:
            return self.embeddings.from_json(row["vector_json"])

        vec = self.embeddings.embed_text(chunk_text)
        if not vec:
            return None

        try:
            with conn:
                conn.execute(
                    "DELETE FROM chunk_embeddings WHERE path = ? AND start_line = ? AND end_line = ? AND chunk_hash <> ?",
                    (path, int(start_line), int(end_line), chunk_hash),
                )
                conn.execute(
                    "INSERT INTO chunk_embeddings(path, start_line, end_line, chunk_hash, model, dim, vector_json, chunk_text, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        path,
                        int(start_line),
                        int(end_line),
                        chunk_hash,
                        self.embeddings.model_name,
                        len(vec),
                        self.embeddings.to_json(vec),
                        chunk_text,
                        time.time(),
                    ),
                )
        except Exception:
            pass
        return vec

    def search_project(self, query: str, top_k: int = 8) -> dict:
        conn = connect(self.config.db_path)
        rows = conn.execute(
            "SELECT path, summary_tiny, summary_detailed, summary_symbols, importance, is_binary FROM files"
        ).fetchall()
        qterms = self._terms(query)
        lexical = self._bm25_candidates(rows, qterms, query)
        semantic_files = self._semantic_file_candidates(conn, query)
        semantic_chunks = self._semantic_chunk_candidates(conn, query)
        symbol_hits = self._symbol_candidates(conn, qterms, query)

        merged: dict[str, dict] = {item["path"]: item for item in lexical}
        for path, sym in symbol_hits.items():
            if path not in merged:
                file_row = next((r for r in rows if r["path"] == path), None)
                merged[path] = {
                    "path": path,
                    "summary": file_row["summary_tiny"] if file_row else path,
                    "lexical_score": 0.0,
                    "bm25_score": 0.0,
                    "path_bonus": 0.0,
                    "symbol_bonus": 0.0,
                    "intent_boost": filetype_intent_boost(path, query),
                    "symbol_score": 0.0,
                    "best_exact_strength": 0,
                    "semantic_score": 0.0,
                    "chunk_semantic_score": 0.0,
                    "importance": int(file_row["importance"]) if file_row else 1,
                    "matches": [],
                }
            merged[path]["symbol_score"] = merged[path].get("symbol_score", 0.0) + sym["symbol_score"]
            merged[path]["best_exact_strength"] = max(merged[path].get("best_exact_strength", 0), sym.get("best_exact_strength", 0))
            merged[path]["matches"] = (merged[path].get("matches", []) + sym["matches"])[: self.config.max_chunk_candidates]
        for path, sem in semantic_files.items():
            if path in merged:
                merged[path]["semantic_score"] = sem["semantic_score"]
            else:
                merged[path] = sem
        for path, semc in semantic_chunks.items():
            if path in merged:
                merged[path]["chunk_semantic_score"] = semc["chunk_semantic_score"]
                merged[path]["matches"] = (merged[path].get("matches", []) + semc["matches"])[: self.config.max_chunk_candidates]
            else:
                file_row = next((r for r in rows if r["path"] == path), None)
                merged[path] = {
                    "path": path,
                    "summary": file_row["summary_tiny"] if file_row else path,
                    "lexical_score": 0.0,
                    "bm25_score": 0.0,
                    "path_bonus": 0.0,
                    "symbol_bonus": 0.0,
                    "intent_boost": filetype_intent_boost(path, query),
                    "symbol_score": 0.0,
                    "best_exact_strength": 0,
                    "semantic_score": 0.0,
                    "chunk_semantic_score": semc["chunk_semantic_score"],
                    "importance": int(file_row["importance"]) if file_row else 1,
                    "matches": semc["matches"],
                }

        combined = []
        for item in merged.values():
            dedup_penalty = 0.0
            if item.get("matches"):
                uniq = []
                for m in sorted(item["matches"], key=lambda x: (-x.get("semantic_score", 0.0), x.get("line", 1))):
                    if any(self._overlap_ratio(m, seen) > 0.6 for seen in uniq):
                        dedup_penalty += self.config.novelty_overlap_penalty
                        continue
                    uniq.append(m)
                item["matches"] = uniq[: self.config.max_chunk_candidates]
            suppression = entrypoint_penalty(item["path"], query) * self.config.entrypoint_suppression_penalty
            suppression += memory_file_penalty(item["path"], query) * 10.0
            if wants_exact_symbol_body(query) and item.get("symbol_score", 0.0) <= 0:
                suppression += 6.0
            elif wants_exact_symbol_body(query) and item.get("symbol_score", 0.0) < self.config.exact_symbol_body_bonus:
                suppression += 3.0
            if wants_exact_symbol_body(query) and item.get("best_exact_strength", 0) == 0:
                suppression += 12.0
            elif wants_exact_symbol_body(query) and item.get("best_exact_strength", 0) == 1:
                suppression += 4.0
            combined_score = (
                item["lexical_score"]
                + item.get("symbol_score", 0.0)
                + (item["semantic_score"] * 8.0)
                + (item["chunk_semantic_score"] * 22.0)
                + (item.get("intent_boost", 0.0) * 1.5)
                + (item["importance"] * 0.15)
                - dedup_penalty
                - suppression
            )
            item["score"] = round(combined_score, 6)
            if item["score"] >= self.config.min_combined_score:
                combined.append(item)

        # Semantic rerank pass over BM25 candidates (graceful fallback to BM25-only)
        if self.embeddings.available:
            query_embedding = self.embeddings.embed_text(query)
        else:
            query_embedding = None

        if query_embedding:
            max_bm25 = max((float(item.get("bm25_score", 0.0)) for item in combined), default=0.0)
            semantic_by_path: dict[str, float] = {}
            for item in combined:
                matches = item.get("matches") or []
                snippet = next((m for m in matches if (m.get("text") or "").strip()), None)
                if not snippet:
                    semantic_by_path[item.get("path", "")] = 0.0
                    continue
                snippet_text = snippet.get("text") or ""
                start_line = int(snippet.get("line", 1) or 1)
                end_line = int(snippet.get("end_line", start_line) or start_line)
                vec = self._get_or_create_chunk_embedding(
                    conn,
                    item.get("path", ""),
                    start_line,
                    end_line,
                    snippet_text,
                )
                semantic_by_path[item.get("path", "")] = self.embeddings.similarity(query_embedding, vec or []) if vec else 0.0

            for item in combined:
                bm25_score = float(item.get("bm25_score", 0.0))
                bm25_norm = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
                semantic_score = float(semantic_by_path.get(item.get("path", ""), 0.0))
                item["semantic_score"] = round(semantic_score, 6)
                item["score"] = round((0.6 * bm25_norm) + (0.4 * semantic_score), 6)

        top = sorted(combined, key=lambda x: (-x.get("score", 0), x.get("path", "")))[:top_k]

        query_symbols = self._query_symbol_candidates(query)
        has_exact = any(item.get("best_exact_strength", 0) > 0 for item in top)
        if query_symbols and has_exact:
            existing_paths = {item.get("path") for item in top}
            supplemental = []
            placeholders = ",".join("?" for _ in query_symbols)
            cg_rows = conn.execute(
                f"SELECT caller_path, caller_name, callee_path, callee_name FROM call_graph WHERE callee_name IN ({placeholders}) OR caller_name IN ({placeholders})",
                tuple(query_symbols + query_symbols),
            ).fetchall()
            for row in cg_rows:
                rel_path = row["caller_path"] or row["callee_path"]
                if not rel_path or rel_path in existing_paths:
                    continue
                full = self.config.root / rel_path
                if not full.exists():
                    continue
                line = 1
                text = self._expand_line_window(full, line, min(12, line + 8)).get("text", "")
                supplemental.append(
                    {
                        "path": rel_path,
                        "summary": f"call-graph relation for {row['caller_name']} -> {row['callee_name']}",
                        "lexical_score": 0.0,
                        "bm25_score": 0.0,
                        "path_bonus": 0.0,
                        "symbol_bonus": 0.0,
                        "intent_boost": 0.0,
                        "symbol_score": 0.0,
                        "best_exact_strength": 0,
                        "semantic_score": 0.0,
                        "chunk_semantic_score": 0.0,
                        "importance": 1,
                        "score": max(self.config.min_combined_score, 3.6),
                        "matches": [
                            {
                                "line": line,
                                "end_line": min(12, line + 8),
                                "text": text,
                                "semantic_score": 0.0,
                                "evidence_type": "call_graph_relation",
                            }
                        ],
                        "evidence_type": "call_graph_relation",
                    }
                )
                existing_paths.add(rel_path)
            if supplemental:
                top.extend(supplemental[: max(1, min(3, top_k - len(top) if top_k > len(top) else 1))])

        if self._wants_git_context(query):
            git_ctx = GitContext(self.config.root)
            if git_ctx.is_available():
                file_hint = self._query_file_hint(query)
                git_items = []
                if file_hint:
                    logs = git_ctx.file_log(file_hint, n=3)
                    if logs:
                        git_items.append({
                            "path": file_hint,
                            "summary": f"recent commits for {file_hint}",
                            "lexical_score": 0.0,
                            "bm25_score": 0.0,
                            "path_bonus": 0.0,
                            "symbol_bonus": 0.0,
                            "intent_boost": 0.0,
                            "symbol_score": 0.0,
                            "semantic_score": 0.0,
                            "chunk_semantic_score": 0.0,
                            "importance": 1,
                            "score": max(self.config.min_combined_score, 3.4),
                            "matches": [
                                {
                                    "line": 1,
                                    "end_line": 1,
                                    "text": "\n".join(f"{c['hash'][:8]} {c['author']} {c['date']} {c['message']}" for c in logs),
                                    "semantic_score": 0.0,
                                    "evidence_type": "git_context",
                                }
                            ],
                            "evidence_type": "git_context",
                        })
                    if "blame" in (query or "").lower() or "who modified" in (query or "").lower() or "last change" in (query or "").lower():
                        start_line = 1
                        end_line = 40
                        for item in top:
                            if item.get("path") == file_hint and item.get("matches"):
                                start_line = int(item["matches"][0].get("line", 1) or 1)
                                end_line = int(item["matches"][0].get("end_line", start_line) or start_line)
                                break
                        blame = git_ctx.blame_summary(file_hint, start_line=start_line, end_line=end_line)
                        if blame:
                            git_items.append({
                                "path": file_hint,
                                "summary": f"blame summary for {file_hint}:{start_line}-{end_line}",
                                "lexical_score": 0.0,
                                "bm25_score": 0.0,
                                "path_bonus": 0.0,
                                "symbol_bonus": 0.0,
                                "intent_boost": 0.0,
                                "symbol_score": 0.0,
                                "semantic_score": 0.0,
                                "chunk_semantic_score": 0.0,
                                "importance": 1,
                                "score": max(self.config.min_combined_score, 3.5),
                                "matches": [
                                    {
                                        "line": start_line,
                                        "end_line": end_line,
                                        "text": "\n".join(f"L{b['line']}: {b['author']} {b['commit_hash'][:8]} {b['content']}" for b in blame[:20]),
                                        "semantic_score": 0.0,
                                        "evidence_type": "git_context",
                                    }
                                ],
                                "evidence_type": "git_context",
                            })
                else:
                    recent = git_ctx.recent_commits(n=5)
                    if recent:
                        git_items.append({
                            "path": ".git",
                            "summary": "recent repository commits",
                            "lexical_score": 0.0,
                            "bm25_score": 0.0,
                            "path_bonus": 0.0,
                            "symbol_bonus": 0.0,
                            "intent_boost": 0.0,
                            "symbol_score": 0.0,
                            "semantic_score": 0.0,
                            "chunk_semantic_score": 0.0,
                            "importance": 1,
                            "score": max(self.config.min_combined_score, 3.2),
                            "matches": [
                                {
                                    "line": 1,
                                    "end_line": 1,
                                    "text": "\n".join(f"{c['hash'][:8]} {c['author']} {c['date']} {c['message']}" for c in recent),
                                    "semantic_score": 0.0,
                                    "evidence_type": "git_context",
                                }
                            ],
                            "evidence_type": "git_context",
                        })
                if git_items:
                    top.extend(git_items[:2])

        snippets = []
        for item in top:
            path = self.config.root / item["path"]
            if not path.exists():
                continue
            text = read_text_safely(path, self.config.max_file_bytes)
            if looks_secret_path(path):
                text = redact_text(text)
            lines = text.splitlines()
            matches = list(item.get("matches", []))
            if not matches and item["lexical_score"] > 0:
                for i, line in enumerate(lines):
                    low = line.lower()
                    if any(term in low for term in qterms):
                        start = max(1, i + 1 - 2)
                        end = min(len(lines), i + 3)
                        window = self._expand_line_window(path, start, end)
                        matches.append({**window, "semantic_score": 0.0})
                    if len(matches) >= self.config.max_chunks_per_file:
                        break
            if not matches and (item["semantic_score"] > 0.04 or item["chunk_semantic_score"] > 0.04):
                window = self._expand_line_window(path, 1, min(5, len(lines)))
                matches.append({**window, "semantic_score": 0.0})
            item["matches"] = self._merge_adjacent_matches(matches, path)[: self.config.max_chunks_per_file]
            if wants_exact_symbol_body(query) and item.get("best_exact_strength", 0) >= 2:
                item["matches"] = [m for m in item["matches"] if m.get("line", 1) > 1 or m.get("end_line", 1) > 40] or item["matches"]
            for m in item["matches"]:
                m["text"] = clamp_text(m["text"], self.config.context_snippet_char_limit)
            item["approx_tokens"] = approx_tokens("\n".join(m["text"] for m in item["matches"]))
            snippets.append(item)
        return {"query": query, "retrieval": "hybrid-chunked-bm25-symbol-novelty", "results": snippets}
