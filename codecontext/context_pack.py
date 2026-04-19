from __future__ import annotations

import hashlib
import json
import re
import time

from .config import AppConfig
from .db import connect
from .metrics import Metrics
from .search import SearchEngine
from .utils import approx_tokens, clamp_text
from .ranking import wants_exact_symbol_body


LOCAL_IMPORT_RE = re.compile(r"from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import|import\s+\.([A-Za-z_][A-Za-z0-9_]*)")


class ContextPackBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.search = SearchEngine(config)

    def _is_explain_style(self, goal: str) -> bool:
        terms = set(goal.lower().replace('/', ' ').replace('-', ' ').split())
        return bool(terms & {'explain', 'architecture', 'overview', 'how', 'why', 'design'}) and not bool(terms & {'exact', 'body', 'defined', 'definition'})

    def _evidence_priority(self, item: dict) -> int:
        order = {
            'exact_symbol_body': 5,
            'lexical_hit': 4,
            'semantic_chunk': 3,
            'summary_fallback': 1,
        }
        role_bonus = 2 if item.get('evidence_role') == 'core_implementation' else 0
        return order.get(item.get('evidence_type', ''), 0) + role_bonus

    def _text_similarity(self, a: str, b: str) -> float:
        aset = set(a.lower().split())
        bset = set(b.lower().split())
        if not aset or not bset:
            return 0.0
        inter = len(aset & bset)
        denom = max(1, min(len(aset), len(bset)))
        return inter / denom

    def _is_searchengine_mechanism(self, goal: str) -> bool:
        lowered = goal.lower()
        return any(phrase in lowered for phrase in ('explain how', 'how is', 'how does', 'walk me through')) and 'searchengine' in lowered and any(term in lowered for term in ('bm25', 'symbol retrieval'))

    def _prune_evidence(self, evidence: list[dict], goal: str, preferred_support_path: str | None = None) -> tuple[list[dict], list[dict], dict]:
        explain_style = self._is_explain_style(goal)
        searchengine_mechanism = self._is_searchengine_mechanism(goal)
        ranked = sorted(
            evidence,
            key=lambda e: (
                0 if e.get('evidence_role') == 'core_implementation' else 1,
                0 if preferred_support_path and e.get('path') == preferred_support_path and e.get('evidence_role') == 'support_evidence' else 1,
                0 if searchengine_mechanism and e.get('evidence_role') == 'support_evidence' and (e.get('path') or '').replace('\\', '/').startswith('codecontext/') else 1,
                0 if searchengine_mechanism and e.get('evidence_role') == 'support_evidence' and (e.get('path') or '').replace('\\', '/').startswith('codecontext/') and e.get('evidence_type') == 'semantic_chunk' else 1,
                -self._evidence_priority(e), -float(e.get('score', 0)), len(e.get('snippet_text', ''))
            ),
        )
        kept: list[dict] = []
        dropped: list[dict] = []
        max_items = 5 if explain_style else max(7, self.config.max_chunks_per_file * 2)
        chars_before = sum(len(e.get('snippet_text', '')) for e in evidence)
        for item in ranked:
            text = item.get('snippet_text', '')
            duplicate = any(
                item.get('path') == prev.get('path') and self._text_similarity(text, prev.get('snippet_text', '')) > 0.72
                for prev in kept
            )
            if duplicate:
                dropped.append({**item, 'drop_reason': 'near-duplicate of already selected evidence'})
                continue
            if explain_style and kept:
                same_path_count = sum(1 for prev in kept if prev.get('path') == item.get('path'))
                if same_path_count >= 2 and self._evidence_priority(item) < 5:
                    dropped.append({**item, 'drop_reason': 'explain-style compaction limited repeated same-file evidence'})
                    continue
            if len(kept) >= max_items:
                dropped.append({**item, 'drop_reason': 'lower marginal value after compaction budget reached'})
                continue
            if searchengine_mechanism and item.get('evidence_role') == 'support_evidence' and preferred_support_path and item.get('path') != preferred_support_path and not (item.get('path', '').replace('\\', '/').startswith('codecontext/')):
                dropped.append({**item, 'drop_reason': 'searchengine mechanism compaction demoted unrelated support outside codecontext'})
                continue
            kept.append({**item, 'retained_reason': 'high-yield evidence survived compaction'})
        chars_after = sum(len(e.get('snippet_text', '')) for e in kept)
        debug = {
            'evidence_count_before_prune': len(evidence),
            'evidence_count_after_prune': len(kept),
            'compaction_chars_saved': max(0, chars_before - chars_after),
            'explain_style_compaction': explain_style,
            'pruned_items': [
                {
                    'path': d.get('path'),
                    'start_line': d.get('start_line'),
                    'end_line': d.get('end_line'),
                    'evidence_type': d.get('evidence_type'),
                    'drop_reason': d.get('drop_reason'),
                }
                for d in dropped[:12]
            ],
        }
        return kept, dropped, debug

    def _overlap_ratio(self, a: dict, b: dict) -> float:
        a1, a2 = a.get("line", 1), a.get("end_line", a.get("line", 1))
        b1, b2 = b.get("line", 1), b.get("end_line", b.get("line", 1))
        overlap = max(0, min(a2, b2) - max(a1, b1) + 1)
        denom = max(1, min(a2 - a1 + 1, b2 - b1 + 1))
        return overlap / denom

    def _candidate_role(self, result: dict, goal: str, existing: list[dict], top_path: str | None = None) -> tuple[str, str]:
        explain_style = self._is_explain_style(goal)
        if result.get('best_exact_strength', 0) >= 2:
            return 'core_implementation', 'direct implementation evidence for target behavior'
        if top_path and result.get('path') == top_path:
            return 'core_implementation', 'top-ranked implementation file for this request'
        if result.get('path', '').startswith('codecontext/search.py'):
            return 'core_implementation', 'direct implementation evidence for retrieval behavior'
        if explain_style:
            if not existing:
                return 'core_implementation', 'highest-ranked implementation candidate for explain-style request'
            if any(e.get('evidence_role') == 'core_implementation' for e in existing):
                return 'support_evidence', 'complementary supporting evidence for explain-style request'
        return 'support_evidence', 'supporting evidence selected from ranked candidates'

    def _path_module_stem(self, path: str) -> str:
        path = path.replace('\\', '/').split('/')[-1]
        return path.rsplit('.', 1)[0]

    def _imported_helper_candidates(self, conn, core_item: dict) -> list[dict]:
        core_text = core_item.get('snippet_text', '')
        modules = []
        for m in LOCAL_IMPORT_RE.finditer(core_text):
            mod = m.group(1) or m.group(2)
            if mod:
                modules.append(mod)
        helpers = []
        seen = set()
        for mod in modules:
            path = f'codecontext/{mod}.py'
            if path in seen:
                continue
            seen.add(path)
            row = conn.execute("SELECT path, summary_tiny, summary_detailed, summary_symbols, importance FROM files WHERE path = ?", (path,)).fetchone()
            if not row:
                continue
            snippet = (row['summary_detailed'] or row['summary_tiny'] or '')[: self.config.context_snippet_char_limit]
            helpers.append({
                'path': row['path'],
                'summary': row['summary_tiny'],
                'lexical_score': 1.0,
                'chunk_semantic_score': 0.0,
                'best_exact_strength': 0,
                'score': float(row['importance']) + 1.0,
                'matches': [{'line': 1, 'end_line': 1, 'text': snippet, 'semantic_score': 0.0}],
                'helper_imported_from_core': True,
            })
        return helpers

    def _support_linkage_score(self, core_item: dict, candidate_result: dict, goal: str = '') -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0
        core_path = core_item.get('path', '')
        cand_path = candidate_result.get('path', '')
        core_text = core_item.get('snippet_text', '')
        cand_texts = '\n'.join(m.get('text', '') for m in candidate_result.get('matches', []))
        core_stem = self._path_module_stem(core_path)
        cand_stem = self._path_module_stem(cand_path)
        lowered = goal.lower()
        mechanism_style = any(phrase in lowered for phrase in ('explain how', 'how does', 'walk me through', 'how is'))
        runtime_mechanism = mechanism_style and any(term in lowered for term in ('runtime', 'interceptor', 'plugin', 'work'))
        cand_norm = cand_path.replace('\\', '/')
        core_norm = core_path.replace('\\', '/')

        if core_path and cand_path and core_path.split('/')[0:1] == cand_path.split('/')[0:1]:
            score += 6.0
            reasons.append('same subsystem path family')
        if runtime_mechanism and ((core_norm.startswith('codecontext-runtime/') or core_norm.startswith('codecontext-runtime/')) and (cand_norm.startswith('codecontext-runtime/') or cand_norm.startswith('codecontext-runtime/') or cand_norm.startswith('codecontext/'))):
            score += 10.0
            reasons.append('same runtime mechanism family')
        if cand_stem and cand_stem in core_text:
            score += 10.0
            reasons.append('candidate module referenced in core snippet')
        if core_stem and core_stem in cand_texts:
            score += 8.0
            reasons.append('core module referenced in support snippet')
        query_terms = set(goal.lower().replace('/', ' ').replace('-', ' ').split())
        if candidate_result.get('helper_imported_from_core'):
            score += 14.0
            reasons.append('direct local import from core snippet')
            if cand_stem.lower() in query_terms:
                score += 12.0
                reasons.append('imported helper explicitly named in query')
        elif query_terms and cand_stem.lower() not in query_terms:
            score -= 6.0
            reasons.append('not a directly imported helper named by query')
        if candidate_result.get('lexical_score', 0) > 0:
            score += 4.0
            reasons.append('high-confidence lexical/code hit')
        if candidate_result.get('chunk_semantic_score', 0.0) > 0.08:
            score += 3.0
            reasons.append('strong semantic support chunk')
        if candidate_result.get('path') == core_path:
            score -= 20.0
            reasons.append('same file as core evidence penalized')
        similarity = self._text_similarity(core_text, cand_texts)
        if similarity > 0.55:
            overlap_penalty = 8.0
            if runtime_mechanism and (cand_norm.startswith('codecontext-runtime/') or cand_norm.startswith('codecontext-runtime/')):
                overlap_penalty = 3.0
                reasons.append('same-mechanism runtime support allowed despite overlap')
            score -= overlap_penalty
            reasons.append('low complementarity to core evidence')
        if '/services/' in cand_norm and '/codecontext/' in core_norm:
            score -= 5.0
            reasons.append('cross-domain support file penalized as loosely related')
        if mechanism_style and (cand_norm.endswith('.md') or cand_norm in {'requirements.txt', 'MEMORY.md'} or cand_norm.startswith('memory/')):
            score -= 18.0
            reasons.append('docs/meta support demoted for mechanism explanation')
        if mechanism_style and cand_norm in {'codecontext/telemetry.py', 'codecontext/outbound_schema.py'}:
            score -= 14.0
            reasons.append('reporting/schema support demoted for mechanism explanation')
        return score, reasons

    def _candidate_quality_score(self, result: dict, role: str, existing: list[dict]) -> float:
        score = float(result.get('score', 0))
        if role == 'core_implementation':
            score += 25.0
        if existing:
            same_path = sum(1 for e in existing if e.get('path') == result.get('path'))
            score -= same_path * 8.0
        if result.get('best_exact_strength', 0) >= 2:
            score += 30.0
        elif result.get('lexical_score', 0) > 0:
            score += 8.0
        elif result.get('chunk_semantic_score', 0.0) > 0:
            score += 4.0
        return score

    def _build_primary_evidence(self, ranked_results: list[dict], recent_hashes: set[str], token_budget: int, spent: int, metrics: Metrics, conn, goal: str) -> tuple[list[dict], int, list[dict], dict]:
        evidence = []
        pre_debug = []
        retention_debug = {
            'support_slot_reserved': False,
            'preferred_support_path': None,
            'chosen_support_survived': False,
            'support_retention_reason': None,
            'support_retention_exception_used': False,
        }
        reserved_support_path = None
        seen_ranges: dict[str, list[dict]] = {}
        ordered_results = sorted(
            ranked_results,
            key=lambda r: (-float(r.get('score', 0)), r.get('path', '')),
        )
        top_path = ordered_results[0].get('path') if ordered_results else None
        explain_style = self._is_explain_style(goal)
        support_debug = []
        if explain_style and ordered_results:
            scored_candidates = []
            for result in ordered_results:
                role, role_reason = self._candidate_role(result, goal, evidence, top_path=top_path)
                quality_score = self._candidate_quality_score(result, role, evidence)
                scored_candidates.append((quality_score, role, role_reason, result))
            scored_candidates.sort(key=lambda x: (-x[0], x[3].get('path', '')))
            core = next((item for item in scored_candidates if item[1] == 'core_implementation'), None)
            support = None
            reserved_support_path = None
            if core:
                core_preview = {
                    'path': core[3].get('path'),
                    'snippet_text': (core[3].get('matches') or [{}])[0].get('text', ''),
                }
                support_scored = []
                helper_candidates = self._imported_helper_candidates(conn, core_preview)
                has_imported_helpers = bool(helper_candidates)
                seen_support_paths = set()
                for _, _, _, result in scored_candidates:
                    if result.get('path') == core[3].get('path'):
                        continue
                    linkage_score, linkage_reasons = self._support_linkage_score(core_preview, result, goal)
                    total = float(result.get('score', 0)) + linkage_score
                    if has_imported_helpers and not result.get('helper_imported_from_core'):
                        total -= 12.0
                        linkage_reasons = list(linkage_reasons) + ['penalized because direct imported helpers are available']
                    support_scored.append((total, result, linkage_reasons))
                    seen_support_paths.add(result.get('path'))
                for result in helper_candidates:
                    if result.get('path') == core[3].get('path') or result.get('path') in seen_support_paths:
                        continue
                    linkage_score, linkage_reasons = self._support_linkage_score(core_preview, result, goal)
                    total = float(result.get('score', 0)) + linkage_score
                    support_scored.append((total, result, linkage_reasons))
                support_scored.sort(key=lambda x: (-x[0], x[1].get('path', '')))
                support = support_scored[0][1] if support_scored else None
                reserved_support_path = support.get('path') if support else None
                retention_debug['support_slot_reserved'] = bool(support)
                retention_debug['preferred_support_path'] = reserved_support_path
                support_debug = [
                    {
                        'path': cand.get('path'),
                        'support_score': round(total, 3),
                        'support_reasons': reasons,
                        'selected': bool(support is cand),
                    }
                    for total, cand, reasons in support_scored[:8]
                ]
            reordered = []
            if core:
                reordered.append(core[3])
            if support and support not in reordered:
                reordered.append(support)
            for _, _, _, result in scored_candidates:
                if result not in reordered:
                    reordered.append(result)
            ordered_results = reordered
        exact_body_mode = wants_exact_symbol_body(goal)
        exact_body_extra_context = self._allow_exact_body_extra_context(goal)
        exact_body_core_selected = False
        exact_body_debug = {
            'mode': exact_body_mode,
            'extra_context_requested': exact_body_extra_context,
            'snippet_history_exception_used': False,
            'non_body_snippet_reason': None,
        }
        for result in ordered_results:
            role, role_reason = self._candidate_role(result, goal, evidence, top_path=top_path)
            quality_score = self._candidate_quality_score(result, role, evidence)
            sorted_matches = self._prefer_exact_match_first(result, result.get("matches", []), goal)
            for match in sorted_matches[: self.config.max_chunks_per_file]:
                if self._is_explain_style(goal) and evidence and any(e.get('path') == result.get('path') for e in evidence) and role != 'core_implementation':
                    continue
                if exact_body_mode and exact_body_core_selected and not exact_body_extra_context:
                    exact_body_debug['non_body_snippet_reason'] = 'suppressed non-body support because exact-body core snippet was already sufficient'
                    continue
                if any(self._overlap_ratio(match, prev) > 0.65 for prev in seen_ranges.get(result["path"], [])):
                    continue
                block = clamp_text(match.get("text", ""), self.config.context_snippet_char_limit)
                if not block.strip():
                    continue
                snippet_hash = hashlib.sha256(f"{result['path']}:{match.get('line', 1)}:{block}".encode("utf-8")).hexdigest()
                cost = approx_tokens(block)
                metrics.inc("chunks_retrieved")
                allow_core_reuse = self._is_explain_style(goal) and role == 'core_implementation' and not evidence
                allow_exact_body_core_reuse = wants_exact_symbol_body(goal) and role == 'core_implementation' and not evidence and result.get('best_exact_strength', 0) >= 2
                allow_reserved_support_reuse = self._is_explain_style(goal) and role == 'support_evidence' and result.get('path') == reserved_support_path and not any(e.get('evidence_role') == 'support_evidence' for e in evidence)
                if snippet_hash in recent_hashes and not allow_core_reuse and not allow_exact_body_core_reuse and not allow_reserved_support_reuse:
                    continue
                if allow_exact_body_core_reuse and snippet_hash in recent_hashes:
                    exact_body_debug['snippet_history_exception_used'] = True
                explain_support_budget_reserve = self._is_explain_style(goal) and role == 'core_implementation' and reserved_support_path is not None
                effective_budget = token_budget - 220 if explain_support_budget_reserve else token_budget
                if spent + cost > effective_budget and not allow_reserved_support_reuse:
                    continue
                spent += cost
                metrics.inc("chunks_sent")
                seen_ranges.setdefault(result["path"], []).append(match)
                evidence_item = {
                    "path": result["path"],
                    "start_line": match.get("line", 1),
                    "end_line": match.get("end_line", match.get("line", 1)),
                    "symbol_name": result.get("symbol_name"),
                    "snippet_text": block,
                    "score": result.get("score", 0),
                    "selection_reason": role_reason + (' (snippet reuse allowed for top core evidence)' if allow_core_reuse and snippet_hash in recent_hashes else '') + (' (snippet reuse allowed for exact-body core evidence)' if allow_exact_body_core_reuse and snippet_hash in recent_hashes else '') + (' (reserved support slot retained with narrow exception)' if allow_reserved_support_reuse else ''),
                    "evidence_type": self._infer_evidence_type(result, match),
                    "evidence_role": role,
                    "snippet_hash": snippet_hash,
                    "candidate_quality_score": quality_score,
                }
                evidence.append(evidence_item)
                if exact_body_mode and evidence_item['evidence_type'] == 'exact_symbol_body' and not exact_body_core_selected:
                    exact_body_core_selected = True
                if exact_body_mode and evidence_item['evidence_type'] != 'exact_symbol_body' and exact_body_debug['non_body_snippet_reason'] is None:
                    exact_body_debug['non_body_snippet_reason'] = 'non-body snippet retained because extra context was explicitly requested'
                if role == 'support_evidence' and result.get('path') == reserved_support_path:
                    retention_debug['chosen_support_survived'] = True
                    retention_debug['support_retention_reason'] = 'reserved support slot kept one chosen support snippet'
                    retention_debug['support_retention_exception_used'] = bool(allow_reserved_support_reuse)
                pre_debug.append({
                    "path": evidence_item["path"],
                    "start_line": evidence_item["start_line"],
                    "end_line": evidence_item["end_line"],
                    "evidence_type": evidence_item["evidence_type"],
                    "evidence_role": evidence_item["evidence_role"],
                    "selection_reason": evidence_item["selection_reason"],
                    "candidate_quality_score": evidence_item["candidate_quality_score"],
                    "supports_distinct_file": not any(prev.get('path') == evidence_item['path'] for prev in evidence[:-1]),
                })
                with conn:
                    conn.execute(
                        "INSERT INTO snippet_history(path, line, snippet_hash, sent_at) VALUES(?,?,?,?)",
                        (result["path"], match.get("line", 1), snippet_hash, time.time()),
                    )
        if support_debug:
            pre_debug.append({'support_choice_debug': support_debug})
        if exact_body_mode:
            pre_debug.append({'exact_body_debug': exact_body_debug})
        if retention_debug['support_slot_reserved'] and not retention_debug['chosen_support_survived']:
            retention_debug['support_retention_reason'] = 'reserved support candidate did not survive quality/history/budget checks'
        return evidence, spent, pre_debug, retention_debug

    def _prefer_exact_match_first(self, result: dict, matches: list[dict], goal: str) -> list[dict]:
        if not wants_exact_symbol_body(goal) or result.get("best_exact_strength", 0) < 2:
            return sorted(matches, key=lambda m: (-m.get("semantic_score", 0.0), m.get("line", 1)))
        return sorted(
            matches,
            key=lambda m: (
                0 if m.get("line", 1) > 1 and "def " in (m.get("text", "") or "")[:120] else 1,
                -(m.get("end_line", m.get("line", 1)) - m.get("line", 1)),
                -m.get("semantic_score", 0.0),
                m.get("line", 1),
            ),
        )

    def _allow_exact_body_extra_context(self, goal: str) -> bool:
        lowered = goal.lower()
        return any(term in lowered for term in ("helper", "helpers", "nearby", "adjacent", "surrounding", "context", "and "))

    def _infer_evidence_type(self, result: dict, match: dict) -> str:
        if result.get("best_exact_strength", 0) >= 2:
            return "exact_symbol_body"
        if match.get("semantic_score", 0.0) > 0 and result.get("chunk_semantic_score", 0.0) > 0:
            return "semantic_chunk"
        if result.get("lexical_score", 0) > 0:
            return "lexical_hit"
        return "summary_fallback"

    def _fallback_evidence(self, search_result: dict, recent_hashes: set[str], token_budget: int, spent: int, metrics: Metrics, conn, allow_reuse: bool = False) -> tuple[list[dict], int, str]:
        results = search_result.get("results", [])
        if not results:
            return [], spent, "no search results available"

        fallback_reason = "used fallback evidence packing"
        evidence = []

        exact = [r for r in results if r.get("best_exact_strength", 0) >= 2]
        lexical = [r for r in results if r.get("lexical_score", 0) > 0]
        semantic = [r for r in results if r.get("chunk_semantic_score", 0.0) > 0]
        ordered = exact or lexical or semantic or results[:1]

        chosen_results = []
        for result in ordered:
            if not chosen_results:
                chosen_results.append(result)
                continue
            if result.get("path") != chosen_results[0].get("path"):
                chosen_results.append(result)
                break
        if not chosen_results:
            chosen_results = ordered[:1]

        for result in chosen_results[:2]:
            matches = result.get("matches", [])
            if matches:
                match = self._prefer_exact_match_first(result, matches, search_result.get("query", ""))[0]
                block = clamp_text(match.get("text", ""), self.config.context_snippet_char_limit)
                if not block.strip():
                    continue
                snippet_hash = hashlib.sha256(f"fallback:{result['path']}:{match.get('line', 1)}:{block}".encode("utf-8")).hexdigest()
                cost = approx_tokens(block)
                if (snippet_hash in recent_hashes and not allow_reuse) or spent + cost > token_budget:
                    continue
                spent += cost
                metrics.inc("chunks_sent")
                evidence.append(
                    {
                        "path": result["path"],
                        "start_line": match.get("line", 1),
                        "end_line": match.get("end_line", match.get("line", 1)),
                        "symbol_name": result.get("symbol_name"),
                        "snippet_text": block,
                        "score": result.get("score", 0),
                        "selection_reason": fallback_reason,
                        "evidence_type": self._infer_evidence_type(result, match),
                        "snippet_hash": snippet_hash,
                    }
                )
                continue

            summary_text = clamp_text(result.get("summary", ""), min(220, self.config.context_snippet_char_limit))
            if not summary_text.strip():
                continue
            snippet_hash = hashlib.sha256(f"summary:{result['path']}:{summary_text}".encode("utf-8")).hexdigest()
            cost = approx_tokens(summary_text)
            if (snippet_hash in recent_hashes and not allow_reuse) or spent + cost > token_budget:
                continue
            spent += cost
            evidence.append(
                {
                    "path": result["path"],
                    "start_line": 1,
                    "end_line": 1,
                    "symbol_name": result.get("symbol_name"),
                    "snippet_text": summary_text,
                    "score": result.get("score", 0),
                    "selection_reason": "summary fallback used because no snippet could be packed",
                    "evidence_type": "summary_fallback",
                    "snippet_hash": snippet_hash,
                }
            )
        return evidence, spent, fallback_reason

    def prepare(self, goal: str, top_k: int = 6, token_budget: int | None = None) -> dict:
        token_budget = token_budget or self.config.default_context_budget_tokens
        metrics = Metrics()
        conn = connect(self.config.db_path)
        project_row = conn.execute("SELECT value FROM project_state WHERE key='project_summary'").fetchone()
        project_summary = json.loads(project_row["value"]) if project_row else {}
        search_result = self.search.search_project(goal, top_k=top_k)
        spent = approx_tokens(goal) + approx_tokens(project_summary)
        recent_hashes = {
            row[0]
            for row in conn.execute(
                "SELECT snippet_hash FROM snippet_history ORDER BY sent_at DESC LIMIT ?",
                (self.config.do_not_resend_window,),
            ).fetchall()
        }

        ranked_results = sorted(search_result.get("results", []), key=lambda r: (-r.get("score", 0), r.get("path", "")))
        evidence, spent, pre_candidates, retention_debug = self._build_primary_evidence(ranked_results, recent_hashes, token_budget, spent, metrics, conn, goal)

        fallback_used = False
        initial_empty_reason = None
        fallback_rule = None
        if not evidence:
            initial_empty_reason = "primary evidence selection produced no packable snippets"
            evidence, spent, fallback_rule = self._fallback_evidence(search_result, recent_hashes, token_budget, spent, metrics, conn)
            fallback_used = bool(evidence)
        if not evidence:
            initial_empty_reason = (initial_empty_reason or "") + " | snippet history suppression or budget prevented all evidence"
            evidence, spent, forced_rule = self._fallback_evidence(search_result, recent_hashes, token_budget, spent, metrics, conn, allow_reuse=True)
            if evidence:
                fallback_used = True
                fallback_rule = forced_rule + " with snippet reuse allowed to prevent empty external payload"

        evidence_before_prune = list(evidence)
        evidence, dropped_evidence, prune_debug = self._prune_evidence(evidence, goal, preferred_support_path=retention_debug.get('preferred_support_path'))
        bytes_sent = sum(len(item["snippet_text"]) for item in evidence)
        metrics.inc("raw_bytes_avoided", max(0, 0 if not evidence else 14000 - bytes_sent))
        metrics.set("estimated_prompt_tokens", spent)
        metrics.set("estimated_saved_tokens", max(0, 14000 - spent))
        metrics.set("evidence_count", len(evidence))
        metrics.set("fallback_evidence_used", fallback_used)

        chosen_core_evidence = next(({
                    'path': k.get('path'),
                    'start_line': k.get('start_line'),
                    'end_line': k.get('end_line'),
                    'evidence_type': k.get('evidence_type'),
                    'selection_reason': k.get('selection_reason'),
                } for k in evidence if k.get('evidence_role') == 'core_implementation'), None)
        chosen_support_evidence = next(({
                    'path': k.get('path'),
                    'start_line': k.get('start_line'),
                    'end_line': k.get('end_line'),
                    'evidence_type': k.get('evidence_type'),
                    'selection_reason': k.get('selection_reason'),
                } for k in evidence if k.get('evidence_role') == 'support_evidence'), None)
        exact_body_debug = next((x.get('exact_body_debug') for x in pre_candidates if isinstance(x, dict) and x.get('exact_body_debug')), None)
        return {
            "goal": goal,
            "policy": {
                "send_minimum_context": True,
                "prefer_cached_summaries": True,
                "prefer_chunk_hits_over_whole_file_context": True,
                "avoid_full_files": True,
                "token_budget": token_budget,
                "do_not_ask_for_more_files_without_justification": True,
            },
            "project_summary": project_summary,
            "evidence": evidence,
            "debug": {
                "evidence_count": len(evidence),
                "fallback_used": fallback_used,
                "initial_empty_reason": initial_empty_reason,
                "fallback_rule": fallback_rule,
                "search_result_count": len(search_result.get("results", [])),
                "forced_evidence_reuse": bool(evidence) and bool(fallback_rule and "reuse allowed" in fallback_rule),
                **retention_debug,
                **prune_debug,
                "pre_compaction_candidates": pre_candidates,
                "support_choice_debug": next((x.get('support_choice_debug') for x in pre_candidates if isinstance(x, dict) and x.get('support_choice_debug')), []),
                "exact_body_debug": exact_body_debug,
                "chosen_core_evidence": chosen_core_evidence,
                "chosen_support_evidence": chosen_support_evidence,
                "exact_body_details": {
                    'selected_symbol_path': chosen_core_evidence.get('path') if chosen_core_evidence and chosen_core_evidence.get('evidence_type') == 'exact_symbol_body' else None,
                    'selected_body_line_range': [chosen_core_evidence.get('start_line'), chosen_core_evidence.get('end_line')] if chosen_core_evidence and chosen_core_evidence.get('evidence_type') == 'exact_symbol_body' else None,
                    'snippet_history_exception_used': bool(exact_body_debug and exact_body_debug.get('snippet_history_exception_used')),
                    'fallback_used': fallback_used,
                    'non_body_snippet_reason': exact_body_debug.get('non_body_snippet_reason') if exact_body_debug else None,
                    'estimated_token_size': spent,
                } if wants_exact_symbol_body(goal) else None,
                "retained_items": [
                    {
                        'path': k.get('path'),
                        'start_line': k.get('start_line'),
                        'end_line': k.get('end_line'),
                        'evidence_type': k.get('evidence_type'),
                        'evidence_role': k.get('evidence_role'),
                        'retained_reason': k.get('retained_reason', k.get('selection_reason')),
                    }
                    for k in evidence
                ],
            },
            "metrics": metrics.finish(),
        }
