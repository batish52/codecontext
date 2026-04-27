from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embeddings import EmbeddingProvider


class Reranker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def rerank(
        self,
        query_embedding: list[float] | None,
        candidates: list[dict],
        embedding_provider: "EmbeddingProvider",
        top_k: int = 10,
    ) -> list[dict]:
        if (not self.enabled) or (query_embedding is None) or (not embedding_provider.is_available()):
            return candidates

        ranked: list[dict] = []
        for candidate in candidates:
            row = dict(candidate)
            vec = None
            raw_vec = row.get("vector_json")
            if isinstance(raw_vec, str) and raw_vec:
                try:
                    parsed = json.loads(raw_vec)
                    if isinstance(parsed, list):
                        vec = [float(x) for x in parsed]
                except Exception:
                    vec = None
            elif isinstance(raw_vec, list):
                vec = [float(x) for x in raw_vec]

            if vec is None:
                snippet_text = str(row.get("snippet_text") or "")
                vec = embedding_provider.embed_text(snippet_text) if snippet_text else None

            score = embedding_provider.similarity(query_embedding, vec or []) if vec else 0.0
            row["semantic_score"] = float(score)
            ranked.append(row)

        ranked.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
        return ranked[:top_k]
