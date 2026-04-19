from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    vector: list[float]


class EmbeddingProvider:
    def __init__(self, enabled: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.enabled = enabled
        self.model_name = model_name
        self._model: Any | None = None
        self._st_module: Any | None = None
        self.available = bool(enabled)
        self.dim = 0
        if enabled:
            try:
                import sentence_transformers  # type: ignore

                self._st_module = sentence_transformers
            except Exception:
                self.available = False
                logger.warning("sentence-transformers unavailable; falling back to BM25-only retrieval")

    def is_available(self) -> bool:
        return self.enabled and self.available

    def _ensure_model(self) -> bool:
        if not self.is_available():
            return False
        if self._model is not None:
            return True
        try:
            SentenceTransformer = getattr(self._st_module, "SentenceTransformer")
            self._model = SentenceTransformer(self.model_name)
            # all-MiniLM-L6-v2 -> 384
            if hasattr(self._model, "get_embedding_dimension"):
                self.dim = int(self._model.get_embedding_dimension())
            else:
                self.dim = int(self._model.get_sentence_embedding_dimension())
            return True
        except Exception as exc:
            self.available = False
            logger.warning("failed to load sentence-transformers model %s: %s", self.model_name, exc)
            return False

    def embed_text(self, text: str) -> list[float] | None:
        if not self._ensure_model():
            return None
        try:
            vec = self._model.encode(text or "", convert_to_numpy=True, normalize_embeddings=True)
            return [float(x) for x in vec.tolist()]
        except Exception as exc:
            logger.warning("embed_text failed: %s", exc)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        if not self._ensure_model():
            return None
        if not texts:
            return []
        try:
            vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [[float(x) for x in row.tolist()] for row in vecs]
        except Exception as exc:
            logger.warning("embed_batch failed: %s", exc)
            return None

    def embed_one(self, text: str) -> EmbeddingResult:
        vec = self.embed_text(text)
        return EmbeddingResult(text=text, vector=vec or [])

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        vecs = self.embed_batch(texts)
        if vecs is None:
            return [EmbeddingResult(text=t, vector=[]) for t in texts]
        return [EmbeddingResult(text=t, vector=v) for t, v in zip(texts, vecs)]

    @staticmethod
    def similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = math.sqrt(sum(a * a for a in vec_a))
        nb = math.sqrt(sum(b * b for b in vec_b))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(dot / (na * nb))

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return EmbeddingProvider.similarity(a, b)

    @staticmethod
    def to_json(vector: list[float]) -> str:
        return json.dumps(vector, separators=(",", ":"))

    @staticmethod
    def from_json(raw: str) -> list[float]:
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return [float(x) for x in obj]
        except Exception:
            pass
        return []
