from __future__ import annotations

import math
from collections import Counter


class BM25Scorer:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = {}
        self.doc_lens = [len(doc) for doc in docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        for doc in docs:
            seen = set(doc)
            for term in seen:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], doc_terms: list[str]) -> float:
        tf = Counter(doc_terms)
        dl = len(doc_terms)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            numer = freq * (self.k1 + 1)
            denom = freq + self.k1 * (1 - self.b + self.b * (dl / max(1.0, self.avgdl)))
            score += self.idf(term) * (numer / max(denom, 1e-9))
        return score
