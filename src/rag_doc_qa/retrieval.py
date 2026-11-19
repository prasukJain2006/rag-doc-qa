from __future__ import annotations

import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    """
    TF-IDF + cosine similarity retriever.

    Honest limitation, stated in the README too: TF-IDF matches keywords,
    not meaning. It is a fast, dependency-light baseline that works well
    on this small, terminology-rich corpus. For paraphrased questions,
    swap in sentence-transformers embeddings (see README 'Next step')
    and keep the same top-k + citation interface.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 3) -> tuple[list[RetrievedChunk], float]:
        start = time.perf_counter()
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in ranked[:top_k]]
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms
