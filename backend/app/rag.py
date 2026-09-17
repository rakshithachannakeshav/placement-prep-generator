"""
Retrieval pipeline for the report generator.

Uses TF-IDF + cosine similarity instead of neural embeddings on purpose:
it needs no external API call and no model download, so it works with
zero extra cost or setup beyond `pip install scikit-learn`. This is a
legitimate, commonly-used retrieval approach for small/medium document
sets. If you outgrow it, swap `TfidfRetriever` for a sentence-transformers
or Groq/OpenAI-embeddings-backed retriever without touching the graph.
"""
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker over raw text."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            break
    return chunks


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class TfidfRetriever:
    """Fit once on all chunks from all source documents, then query."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._chunks: list[str] = []
        self._sources: list[str] = []

    def build(self, documents: dict[str, str]) -> None:
        """documents: {source_name: raw_text}"""
        self._chunks, self._sources = [], []
        for source, text in documents.items():
            for c in chunk_text(text):
                self._chunks.append(c)
                self._sources.append(source)

        if not self._chunks:
            raise ValueError("No text to index — check that documents aren't empty.")

        self._matrix = self._vectorizer.fit_transform(self._chunks)

    def query(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("Call build() before query().")
        q_vec = self._vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [
            RetrievedChunk(text=self._chunks[i], source=self._sources[i], score=float(sims[i]))
            for i in top_idx
            if sims[i] > 0
        ]
