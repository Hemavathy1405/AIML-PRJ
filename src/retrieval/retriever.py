"""
Evidence Retrieval Module
Performs knowledge chunk indexing and hybrid retrieval (lexical BM25/TF-IDF + semantic embeddings)
Reference: Lewis et al. (2020) RAG, Es et al. (2024) RAGAs
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..features.semantic_embeddings import compute_cosine_similarity

class EvidenceRetriever:
    """Indexes and retrieves reference evidence chunks for user queries."""

    def __init__(self, corpus_path: Optional[str] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        if corpus_path:
            self.load_corpus(corpus_path)

    def load_corpus(self, corpus_path: str):
        """Loads reference knowledge chunks from a JSON file."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self._build_index()

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Adds new chunks to the in-memory retrieval index."""
        self.chunks.extend(new_chunks)
        self._build_index()

    def _build_index(self):
        """Builds TF-IDF search index over chunk contents."""
        if not self.chunks:
            return
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: combines lexical match with semantic similarity.
        Returns top-k matching evidence chunks with retrieval score.
        """
        if not self.chunks or self.vectorizer is None:
            return []

        # 1. Lexical TF-IDF scores
        query_vec = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 2. Combine with semantic similarity
        results = []
        for idx, chunk in enumerate(self.chunks):
            lex_score = float(lexical_scores[idx])
            sem_score = compute_cosine_similarity(query, chunk["text"][:300])
            # Hybrid score
            hybrid_score = 0.55 * sem_score + 0.45 * lex_score
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chk_{idx}"),
                "title": chunk.get("title", f"Chunk {idx}"),
                "domain": chunk.get("domain", "General"),
                "text": chunk["text"],
                "score": round(float(hybrid_score), 4)
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_best_evidence(self, query: str) -> str:
        """Helper to get the single most relevant evidence passage text."""
        hits = self.retrieve(query, top_k=1)
        if hits:
            return hits[0]["text"]
        return "No reference evidence available."
