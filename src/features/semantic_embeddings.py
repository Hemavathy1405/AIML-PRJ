"""
Semantic Embedding & Consistency Module
Calculates semantic similarity and consistency across sampled generations / rephrasings
Reference: Manakul et al. (2023) SelfCheckGPT, Farquhar et al. (2024)
"""

import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try loading sentence-transformers if enabled and available
_ST_MODEL = None
_ST_LOAD_ATTEMPTED = False

def get_sentence_transformer():
    global _ST_MODEL, _ST_LOAD_ATTEMPTED
    if _ST_LOAD_ATTEMPTED:
        return _ST_MODEL
    _ST_LOAD_ATTEMPTED = True
    try:
        from config.config import USE_TRANSFORMER_BACKENDS
        if not USE_TRANSFORMER_BACKENDS:
            return None
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _ST_MODEL = None
    return _ST_MODEL

def compute_text_embedding(text: str) -> np.ndarray:
    """Computes dense or tf-idf normalized embedding for a text."""
    st = get_sentence_transformer()
    if st is not None:
        try:
            emb = st.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
            return emb
        except Exception:
            pass
    # Deterministic character-n-gram/word TF-IDF fallback for lightning fast offline execution
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=256)
    vec.fit([text, "placeholder text for vocabulary balance and stability"])
    res = vec.transform([text]).toarray()[0]
    norm = np.linalg.norm(res)
    return res / (norm + 1e-9)

def compute_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates cosine semantic similarity between two texts in [0, 1]."""
    st = get_sentence_transformer()
    if st is not None:
        try:
            embs = st.encode([text1, text2], show_progress_bar=False, normalize_embeddings=True)
            sim = float(np.dot(embs[0], embs[1]))
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            pass
    
    # Fast TF-IDF word/subword cosine similarity
    vec = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf = vec.fit_transform([text1, text2])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Fallback to Jaccard word overlap
        s1 = set(text1.lower().split())
        s2 = set(text2.lower().split())
        if not s1 or not s2:
            return 0.0
        return float(len(s1 & s2) / len(s1 | s2))

def compute_semantic_consistency(primary_response: str, sampled_responses: List[str]) -> float:
    """
    Computes Semantic Consistency across repeated samples / rephrasings (SelfCheckGPT principle).
    If repeated responses say essentially the same thing, consistency is high (close to 1.0).
    If repeated samples diverge or contradict, consistency is low.
    """
    if not sampled_responses:
        return 0.85 # Default prior consistency
    
    sims = [compute_cosine_similarity(primary_response, sample) for sample in sampled_responses]
    return float(np.clip(np.mean(sims), 0.0, 1.0))
