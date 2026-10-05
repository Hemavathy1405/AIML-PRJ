"""
Response Uncertainty Estimation Module
Estimates model uncertainty using token-level confidence spread, semantic entropy across sampled completions,
and linguistic hedge analysis.
Reference: Farquhar et al. (2024) Nature - Detecting hallucinations using semantic entropy
"""

import math
import numpy as np
from typing import List, Optional
from .semantic_embeddings import compute_cosine_similarity

HEDGE_TERMS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "unsure",
    "unclear", "approximate", "roughly", "it seems", "suggests", "estimated",
    "conceivably", "doubtful", "allegedly"
}

CONFIDENT_ASSERTIONS = {
    "definitely", "certainly", "precisely", "always", "undeniably", "strictly",
    "conclusively", "without question", "proven", "exact", "established fact"
}

def compute_semantic_entropy(samples: List[str], similarity_threshold: float = 0.70) -> float:
    """
    Computes Semantic Entropy over sampled completions (Farquhar et al. 2024).
    Groups samples into semantic equivalence clusters and calculates normalized Shannon entropy.
    Low entropy (0.0): All samples mean the exact same thing (confident).
    High entropy (1.0): Samples diverge into distinct contradictory clusters (uncertain).
    """
    if not samples or len(samples) <= 1:
        return 0.15 # Baseline low entropy

    clusters = []
    for s in samples:
        matched = False
        for c in clusters:
            # Check similarity with representative sample
            rep = c[0]
            sim = compute_cosine_similarity(s, rep)
            if sim >= similarity_threshold:
                c.append(s)
                matched = True
                break
        if not matched:
            clusters.append([s])

    n_total = len(samples)
    entropy = 0.0
    for c in clusters:
        p_c = len(c) / n_total
        entropy -= p_c * math.log(p_c)

    # Max possible entropy is log(n_clusters) or log(n_total)
    max_entropy = math.log(max(2, n_total))
    normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return round(normalized_entropy, 4)

def estimate_token_spread_uncertainty(token_logprobs: Optional[List[float]] = None) -> float:
    """
    Computes token-level probability spread uncertainty.
    If token logprobs are provided: 1 - mean(exp(logprob))
    """
    if token_logprobs and len(token_logprobs) > 0:
        probs = [math.exp(lp) for lp in token_logprobs if lp <= 0]
        if probs:
            mean_prob = float(np.mean(probs))
            return round(float(np.clip(1.0 - mean_prob, 0.0, 1.0)), 4)
    return 0.25 # Prior default

def analyze_linguistic_confidence(text: str) -> float:
    """
    Analyzes assertive vs hedging linguistic markers.
    Returns estimated uncertainty in [0, 1].
    """
    text_lower = text.lower()
    hedge_count = sum(1 for h in HEDGE_TERMS if h in text_lower)
    conf_count = sum(1 for c in CONFIDENT_ASSERTIONS if c in text_lower)

    # Base uncertainty = 0.25
    uncertainty = 0.25 + (hedge_count * 0.15) - (conf_count * 0.12)
    return float(np.clip(uncertainty, 0.05, 0.95))

def compute_response_uncertainty(
    primary_response: str,
    sampled_responses: Optional[List[str]] = None,
    token_logprobs: Optional[List[float]] = None
) -> float:
    """
    Aggregates semantic entropy, token logprob spread, and linguistic confidence
    into a unified response uncertainty score in [0.0, 1.0].
    """
    weights = []
    scores = []

    # 1. Semantic Entropy (weight: 0.50 if samples provided)
    if sampled_responses and len(sampled_responses) > 0:
        all_samples = [primary_response] + sampled_responses
        sem_ent = compute_semantic_entropy(all_samples)
        scores.append(sem_ent)
        weights.append(0.50)

    # 2. Token logprob spread (weight: 0.30 if provided)
    if token_logprobs is not None:
        tok_unc = estimate_token_spread_uncertainty(token_logprobs)
        scores.append(tok_unc)
        weights.append(0.30)

    # 3. Linguistic analysis (weight: 0.20)
    ling_unc = analyze_linguistic_confidence(primary_response)
    scores.append(ling_unc)
    weights.append(0.20 if len(scores) > 1 else 1.0)

    # Normalize weights
    weights = np.array(weights) / sum(weights)
    final_uncertainty = float(np.dot(scores, weights))
    return round(float(np.clip(final_uncertainty, 0.0, 1.0)), 4)
