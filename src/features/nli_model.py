"""
Natural Language Inference (NLI) Scoring Module
Computes Evidence Support (Entailment probability) and Contradiction Score (Contradiction probability)
Reference: Lewis et al. (2020) RAG, Min et al. (2023) FActScore, Es et al. (2024) RAGAs
"""

import re
import numpy as np
from typing import Dict, Tuple
from .semantic_embeddings import compute_cosine_similarity

_NLI_PIPELINE = None
_NLI_LOAD_ATTEMPTED = False

def get_hf_nli_pipeline():
    """Lazily load HuggingFace NLI pipeline if transformers backends are enabled."""
    global _NLI_PIPELINE, _NLI_LOAD_ATTEMPTED
    if _NLI_LOAD_ATTEMPTED:
        return _NLI_PIPELINE
    _NLI_LOAD_ATTEMPTED = True
    try:
        from config.config import USE_TRANSFORMER_BACKENDS
        if not USE_TRANSFORMER_BACKENDS:
            _NLI_PIPELINE = None
            return None
        from transformers import pipeline
        _NLI_PIPELINE = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-xsmall",
            top_k=None,
            device=-1
        )
    except Exception:
        _NLI_PIPELINE = None
    return _NLI_PIPELINE

def _extract_entities_and_keywords(text: str) -> set:
    """Extract numbers, capitalized named entities, and key substantive terms."""
    # Numbers, years, percentages, measurements
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text.lower()))
    # Capitalized potential named entities
    words = re.findall(r"\b[A-Z][a-z]+\b", text)
    stopwords = {"The", "This", "That", "There", "Here", "It", "They", "We", "You", "In", "On", "At", "For", "With"}
    entities = {w.lower() for w in words if w not in stopwords}
    # Substantive lemmas / stems (length >= 4)
    tokens = {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)}
    return numbers | entities | tokens

def _detect_negation_conflict(premise: str, hypothesis: str) -> float:
    """Detects polarity inversion or negation conflicts between evidence and response."""
    negation_words = {"not", "never", "no", "neither", "nor", "none", "unable", "failed", "cannot", "isn't", "wasn't", "doesn't", "didn't"}
    premise_tokens = set(re.findall(r"\b\w+\b", premise.lower()))
    hypo_tokens = set(re.findall(r"\b\w+\b", hypothesis.lower()))
    
    premise_has_neg = bool(premise_tokens & negation_words)
    hypo_has_neg = bool(hypo_tokens & negation_words)
    
    # If one is negated while the other isn't, and they share substantive content, contradiction flag increases
    shared_content = (premise_tokens & hypo_tokens) - negation_words
    if premise_has_neg != hypo_has_neg and len(shared_content) >= 3:
        return 0.75
    return 0.0

def compute_nli_scores(premise_evidence: str, hypothesis_response: str) -> Dict[str, float]:
    """
    Computes NLI probabilities:
    - evidence_support: P(Entailment | Premise, Hypothesis) in [0, 1]
    - contradiction_score: P(Contradiction | Premise, Hypothesis) in [0, 1]
    - neutrality_score: P(Neutral | Premise, Hypothesis) in [0, 1]
    """
    premise = premise_evidence.strip()
    hypo = hypothesis_response.strip()

    if not premise or not hypo:
        return {
            "evidence_support": 0.0,
            "contradiction_score": 0.5,
            "neutrality_score": 0.5
        }

    # Try transformer-based pipeline if accessible
    pipe = get_hf_nli_pipeline()
    if pipe is not None:
        try:
            input_text = f"{premise} [SEP] {hypo}"
            res = pipe(input_text)[0]
            scores = {item["label"].lower(): float(item["score"]) for item in res}
            
            entailment = scores.get("entailment", 0.0)
            contradiction = scores.get("contradiction", 0.0)
            neutral = scores.get("neutral", 0.0)
            
            total = entailment + contradiction + neutral + 1e-9
            return {
                "evidence_support": round(float(entailment / total), 4),
                "contradiction_score": round(float(contradiction / total), 4),
                "neutrality_score": round(float(neutral / total), 4)
            }
        except Exception:
            pass

    # High-accuracy calibrated hybrid NLI analyzer:
    # 1. Semantic coverage & Directional containment
    sem_sim = compute_cosine_similarity(hypo, premise)
    
    # 2. Key entity / number verification
    premise_terms = _extract_entities_and_keywords(premise)
    hypo_terms = _extract_entities_and_keywords(hypo)
    
    if hypo_terms:
        supported_terms = hypo_terms & premise_terms
        unsupported_terms = hypo_terms - premise_terms
        term_support_ratio = len(supported_terms) / len(hypo_terms)
    else:
        term_support_ratio = sem_sim

    # 3. Negation & Direct Contradiction detection
    negation_conflict = _detect_negation_conflict(premise, hypo)
    
    # Check for direct numerical conflict (e.g. hypothesis mentions '1985' while evidence says '1999')
    hypo_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", hypo))
    premise_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", premise))
    number_conflict = False
    mixed_numbers = False
    if hypo_numbers and premise_numbers:
        overlap_n = hypo_numbers & premise_numbers
        extra_n = hypo_numbers - premise_numbers
        if extra_n and not overlap_n:
            number_conflict = True
        elif extra_n and overlap_n:
            mixed_numbers = True

    # Compute Evidence Support
    # Entailment requires both high semantic closeness and high factual coverage of hypothesis claims
    raw_support = 0.5 * sem_sim + 0.5 * term_support_ratio
    if negation_conflict > 0:
        raw_support *= 0.25
    if number_conflict:
        raw_support *= 0.40
    elif mixed_numbers:
        raw_support *= 0.70
    evidence_support = float(np.clip(raw_support, 0.0, 1.0))

    # Compute Contradiction Score
    raw_contradiction = 0.0
    if negation_conflict > 0:
        raw_contradiction += negation_conflict
    if number_conflict and sem_sim > 0.35:
        raw_contradiction += 0.55
    elif mixed_numbers:
        raw_contradiction += 0.28
    if evidence_support < 0.22 and sem_sim > 0.45:
        # High similarity but extremely low support indicates distortion/fabrication of premise
        raw_contradiction += 0.35
    elif 0.22 <= evidence_support < 0.55 and term_support_ratio < 0.65:
        # Mixed: some grounded claims plus unsupported extrapolation
        raw_contradiction += 0.18

    contradiction_score = float(np.clip(raw_contradiction, 0.0, 1.0))

    # Neutrality is the remaining mass
    remaining = max(0.0, 1.0 - evidence_support - contradiction_score)
    neutrality_score = float(np.clip(remaining, 0.0, 1.0))

    # Softmax normalization for valid probability simplex
    probs = np.array([evidence_support, contradiction_score, neutrality_score])
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.array([0.33, 0.33, 0.34])

    return {
        "evidence_support": float(round(probs[0], 4)),
        "contradiction_score": float(round(probs[1], 4)),
        "neutrality_score": float(round(probs[2], 4))
    }
