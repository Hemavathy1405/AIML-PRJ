"""
Unit Tests for Hallucination Feature Extraction
Tests all 4 Core + 3 Add-on Derived Features
"""

import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.feature_extractor import HallucinationFeatureExtractor
from src.features.nli_model import compute_nli_scores
from src.features.semantic_embeddings import compute_semantic_consistency, compute_cosine_similarity
from src.features.uncertainty import compute_response_uncertainty

def test_feature_ranges():
    extractor = HallucinationFeatureExtractor()
    evidence = "CRISPR-Cas9 enables targeted genome editing using a guide RNA and Cas9 endonuclease."
    response = "CRISPR-Cas9 edits genomes with RNA guidance."

    feats = extractor.extract_features(
        reference_evidence=evidence,
        response=response,
        sampled_responses=["CRISPR-Cas9 cuts DNA using guide RNA."],
        token_logprobs=[-0.05, -0.02, -0.04]
    )

    # All 7 features must be in valid [0, 1] range
    expected_keys = [
        "semantic_consistency", "evidence_support", "contradiction_score",
        "response_uncertainty", "claim_evidence_gap", "confidence_factuality_gap",
        "semantic_evidence_interaction"
    ]
    for key in expected_keys:
        assert key in feats, f"Missing key: {key}"
        assert 0.0 <= feats[key] <= 1.0, f"Feature {key} out of bounds: {feats[key]}"

def test_derived_features_logic():
    extractor = HallucinationFeatureExtractor()
    
    # 1. Confidently wrong response: asserts contradiction
    evidence = "The Earth revolves around the Sun in approximately 365.25 days."
    confidently_wrong_response = "The Sun definitely revolves around the Earth in 24 hours."

    feats_wrong = extractor.extract_features(
        reference_evidence=evidence,
        response=confidently_wrong_response,
        sampled_responses=[
            "The Sun definitely revolves around the Earth in 24 hours.",
            "It is certain that the Sun orbits the Earth every 24 hours."
        ],
        token_logprobs=[-0.01, -0.02] # Very high confidence (low uncertainty)
    )

    # Confidence-factuality gap must be high for confidently wrong statements
    assert feats_wrong["confidence_factuality_gap"] > 0.3
    # Contradiction score must be elevated
    assert feats_wrong["contradiction_score"] > 0.2

    # 2. Fully grounded response
    grounded_response = "The Earth orbits around the Sun over roughly 365 days."
    feats_grounded = extractor.extract_features(
        reference_evidence=evidence,
        response=grounded_response,
        sampled_responses=["Earth takes about 365 days to circle the Sun."],
        token_logprobs=[-0.03, -0.04]
    )

    # Evidence support must be significantly higher than the wrong response
    assert feats_grounded["evidence_support"] > feats_wrong["evidence_support"]
    # Contradiction must be lower
    assert feats_grounded["contradiction_score"] < feats_wrong["contradiction_score"]

if __name__ == "__main__":
    test_feature_ranges()
    test_derived_features_logic()
    print("All feature extraction tests passed successfully!")
