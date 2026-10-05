"""
Unit Tests for Model Classification and Risk Scorer
"""

import sys
import numpy as np
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.risk_scorer import HallucinationRiskScorer
from src.human_loop.review_manager import HumanReviewManager

def test_risk_scorer_heuristics_and_categories():
    scorer = HallucinationRiskScorer()

    # Grounded features -> LOW
    grounded_feats = {
        "semantic_consistency": 0.95,
        "evidence_support": 0.92,
        "contradiction_score": 0.03,
        "response_uncertainty": 0.12,
        "claim_evidence_gap": 0.05,
        "confidence_factuality_gap": 0.04,
        "semantic_evidence_interaction": 0.87
    }
    res_low = scorer.predict_from_features(grounded_feats)
    assert res_low["risk_category"] == "LOW"
    assert res_low["action"] == "SHOW_ANSWER"

    # Confidently wrong features -> HIGH
    halluc_feats = {
        "semantic_consistency": 0.88,
        "evidence_support": 0.10,
        "contradiction_score": 0.85,
        "response_uncertainty": 0.08, # Highly confident!
        "claim_evidence_gap": 0.65,
        "confidence_factuality_gap": 0.82,
        "semantic_evidence_interaction": 0.09
    }
    res_high = scorer.predict_from_features(halluc_feats)
    assert res_high["risk_category"] == "HIGH"
    assert res_high["action"] == "HUMAN_REVIEW"

def test_human_review_lifecycle(tmp_path):
    queue_file = tmp_path / "test_review_queue.json"
    manager = HumanReviewManager(queue_file=queue_file)

    sample_assessment = {
        "question": "What is the boiling point of nitrogen?",
        "response": "Nitrogen boils at 500 degrees Celsius.",
        "reference_evidence": "Nitrogen has a boiling point of 77 Kelvin (-196 C).",
        "risk_category": "HIGH",
        "risk_probability": 0.94,
        "features": {},
        "risk_drivers": ["Direct factual contradiction"]
    }

    review_id = manager.enqueue_for_review(sample_assessment)
    assert review_id.startswith("rev_")
    
    pending = manager.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0]["review_id"] == review_id

    # Resolve review
    ok = manager.submit_review(
        review_id=review_id,
        status="REJECTED",
        assigned_error_type="Factual Error (Wrong Entity/Number/Date)",
        reviewer_notes="Model mixed up boiling point of nitrogen with high temperature steel.",
        corrected_response="Nitrogen boils at -196 C (77 K)."
    )
    assert ok is True
    assert len(manager.get_pending_reviews()) == 0

if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        test_risk_scorer_heuristics_and_categories()
        test_human_review_lifecycle(Path(tmpdir))
    print("All model and human review tests passed successfully!")
