"""
Human-in-the-Loop Review Manager (Step 8)
Manages triage queue for MEDIUM and HIGH hallucination-risk responses.
Tracks human verification, error taxonomy classification (Sun et al., 2024), and audit trails.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from config.config import HUMAN_REVIEW_QUEUE_FILE, ERROR_TAXONOMY
except (ImportError, ValueError):
    from ...config.config import HUMAN_REVIEW_QUEUE_FILE, ERROR_TAXONOMY

class HumanReviewManager:
    """Manages the review lifecycle for escalated responses."""

    def __init__(self, queue_file: Optional[Path] = None):
        self.queue_file = queue_file or HUMAN_REVIEW_QUEUE_FILE
        self.queue: List[Dict[str, Any]] = []
        self._load_queue()

    def _load_queue(self):
        """Loads persistent review queue from disk."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    self.queue = json.load(f)
            except Exception:
                self.queue = []
        else:
            self.queue = []

    def _save_queue(self):
        """Saves review queue to disk."""
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(self.queue, f, indent=2)

    def enqueue_for_review(self, assessment_result: Dict[str, Any], context: str = "") -> str:
        """
        Adds a flagged response (MEDIUM or HIGH risk) to the human triage queue.
        Returns the unique review item ID.
        """
        review_id = f"rev_{len(self.queue) + 1:04d}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        item = {
            "review_id": review_id,
            "timestamp": datetime.now().isoformat(),
            "question": assessment_result.get("question", ""),
            "response": assessment_result.get("response", ""),
            "reference_evidence": assessment_result.get("reference_evidence", ""),
            "risk_category": assessment_result.get("risk_category", "HIGH"),
            "risk_probability": assessment_result.get("risk_probability", 0.0),
            "features": assessment_result.get("features", {}),
            "risk_drivers": assessment_result.get("risk_drivers", []),
            "status": "PENDING_REVIEW", # PENDING_REVIEW, APPROVED, REJECTED, MODIFIED
            "reviewer_notes": "",
            "corrected_response": "",
            "assigned_error_type": "Pending Classification",
            "context": context
        }
        self.queue.append(item)
        self._save_queue()
        return review_id

    def seed_demo_queue(self) -> None:
        """Load two example tickets so the review workstation is usable on first launch."""
        if self.queue:
            return
        self.enqueue_for_review(
            {
                "question": "Where is the James Webb Space Telescope positioned and how is its primary mirror constructed?",
                "response": "The James Webb Space Telescope orbits Earth in Low Earth Orbit at 400 km altitude and carries a monolithic 20-meter glass mirror manufactured by Galileo Galilei in Florence.",
                "reference_evidence": "JWST operates at the second Sun-Earth Lagrange point (L2), approximately 1.5 million kilometers from Earth, and features a 6.5-meter primary mirror composed of 18 hexagonal beryllium segments coated with gold.",
                "risk_category": "HIGH",
                "risk_probability": 0.91,
                "features": {
                    "semantic_consistency": 0.94,
                    "evidence_support": 0.08,
                    "contradiction_score": 0.81,
                    "response_uncertainty": 0.07,
                    "claim_evidence_gap": 0.62,
                    "confidence_factuality_gap": 0.85,
                    "semantic_evidence_interaction": 0.07
                },
                "risk_drivers": [
                    "Direct factual contradiction against reference evidence",
                    "High model confidence despite low evidence grounding (Confidently-Wrong)"
                ]
            },
            context="Seeded demo ticket"
        )
        self.enqueue_for_review(
            {
                "question": "How does the CRISPR-Cas9 system identify its DNA target and create genomic edits?",
                "response": "CRISPR-Cas9 recognizes genomic targets using a guide RNA and cuts the DNA adjacent to a PAM motif. It also actively repairs the DNA strand itself using an intrinsic polymerase sub-domain before releasing the cleaved sequence.",
                "reference_evidence": "The Cas9 endonuclease is directed by a single guide RNA matching the genomic target adjacent to a PAM. Host repair mechanisms (NHEJ or HDR) resolve the double-strand break.",
                "risk_category": "MEDIUM",
                "risk_probability": 0.48,
                "features": {
                    "semantic_consistency": 0.88,
                    "evidence_support": 0.46,
                    "contradiction_score": 0.31,
                    "response_uncertainty": 0.33,
                    "claim_evidence_gap": 0.29,
                    "confidence_factuality_gap": 0.21,
                    "semantic_evidence_interaction": 0.40
                },
                "risk_drivers": [
                    "Topic semantic overlap without genuine logical entailment (Alignment Gap)"
                ]
            },
            context="Seeded demo ticket"
        )

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Returns all items awaiting human review."""
        return [item for item in self.queue if item["status"] == "PENDING_REVIEW"]

    def get_all_reviews(self) -> List[Dict[str, Any]]:
        """Returns all review queue items."""
        return self.queue

    def submit_review(
        self,
        review_id: str,
        status: str,
        assigned_error_type: str,
        reviewer_notes: str = "",
        corrected_response: str = ""
    ) -> bool:
        """Submits human verification decision, taxonomy tagging, and optional correction."""
        for item in self.queue:
            if item["review_id"] == review_id:
                item["status"] = status
                item["assigned_error_type"] = assigned_error_type
                item["reviewer_notes"] = reviewer_notes
                item["corrected_response"] = corrected_response
                item["resolved_at"] = datetime.now().isoformat()
                self._save_queue()
                return True
        return False
