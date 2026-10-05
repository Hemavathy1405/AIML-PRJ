"""
Hallucination Risk Scorer & Pipeline Orchestrator
Converts model predictions to calibrated probabilities, maps to LOW/MEDIUM/HIGH categories,
and triggers human review routing.
Reference: Section 4.4 Step 7 & Step 8, and Section 4.5 System Architecture
"""

import os
import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from config.config import (
        XGBOOST_MODEL_FILE, SVM_MODEL_FILE, RF_MODEL_FILE, LR_MODEL_FILE,
        SCALER_FILE, MODEL_METADATA_FILE, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
        ALL_FEATURES, RISK_LABELS, ERROR_TAXONOMY
    )
    from src.features.feature_extractor import HallucinationFeatureExtractor
except (ImportError, ValueError):
    from ...config.config import (
        XGBOOST_MODEL_FILE, SVM_MODEL_FILE, RF_MODEL_FILE, LR_MODEL_FILE,
        SCALER_FILE, MODEL_METADATA_FILE, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
        ALL_FEATURES, RISK_LABELS, ERROR_TAXONOMY
    )
    from ..features.feature_extractor import HallucinationFeatureExtractor

class HallucinationRiskScorer:
    """Predicts hallucination risk probability and determines routing decision."""

    def __init__(self, model_name: str = "XGBoost"):
        self.model_name = model_name
        self.extractor = HallucinationFeatureExtractor()
        self.scaler = None
        self.model = None
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads fitted scaler and model if they exist."""
        if SCALER_FILE.exists():
            self.scaler = joblib.load(SCALER_FILE)
        
        model_paths = {
            "XGBoost": XGBOOST_MODEL_FILE,
            "SVM": SVM_MODEL_FILE,
            "Random Forest": RF_MODEL_FILE,
            "Logistic Regression": LR_MODEL_FILE
        }
        target_path = model_paths.get(self.model_name, XGBOOST_MODEL_FILE)
        if target_path.exists():
            self.model = joblib.load(target_path)
            
        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict_from_features(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Calculates risk probability and category directly from feature dictionary."""
        # Feature array in exact order
        raw_vec = np.array([[features_dict[f] for f in ALL_FEATURES]], dtype=np.float32)

        if self.scaler is not None and self.model is not None:
            X_scaled = self.scaler.transform(raw_vec)
            class_probs = self.model.predict_proba(X_scaled)[0] # [P_low, P_med, P_high]
            
            p_low = float(class_probs[0])
            p_med = float(class_probs[1]) if len(class_probs) > 1 else 0.0
            p_high = float(class_probs[2]) if len(class_probs) > 2 else 0.0

            # Calibrated hallucination risk score: continuous expected risk severity in [0, 1]
            risk_probability = float(np.clip(0.0 * p_low + 0.5 * p_med + 1.0 * p_high, 0.0, 1.0))
            # Safety override: never pass an obvious contradiction as LOW
            contra = features_dict.get("contradiction_score", 0.0)
            supp = features_dict.get("evidence_support", 1.0)
            conf_gap = features_dict.get("confidence_factuality_gap", 0.0)
            if supp < 0.20 and (contra >= 0.55 or conf_gap >= 0.55):
                risk_probability = max(risk_probability, 0.78)
                p_high = max(p_high, 0.70)
                p_med = min(p_med, 0.20)
                p_low = max(0.0, 1.0 - p_high - p_med)
            elif supp < 0.45 and contra >= 0.25 and conf_gap >= 0.20:
                risk_probability = max(risk_probability, 0.42)
        else:
            # Fallback heuristic rule-engine based on paper formulas if models not yet trained
            # High gap or high contradiction -> High risk
            # High support + consistency -> Low risk
            c_gap = features_dict["claim_evidence_gap"]
            conf_gap = features_dict["confidence_factuality_gap"]
            contra = features_dict["contradiction_score"]
            supp = features_dict["evidence_support"]
            
            raw_risk = (0.35 * contra) + (0.30 * conf_gap) + (0.25 * c_gap) + (0.10 * (1.0 - supp))
            risk_probability = float(np.clip(raw_risk, 0.0, 1.0))
            p_low = max(0.0, 1.0 - risk_probability)
            p_med = 0.5 * risk_probability
            p_high = 0.5 * risk_probability

        # Categorization based on paper thresholds
        if risk_probability <= LOW_RISK_THRESHOLD:
            category = "LOW"
            action = "SHOW_ANSWER"
            action_desc = "Answer is verified and strongly grounded. Show directly to the user."
            color = "#28a745" # Green
        elif risk_probability < HIGH_RISK_THRESHOLD:
            category = "MEDIUM"
            action = "RECHECK_EVIDENCE"
            action_desc = "Moderate risk / partial evidence grounding. Flagged for evidence re-check or user clarification."
            color = "#ffc107" # Amber
        else:
            category = "HIGH"
            action = "HUMAN_REVIEW"
            action_desc = "Critical hallucination risk detected. Suppress answer and route to Human Review loop."
            color = "#dc3545" # Red

        # Identify primary drivers of hallucination risk
        drivers = []
        if features_dict.get("contradiction_score", 0) > 0.35:
            drivers.append("Direct factual contradiction against reference evidence")
        if features_dict.get("confidence_factuality_gap", 0) > 0.40:
            drivers.append("High model confidence despite low evidence grounding (Confidently-Wrong)")
        if features_dict.get("claim_evidence_gap", 0) > 0.30:
            drivers.append("Topic semantic overlap without genuine logical entailment (Alignment Gap)")
        if features_dict.get("evidence_support", 1.0) < 0.30:
            drivers.append("Insufficient evidence support / ungrounded assertions")

        if not drivers and category == "LOW":
            drivers.append("High evidence entailment and stable cross-sample consistency")

        return {
            "risk_probability": round(risk_probability, 4),
            "risk_category": category,
            "action": action,
            "action_desc": action_desc,
            "color": color,
            "probabilities": {
                "LOW": round(p_low, 4),
                "MEDIUM": round(p_med, 4),
                "HIGH": round(p_high, 4)
            },
            "features": features_dict,
            "risk_drivers": drivers,
            "model_used": self.model_name
        }

    def assess_response(
        self,
        question: str,
        reference_evidence: str,
        response: str,
        sampled_responses: Optional[List[str]] = None,
        token_logprobs: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """End-to-end evaluation: raw text -> 7 features -> probability -> routing decision."""
        features = self.extractor.extract_features(
            reference_evidence=reference_evidence,
            response=response,
            sampled_responses=sampled_responses,
            token_logprobs=token_logprobs,
            return_dict=True
        )

        result = self.predict_from_features(features)
        result["question"] = question
        result["response"] = response
        result["reference_evidence"] = reference_evidence
        return result
