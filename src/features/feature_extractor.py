"""
Feature Extractor Module
Extracts 4 Core Signals + 3 Add-on Derived Features for AI Hallucination Risk Prediction
Total: 7 Features exactly as formulated in the Project Report
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from .semantic_embeddings import compute_semantic_consistency, compute_cosine_similarity
from .nli_model import compute_nli_scores
from .uncertainty import compute_response_uncertainty

class HallucinationFeatureExtractor:
    """
    Computes the 7 hallucination risk features from:
    - User Question
    - Reference Evidence Text (retrieved or provided)
    - Primary Generated Response
    - Sampled Alternative Responses (optional, generated for consistency/entropy)
    - Token Logprobs (optional)
    """

    def __init__(self):
        pass

    def extract_features(
        self,
        reference_evidence: str,
        response: str,
        sampled_responses: Optional[List[str]] = None,
        token_logprobs: Optional[List[float]] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, float], np.ndarray]:
        """
        Calculates all 4 Core + 3 Add-on Derived features.
        """
        if sampled_responses is None:
            # Generate deterministic minor paraphrasings for sampling if none provided
            sampled_responses = [response]

        # -------------------------------------------------------------
        # 1. CORE FEATURES (from Task 1)
        # -------------------------------------------------------------
        
        # Core Signal 1: Semantic Consistency
        # Stability across repeated samplings or rephrasings
        semantic_consistency = compute_semantic_consistency(response, sampled_responses)

        # Core Signals 2 & 3: Evidence Support & Contradiction Score (NLI)
        nli_res = compute_nli_scores(premise_evidence=reference_evidence, hypothesis_response=response)
        evidence_support = nli_res["evidence_support"]
        contradiction_score = nli_res["contradiction_score"]

        # Core Signal 4: Response Uncertainty
        response_uncertainty = compute_response_uncertainty(
            primary_response=response,
            sampled_responses=sampled_responses,
            token_logprobs=token_logprobs
        )

        # -------------------------------------------------------------
        # 2. ADD-ON DERIVED FEATURES (New Contribution)
        # -------------------------------------------------------------

        # Auxiliary Metric: Semantic similarity between response and reference evidence
        sim_response_evidence = compute_cosine_similarity(response, reference_evidence)

        # Derived Feature 1: Claim-Evidence Alignment Gap
        # Difference between topical semantic similarity and true logical entailment
        # A large gap detects responses that "sound related" but aren't actually backed by source
        claim_evidence_gap = float(np.clip(sim_response_evidence - evidence_support, 0.0, 1.0))

        # Derived Feature 2: Confidence-Factuality Gap
        # Difference between model's self-confidence (1 - uncertainty) and actual evidence support
        # High value indicates a dangerously confident, ungrounded answer
        model_confidence = float(np.clip(1.0 - response_uncertainty, 0.0, 1.0))
        confidence_factuality_gap = float(np.clip(model_confidence - evidence_support, 0.0, 1.0))

        # Derived Feature 3: Semantic-Evidence Interaction
        # Product of semantic consistency and evidence support
        # Distinguishes consistently grounded answers from consistently hallucinated ones
        semantic_evidence_interaction = float(np.clip(semantic_consistency * evidence_support, 0.0, 1.0))

        features = {
            # 4 Core Features
            "semantic_consistency": round(semantic_consistency, 4),
            "evidence_support": round(evidence_support, 4),
            "contradiction_score": round(contradiction_score, 4),
            "response_uncertainty": round(response_uncertainty, 4),
            
            # 3 Add-on Derived Features
            "claim_evidence_gap": round(claim_evidence_gap, 4),
            "confidence_factuality_gap": round(confidence_factuality_gap, 4),
            "semantic_evidence_interaction": round(semantic_evidence_interaction, 4),
            
            # Diagnostic metadata
            "_semantic_similarity": round(sim_response_evidence, 4),
            "_model_confidence": round(model_confidence, 4)
        }

        if return_dict:
            return features
        else:
            return np.array([
                features["semantic_consistency"],
                features["evidence_support"],
                features["contradiction_score"],
                features["response_uncertainty"],
                features["claim_evidence_gap"],
                features["confidence_factuality_gap"],
                features["semantic_evidence_interaction"]
            ], dtype=np.float32)

    def extract_batch(self, df_records: pd.DataFrame) -> pd.DataFrame:
        """Extract features for a whole dataframe of records."""
        extracted_rows = []
        for idx, row in df_records.iterrows():
            samples = row.get("sampled_responses", None)
            if isinstance(samples, str):
                import json
                try:
                    samples = json.loads(samples)
                except Exception:
                    samples = [samples]
            feats = self.extract_features(
                reference_evidence=row["reference_evidence"],
                response=row["response"],
                sampled_responses=samples
            )
            extracted_rows.append(feats)
        return pd.DataFrame(extracted_rows)
