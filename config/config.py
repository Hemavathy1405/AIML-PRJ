"""
Project Configuration: AI Hallucination Risk Prediction
Course: 24DSP501 - Technical Communication in Data Science
Authors: Hemavathy K (Roll No: 21662), Madhumitha P (Roll No: 22286)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
REFERENCE_CORPUS_FILE = RAW_DATA_DIR / "reference_corpus.json"
CURATED_QA_FILE = RAW_DATA_DIR / "curated_qa_pairs.json"
DATASET_LABELED_FILE = PROCESSED_DATA_DIR / "hallucination_dataset_labeled.csv"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.csv"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "test_features.csv"

# Model Artifacts
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.joblib"
SVM_MODEL_FILE = MODELS_DIR / "svm_model.joblib"
RF_MODEL_FILE = MODELS_DIR / "rf_model.joblib"
LR_MODEL_FILE = MODELS_DIR / "lr_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
EVALUATION_RESULTS_FILE = EXPERIMENTS_DIR / "evaluation_results.json"
HUMAN_REVIEW_QUEUE_FILE = DATA_DIR / "human_review_queue.json"

# Feature Definitions
CORE_FEATURES = [
    "semantic_consistency",      # Agreement across sampled generations / rephrasings
    "evidence_support",          # NLI entailment probability against reference evidence
    "contradiction_score",       # NLI contradiction probability against reference evidence
    "response_uncertainty"       # Semantic entropy / confidence spread across tokens
]

ADDON_FEATURES = [
    "claim_evidence_gap",        # max(0, semantic_sim - evidence_support) -> catches superficially related but unsupported claims
    "confidence_factuality_gap", # max(0, (1 - uncertainty) - evidence_support) -> catches confidently-wrong claims
    "semantic_evidence_interaction" # semantic_consistency * evidence_support -> grounding interaction
]

ALL_FEATURES = CORE_FEATURES + ADDON_FEATURES

# Risk Probability Thresholds (mapped to LOW, MEDIUM, HIGH)
LOW_RISK_THRESHOLD = 0.35     # Probability <= 0.35 -> LOW (Approved, show to user)
HIGH_RISK_THRESHOLD = 0.70    # Probability >= 0.70 -> HIGH (Flagged for human review)
# Between 0.35 and 0.70 -> MEDIUM (Evidence re-check / human verification loop)

RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

# Sun et al. (2024) Error Taxonomy Mapping
ERROR_TAXONOMY = [
    "Factual Error (Wrong Entity/Number/Date)",
    "Unfounded Fabrication (Entirely Invented Fact)",
    "Contradiction to Evidence",
    "Reasoning / Logic Error",
    "Overconfidence in Extrapolation",
    "None (Grounded & Faithful)"
]

# Random Seed for Reproducibility
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.25

# Optional Hugging Face backends (off by default so the project runs offline).
# Set HALLUCINATION_USE_TRANSFORMERS=1 to enable MiniLM embeddings and DeBERTa NLI.
USE_TRANSFORMER_BACKENDS = os.environ.get("HALLUCINATION_USE_TRANSFORMERS", "0") == "1"
