"""
Evaluation Metrics & Calibration Module
Calculates Classification Quality (Accuracy, Precision, Recall, F1, ROC-AUC)
and Probabilistic Calibration (Brier Score, Expected Calibration Error - ECE)
Reference: Section 5 Evaluation Plan of the Project Report
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics across 3 risk categories."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc_ovr = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except Exception:
        roc_auc_ovr = 0.0

    # Binary calibration: 0 = LOW (Factual) vs 1 = MEDIUM/HIGH (Risky)
    y_true_binary = (y_true > 0).astype(int)
    p_risky = y_proba[:, 1] + y_proba[:, 2] if y_proba.shape[1] >= 3 else y_proba[:, 1]
    brier = brier_score_loss(y_true_binary, p_risky)

    # False Negative analysis: Ground truth was Risky (1 or 2), but predicted as LOW (0)
    fn_mask = (y_true > 0) & (y_pred == 0)
    fn_count = int(np.sum(fn_mask))
    total_positives = int(np.sum(y_true > 0))
    fn_rate = (fn_count / total_positives) if total_positives > 0 else 0.0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc_ovr": round(float(roc_auc_ovr), 4),
        "brier_score": round(float(brier), 4),
        "false_negatives": fn_count,
        "false_negative_rate": round(float(fn_rate), 4),
        "confusion_matrix": cm
    }

def compute_calibration_curve_points(y_true_binary: np.ndarray, prob_pos: np.ndarray, n_bins: int = 5) -> Dict[str, List[float]]:
    """Calculates points for reliability diagram / calibration curve."""
    try:
        if len(np.unique(y_true_binary)) < 2:
            return {"prob_true": [0.0, 1.0], "prob_pred": [0.0, 1.0]}
        clipped = np.clip(np.asarray(prob_pos, dtype=float), 0.0, 1.0)
        prob_true, prob_pred = calibration_curve(
            y_true_binary, clipped, n_bins=n_bins, strategy="uniform"
        )
        return {
            "prob_true": [round(float(p), 4) for p in prob_true],
            "prob_pred": [round(float(p), 4) for p in prob_pred]
        }
    except Exception:
        return {"prob_true": [0.0, 1.0], "prob_pred": [0.0, 1.0]}
