"""
Error Taxonomy Analysis Module
Analyzes hallucination distributions, misclassified cases, and false negatives
according to the Sun et al. (2024) taxonomy.
Reference: Section 5 Evaluation Plan of the Project Report
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def analyze_error_taxonomy_breakdown(df_test_predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Groups predictions by Sun et al. error taxonomy and inspects false negatives.
    """
    taxonomy_counts = df_test_predictions["error_taxonomy"].value_counts().to_dict()
    
    # Inspect false negatives (Ground truth was hallucinated/partial, but predicted as LOW)
    fn_df = df_test_predictions[(df_test_predictions["risk_label"] > 0) & (df_test_predictions["pred_risk_label"] == 0)]
    fn_breakdown = fn_df["error_taxonomy"].value_counts().to_dict() if len(fn_df) > 0 else {}

    # Performance per taxonomy category
    category_performance = []
    for category in df_test_predictions["error_taxonomy"].unique():
        cat_df = df_test_predictions[df_test_predictions["error_taxonomy"] == category]
        correct = (cat_df["risk_label"] == cat_df["pred_risk_label"]).sum()
        total = len(cat_df)
        acc = round(float(correct / total), 4) if total > 0 else 0.0
        category_performance.append({
            "category": category,
            "sample_count": total,
            "accuracy": acc,
            "false_negatives": int(((cat_df["risk_label"] > 0) & (cat_df["pred_risk_label"] == 0)).sum())
        })

    return {
        "overall_distribution": taxonomy_counts,
        "false_negative_count": len(fn_df),
        "false_negative_taxonomy": fn_breakdown,
        "category_performance": category_performance,
        "taxonomy_reference": "Sun et al. (2024) - AI hallucination taxonomy: factual errors, reasoning errors, logic errors, unfounded fabrication."
    }
