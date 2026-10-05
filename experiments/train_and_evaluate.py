"""
End-to-End Model Training, Benchmarking, and Evaluation Pipeline
Trains: XGBoost (Primary), SVM, Random Forest, Logistic Regression
Performs: Cross-Model Benchmark, Ablation Study (Core 4 vs All 7), Calibration Analysis, Error Taxonomy Breakdown
Reference: Section 4.4 (Steps 5, 6, 7), Section 5 (Evaluation Plan), Section 6 (Expected Outcomes)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    TRAIN_FEATURES_FILE, TEST_FEATURES_FILE, EVALUATION_RESULTS_FILE,
    ALL_FEATURES, CORE_FEATURES, RANDOM_SEED
)
from src.models.classifier import HallucinationClassifierTrainer
from src.evaluation.metrics import compute_classification_metrics, compute_calibration_curve_points
from src.evaluation.ablation import run_ablation_study
from src.evaluation.error_taxonomy import analyze_error_taxonomy_breakdown

def run_training_and_evaluation():
    print("=" * 70)
    print("AI Hallucination Risk Prediction: Model Training & Evaluation Suite")
    print("=" * 70)

    if not TRAIN_FEATURES_FILE.exists() or not TEST_FEATURES_FILE.exists():
        print("[!] Feature files missing. Generating the labeled dataset first...")
        from data.generate_dataset import main as generate_dataset_main
        generate_dataset_main()

    # 1. Load Train and Test datasets
    train_df = pd.read_csv(TRAIN_FEATURES_FILE)
    test_df = pd.read_csv(TEST_FEATURES_FILE)
    print(f"[+] Loaded Train Set: {len(train_df)} samples, Test Set: {len(test_df)} samples")

    trainer = HallucinationClassifierTrainer(random_seed=RANDOM_SEED)

    # 2. Scale features and prepare matrices (All 7 Features)
    X_train, y_train, X_test, y_test, feature_names = trainer.prepare_data(
        train_df, test_df, feature_set="all"
    )

    # 3. Train all 4 models
    print("\n[*] Training Classifiers (XGBoost, Random Forest, SVM, Logistic Regression)...")
    models = trainer.train_all_models(X_train, y_train)

    # 4. Evaluate each model
    results = {}
    print("\n" + "=" * 70)
    print(f"{'Model':<22} | {'Accuracy':<8} | {'F1-Macro':<8} | {'ROC-AUC':<8} | {'Brier':<8} | {'FN Rate':<8}")
    print("-" * 70)

    for name, model in models.items():
        metrics = trainer.evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(
            f"{name:<22} | {metrics['accuracy']:<8.4f} | {metrics['f1_macro']:<8.4f} | "
            f"{metrics['roc_auc_ovr']:<8.4f} | {metrics['brier_score']:<8.4f} | {metrics['false_negative_rate']:<8.4f}"
        )
    print("=" * 70)

    # 5. Save model artifacts
    trainer.save_artifacts(feature_names=feature_names)
    print("\n[+] Model artifacts saved to 'models/' directory.")

    # 6. Run Ablation Study (Core 4 vs All 7 Features)
    print("\n[*] Running Ablation Study (Core 4 vs All 7 Features)...")
    ablation_res = run_ablation_study(train_df, test_df)
    print("-" * 70)
    print(f"Baseline (Core 4):   Acc={ablation_res['baseline_core_4']['accuracy']:.4f}, "
          f"F1={ablation_res['baseline_core_4']['f1_macro']:.4f}, "
          f"ROC={ablation_res['baseline_core_4']['roc_auc_ovr']:.4f}, "
          f"FN={ablation_res['baseline_core_4']['false_negatives']}")
    print(f"Proposed (All 7):    Acc={ablation_res['proposed_all_7']['accuracy']:.4f}, "
          f"F1={ablation_res['proposed_all_7']['f1_macro']:.4f}, "
          f"ROC={ablation_res['proposed_all_7']['roc_auc_ovr']:.4f}, "
          f"FN={ablation_res['proposed_all_7']['false_negatives']}")
    print(f"Outcome: {ablation_res['conclusion']}")
    print("-" * 70)

    # 7. Calibration Curve computation for XGBoost
    xgb_model = models["XGBoost"]
    y_test_binary = (y_test > 0).astype(int)
    prob_risky = xgb_model.predict_proba(X_test)[:, 1] + xgb_model.predict_proba(X_test)[:, 2]
    calibration_data = compute_calibration_curve_points(y_test_binary, prob_risky)

    # 8. Error Taxonomy Breakdown on Test Set
    test_pred_df = test_df.copy()
    test_pred_df["pred_risk_label"] = xgb_model.predict(X_test)
    taxonomy_res = analyze_error_taxonomy_breakdown(test_pred_df)

    # 9. Consolidate & Persist Full Evaluation Report
    full_evaluation = {
        "benchmark_models": results,
        "ablation_study": ablation_res,
        "calibration_curve": calibration_data,
        "error_taxonomy_breakdown": taxonomy_res,
        "dataset_statistics": {
            "total_samples": len(train_df) + len(test_df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "features_used": ALL_FEATURES
        }
    }

    with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)
    print(f"\n[+] Full evaluation results exported to: {EVALUATION_RESULTS_FILE}")
    print("=" * 70)
    return full_evaluation

if __name__ == "__main__":
    run_training_and_evaluation()
