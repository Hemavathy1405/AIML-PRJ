"""
Model Training & Classification Module
Trains XGBoost (primary) and baselines (SVM, Random Forest, Logistic Regression)
Reference: Section 4.4 Step 6 of the Project Report
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)

try:
    from config.config import (
        ALL_FEATURES, CORE_FEATURES, ADDON_FEATURES,
        XGBOOST_MODEL_FILE, SVM_MODEL_FILE, RF_MODEL_FILE, LR_MODEL_FILE,
        SCALER_FILE, MODEL_METADATA_FILE, RANDOM_SEED
    )
except (ImportError, ValueError):
    from ...config.config import (
        ALL_FEATURES, CORE_FEATURES, ADDON_FEATURES,
        XGBOOST_MODEL_FILE, SVM_MODEL_FILE, RF_MODEL_FILE, LR_MODEL_FILE,
        SCALER_FILE, MODEL_METADATA_FILE, RANDOM_SEED
    )

class HallucinationClassifierTrainer:
    """Manages training and saving of XGBoost and baseline models."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_set: str = "all"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Scales features and prepares X, y matrices."""
        features = ALL_FEATURES if feature_set == "all" else CORE_FEATURES
        
        X_train_raw = train_df[features].values
        y_train = train_df["risk_label"].values

        X_test_raw = test_df[features].values
        y_test = test_df["risk_label"].values

        if feature_set == "all":
            X_train = self.scaler.fit_transform(X_train_raw)
            X_test = self.scaler.transform(X_test_raw)
        else:
            # Scaler specifically for ablation
            core_scaler = StandardScaler()
            X_train = core_scaler.fit_transform(X_train_raw)
            X_test = core_scaler.transform(X_test_raw)

        return X_train, y_train, X_test, y_test, features

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Instantiates and trains XGBoost, SVM, RF, and Logistic Regression."""
        # 1. XGBoost (Primary Classifier)
        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            random_state=self.random_seed,
            eval_metric="mlogloss",
            n_jobs=1
        )
        xgb.fit(X_train, y_train)

        # 2. Random Forest Baseline
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            random_state=self.random_seed
        )
        rf.fit(X_train, y_train)

        # 3. Support Vector Machine (SVM) Baseline
        svm = SVC(
            C=1.5,
            kernel="rbf",
            probability=True,
            random_state=self.random_seed
        )
        svm.fit(X_train, y_train)

        # 4. Logistic Regression Baseline
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed
        )
        lr.fit(X_train, y_train)

        self.models = {
            "XGBoost": xgb,
            "Random Forest": rf,
            "SVM": svm,
            "Logistic Regression": lr
        }
        return self.models

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates multi-class and binary hallucination metrics + calibration."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Multi-class ROC-AUC (OvR)
        try:
            roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            roc_auc_ovr = 0.0

        # Binary conversion for Brier Score & False Negative analysis (0=LOW vs 1=MEDIUM/HIGH)
        y_test_binary = (y_test > 0).astype(int)
        # Probability of being hallucinated (MEDIUM or HIGH) = P(MED) + P(HIGH)
        prob_hallucination = y_proba[:, 1] + y_proba[:, 2]
        brier = brier_score_loss(y_test_binary, prob_hallucination)

        # Count false negatives: actual hallucination (MED or HIGH) predicted as LOW
        fn_count = int(np.sum((y_test > 0) & (y_pred == 0)))
        total_positives = int(np.sum(y_test > 0))
        fn_rate = (fn_count / total_positives) if total_positives > 0 else 0.0
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

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

    def save_artifacts(self, feature_names: list):
        """Persists trained models and scaler."""
        joblib.dump(self.scaler, SCALER_FILE)
        joblib.dump(self.models["XGBoost"], XGBOOST_MODEL_FILE)
        joblib.dump(self.models["SVM"], SVM_MODEL_FILE)
        joblib.dump(self.models["Random Forest"], RF_MODEL_FILE)
        joblib.dump(self.models["Logistic Regression"], LR_MODEL_FILE)

        # Extract feature importances from XGBoost
        xgb = self.models["XGBoost"]
        importances = dict(zip(feature_names, [round(float(v), 4) for v in xgb.feature_importances_]))

        metadata = {
            "primary_model": "XGBoost",
            "baselines": ["Random Forest", "SVM", "Logistic Regression"],
            "features": feature_names,
            "feature_importances": importances,
            "classes": ["LOW", "MEDIUM", "HIGH"]
        }
        with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
