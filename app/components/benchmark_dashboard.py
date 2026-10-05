"""
Model Benchmark & Performance Dashboard
Displays metrics across XGBoost (Primary), SVM, Random Forest, and Logistic Regression
Includes Confusion Matrices, Calibration Curves, and Feature Importance.
Reference: Section 4.4 Step 6, Section 5 Evaluation Plan
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config.config import EVALUATION_RESULTS_FILE, MODEL_METADATA_FILE

def render_benchmark_dashboard():
    st.subheader("📈 Multi-Model Benchmark & Calibration Dashboard")
    st.caption("Comparison of Primary Classifier (XGBoost) against Baselines (Random Forest, SVM, Logistic Regression)")

    if not EVALUATION_RESULTS_FILE.exists():
        st.warning("⚠️ Model evaluation results not found. Please train models first using the pipeline.")
        return

    with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    models_data = eval_data.get("benchmark_models", {})

    # 1. Benchmark Comparison Table
    st.markdown("#### 1. Performance Comparison (Held-out Test Set)")
    table_rows = []
    for model_name, m in models_data.items():
        table_rows.append({
            "Classifier Model": f"⭐ {model_name} (Primary)" if model_name == "XGBoost" else f"{model_name} (Baseline)",
            "Accuracy": f"{m['accuracy']:.4f}",
            "F1-Macro": f"{m['f1_macro']:.4f}",
            "F1-Weighted": f"{m['f1_weighted']:.4f}",
            "ROC-AUC (OvR)": f"{m['roc_auc_ovr']:.4f}",
            "Brier Score (Calib)": f"{m['brier_score']:.4f}",
            "False Negatives": m['false_negatives'],
            "FN Rate": f"{m['false_negative_rate'] * 100:.1f}%"
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True)

    # 2. Bar Chart Comparison
    st.markdown("#### 2. Key Metrics Visualization")
    chart_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": m["accuracy"],
            "F1-Macro": m["f1_macro"],
            "ROC-AUC": m["roc_auc_ovr"]
        }
        for name, m in models_data.items()
    ])
    melted = chart_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    fig_bench = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Benchmark: Accuracy, F1-Macro, and ROC-AUC",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_bench.update_layout(yaxis=dict(range=[0.5, 1.05]), height=380)
    st.plotly_chart(fig_bench, use_container_width=True)

    # 3. Confusion Matrix & Calibration Curve
    col_cm, col_cal = st.columns(2)

    with col_cm:
        st.markdown("#### 3. Confusion Matrix")
        chosen_m = st.selectbox("Select Model for Confusion Matrix:", list(models_data.keys()), index=0)
        cm_matrix = models_data[chosen_m].get("confusion_matrix")
        if not cm_matrix:
            st.info("Retrain models to store confusion matrices. Showing a placeholder until then.")
            cm_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        labels = ["LOW (0)", "MEDIUM (1)", "HIGH (2)"]
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Risk Category", y="Actual Ground Truth", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {chosen_m}"
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cal:
        st.markdown("#### 4. Probability Calibration (Reliability Curve)")
        calib_data = eval_data.get("calibration_curve", {})
        prob_pred = calib_data.get("prob_pred", [])
        prob_true = calib_data.get("prob_true", [])

        fig_cal = go.Figure()
        # Perfectly calibrated diagonal
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        # Actual model calibration
        fig_cal.add_trace(go.Scatter(
            x=prob_pred, y=prob_true,
            mode='lines+markers',
            name='XGBoost Risk Calibration',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8)
        ))
        fig_cal.update_layout(
            title=f"Reliability Diagram (Brier Score: {models_data.get('XGBoost', {}).get('brier_score', 0):.4f})",
            xaxis=dict(title="Mean Predicted Risk Probability", range=[0, 1]),
            yaxis=dict(title="Empirical Fraction of Positives", range=[0, 1]),
            height=350
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # 4. Feature Importance Analysis
    st.markdown("#### 5. XGBoost Feature Importance Breakdown")
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fi = meta.get("feature_importances", {})
        if fi:
            fi_df = pd.DataFrame([
                {"Feature": k, "Importance": v, "Type": "Add-on (New)" if "gap" in k or "interaction" in k else "Core Signal"}
                for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)
            ])
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Type",
                title="Gini Feature Importance in Primary XGBoost Classifier",
                color_discrete_map={"Core Signal": "#007bff", "Add-on (New)": "#e83e8c"}
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=320)
            st.plotly_chart(fig_fi, use_container_width=True)
