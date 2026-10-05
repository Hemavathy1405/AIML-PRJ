"""
Ablation Study View Component
Side-by-side comparison of Core 4 Features vs Core 4 + 3 Add-on Derived Features (All 7 Features).
Validates Section 6 Expected Outcomes:
'Evidence that the three add-on features improve detection of confidently-wrong answers compared to the four core features alone.'
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from config.config import EVALUATION_RESULTS_FILE

def render_ablation_view():
    st.subheader("🔬 Ablation Study: Impact of the 3 Add-on Derived Features")
    st.markdown("""
    This ablation study evaluates the paper's core hypothesis:
    > *Do the 3 add-on derived features (**Claim–Evidence Alignment Gap**, **Confidence–Factuality Gap**, and **Semantic–Evidence Interaction**) detect subtle, confident hallucinations that the 4 basic features miss?*
    """)

    if not EVALUATION_RESULTS_FILE.exists():
        st.warning("Evaluation results not found. Run model training first.")
        return

    with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    ablation = eval_data.get("ablation_study", {})
    if not ablation:
        st.info("No ablation data available.")
        return

    core4 = ablation.get("baseline_core_4", {})
    all7 = ablation.get("proposed_all_7", {})
    deltas = ablation.get("deltas", {})

    st.markdown("#### 1. Performance Comparison: Baseline (4 Core) vs Proposed (7 Features)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{all7.get('accuracy', 0):.4f}",
            delta=f"{deltas.get('accuracy_change', 0):+.4f}"
        )
    with col2:
        st.metric(
            label="F1-Macro Score",
            value=f"{all7.get('f1_macro', 0):.4f}",
            delta=f"{deltas.get('f1_macro_change', 0):+.4f}"
        )
    with col3:
        st.metric(
            label="ROC-AUC (OvR)",
            value=f"{all7.get('roc_auc_ovr', 0):.4f}",
            delta=f"{deltas.get('roc_auc_change', 0):+.4f}"
        )
    with col4:
        fn_red = deltas.get("false_negatives_reduced", 0)
        st.metric(
            label="False Negatives",
            value=f"{all7.get('false_negatives', 0)}",
            delta=f"-{fn_red}" if fn_red > 0 else f"{fn_red}",
            delta_color="inverse"
        )

    # Comparison Table
    st.markdown("#### 2. Detailed Metric Matrix")
    metrics_table = [
        {
            "Feature Configuration": "Baseline: 4 Core Features (Task 1)",
            "Features Included": "Consistency, Evidence Support, Contradiction, Uncertainty",
            "Accuracy": f"{core4.get('accuracy', 0):.4f}",
            "F1-Macro": f"{core4.get('f1_macro', 0):.4f}",
            "ROC-AUC": f"{core4.get('roc_auc_ovr', 0):.4f}",
            "Brier Score (Calib)": f"{core4.get('brier_score', 0):.4f}",
            "False Negatives": core4.get("false_negatives", 0)
        },
        {
            "Feature Configuration": "⭐ Proposed: All 7 Features (Task 2 Contribution)",
            "Features Included": "4 Core + Claim Gap + Conf-Factuality Gap + Sem-Evid Interaction",
            "Accuracy": f"{all7.get('accuracy', 0):.4f}",
            "F1-Macro": f"{all7.get('f1_macro', 0):.4f}",
            "ROC-AUC": f"{all7.get('roc_auc_ovr', 0):.4f}",
            "Brier Score (Calib)": f"{all7.get('brier_score', 0):.4f}",
            "False Negatives": all7.get("false_negatives", 0)
        }
    ]
    st.dataframe(pd.DataFrame(metrics_table), use_container_width=True)

    # 3. Bar Chart Comparison
    st.markdown("#### 3. Signal Architecture Breakdown")
    col_desc1, col_desc2 = st.columns(2)

    with col_desc1:
        st.info("""
        **Why Core Features Alone Fall Short:**
        - **Superficial Topicality:** A hallucinated response that uses the same keywords as the evidence gets high lexical/semantic similarity.
        - **Confident Confabulation:** An LLM can produce fluent nonsense with high internal token probabilities (low uncertainty).
        - **Absence of Direct Negation:** Hallucinations rarely use words like 'NOT'; they invent new plausible entities, so raw NLI contradiction scores can stay low.
        """)

    with col_desc2:
        st.success("""
        **How the 3 Add-on Features Close the Gap:**
        1. **Claim–Evidence Alignment Gap:** Flags when a response sounds related to the topic but is not logically entailed ($Sim - Support$).
        2. **Confidence–Factuality Gap:** Catches dangerous cases where model confidence is high while evidence support is low ($Confidence - Support$).
        3. **Semantic–Evidence Interaction:** Verifies that answer consistency across samples is truly anchored in the source text ($Consistency \\times Support$).
        """)

    st.markdown("#### 4. Scientific Conclusion & Paper Validation")
    st.success(ablation.get("conclusion", "The add-on features validate the research hypothesis."))
