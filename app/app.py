"""
Streamlit Web Application: AI Hallucination Risk Prediction
Course: 24DSP501 - Technical Communication in Data Science
Authors: Hemavathy K (Roll No: 21662), Madhumitha P (Roll No: 22286)
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.components.live_detector import render_live_detector
from app.components.benchmark_dashboard import render_benchmark_dashboard
from app.components.ablation_view import render_ablation_view
from app.components.review_queue import render_review_queue
from app.components.dataset_browser import render_dataset_browser

from config.config import (
    XGBOOST_MODEL_FILE, SCALER_FILE, REFERENCE_CORPUS_FILE,
    DATASET_LABELED_FILE, EVALUATION_RESULTS_FILE
)

st.set_page_config(
    page_title="AI Hallucination Risk Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 26px;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #e0e6ed !important;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main Header Banner
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Hallucination Risk Prediction Framework</h1>
    <p><b>Using Semantic Consistency, Evidence Support, and Uncertainty Estimation with Derived Gap Signals</b></p>
    <div>
        <span class="badge">Course: 24DSP501 Technical Communication in Data Science</span>
        <span class="badge">Team: Hemavathy K (Roll: 21662) & Madhumitha P (Roll: 22286)</span>
        <span class="badge">Primary Model: XGBoost</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=120)
    st.title("Navigation")
    menu = st.radio(
        "Select Module:",
        [
            "🎯 Live Checkpoint (Demo)",
            "📈 Benchmark & Calibration",
            "🔬 Ablation Study (Core vs 7)",
            "🛡️ Human Review Station (Step 8)",
            "📚 Dataset & Knowledge Corpus"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("System Status")
    
    models_ready = XGBOOST_MODEL_FILE.exists() and SCALER_FILE.exists()
    corpus_ready = REFERENCE_CORPUS_FILE.exists()
    dataset_ready = DATASET_LABELED_FILE.exists()

    st.write(f"• **ML Classifiers:** {'✅ Ready' if models_ready else '⚠️ Untrained'}")
    st.write(f"• **Reference Corpus:** {'✅ Indexed' if corpus_ready else '❌ Missing'}")
    st.write(f"• **Labeled Dataset:** {'✅ 450 Samples' if dataset_ready else '❌ Missing'}")

    st.markdown("---")
    if st.button("🔄 Retrain Models & Rerun Pipeline"):
        with st.spinner("Retraining all models, running ablation & computing metrics..."):
            from experiments.train_and_evaluate import run_training_and_evaluation
            run_training_and_evaluation()
            st.success("Models retrained and evaluated successfully!")
            st.rerun()

    st.markdown("---")
    st.caption("24DSP501 Project Implementation | Fully Grounded & Calibrated")

# Route to selected module
if menu == "🎯 Live Checkpoint (Demo)":
    render_live_detector()
elif menu == "📈 Benchmark & Calibration":
    render_benchmark_dashboard()
elif menu == "🔬 Ablation Study (Core vs 7)":
    render_ablation_view()
elif menu == "🛡️ Human Review Station (Step 8)":
    render_review_queue()
elif menu == "📚 Dataset & Knowledge Corpus":
    render_dataset_browser()
