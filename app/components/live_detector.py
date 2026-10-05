"""
Live Hallucination Checkpoint & Interactive Predictor
Implements the system architecture described in Figure 1 of the paper.
"""

import streamlit as st
import json
from src.retrieval.retriever import EvidenceRetriever
from src.generator.generator import RAGResponseGenerator
from src.models.risk_scorer import HallucinationRiskScorer
from src.human_loop.review_manager import HumanReviewManager
from config.config import REFERENCE_CORPUS_FILE, CURATED_QA_FILE
from app.utils import render_risk_gauge, render_feature_bar_chart, render_radar_chart

MODEL_NAME_MAP = {
    "XGBoost (Primary)": "XGBoost",
    "Random Forest": "Random Forest",
    "SVM": "SVM",
    "Logistic Regression": "Logistic Regression",
}


@st.cache_resource
def get_retriever():
    path = str(REFERENCE_CORPUS_FILE) if REFERENCE_CORPUS_FILE.exists() else None
    return EvidenceRetriever(path)


@st.cache_resource
def get_scorer(model_name: str):
    return HallucinationRiskScorer(model_name=model_name)


def _init_defaults(question: str, evidence: str, response: str, sample1: str, sample2: str):
    defaults = {
        "question_input": question,
        "evidence_input": evidence,
        "response_input": response,
        "sample1": sample1,
        "sample2": sample2,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_live_detector():
    st.subheader("🎯 Step-by-Step Live Hallucination Checkpoint")
    st.caption("Inspects draft LLM answers before they reach the user (Figure 1: End-to-End Architecture)")

    retriever = get_retriever()
    generator = RAGResponseGenerator()
    review_manager = HumanReviewManager()

    qa_presets = []
    if CURATED_QA_FILE.exists():
        with open(CURATED_QA_FILE, "r", encoding="utf-8") as f:
            qa_presets = json.load(f)

    st.markdown("#### 1. Input Question & Reference Evidence")
    col_preset, col_model = st.columns([3, 1])

    with col_model:
        selected_model = st.selectbox(
            "Classification Model",
            list(MODEL_NAME_MAP.keys()),
            index=0,
            help="Choose the model to score hallucination risk"
        )
        model_key = MODEL_NAME_MAP[selected_model]
        scorer = get_scorer(model_key)

    preset_options = ["Custom Input"] + [f"[{item['chunk_id']}] {item['question']}" for item in qa_presets]
    selected_preset = col_preset.selectbox("Load Curated Preset Case or Choose Custom", preset_options)

    default_q = "How does the CRISPR-Cas9 system identify its DNA target and create genomic edits?"
    default_ev = retriever.chunks[0]["text"] if retriever.chunks else ""
    default_resp = (
        "CRISPR-Cas9 uses a single guide RNA containing a 20-nucleotide sequence that matches the "
        "target genomic locus directly adjacent to a PAM sequence, typically 5'-NGG-3'. The Cas9 "
        "endonuclease generates double-strand breaks, which are subsequently resolved by cellular "
        "repair mechanisms like NHEJ or HDR."
    )
    default_s1 = "According to source documentation, CRISPR-Cas9 targets DNA matching its 20-nt guide RNA."
    default_s2 = "The reference confirms Cas9 cuts adjacent to a PAM sequence before DNA repair takes over."

    if selected_preset != "Custom Input":
        idx = preset_options.index(selected_preset) - 1
        item = qa_presets[idx]
        default_q = item["question"]
        default_ev = retriever.get_best_evidence(default_q)
        default_resp = item["factual"]
        default_s1 = f"Verified records state that {item['factual']}"
        default_s2 = f"As noted in the reference evidence: {item['factual']}"
    elif retriever.chunks:
        default_ev = retriever.chunks[0]["text"]

    if st.session_state.get("last_preset") != selected_preset:
        st.session_state.last_preset = selected_preset
        st.session_state.question_input = default_q
        st.session_state.evidence_input = default_ev
        st.session_state.response_input = default_resp
        st.session_state.sample1 = default_s1
        st.session_state.sample2 = default_s2

    _init_defaults(default_q, default_ev, default_resp, default_s1, default_s2)

    col_q, col_ev = st.columns(2)
    with col_q:
        st.text_area("User Question:", height=110, key="question_input")
        auto_retrieve = st.button("🔍 Auto-Retrieve Relevant Evidence from Knowledge Corpus")

    with col_ev:
        if auto_retrieve:
            st.session_state.evidence_input = retriever.get_best_evidence(st.session_state.question_input)
        st.text_area(
            "Reference Evidence Passage (Trustworthy Ground Truth):",
            height=110,
            key="evidence_input"
        )

    st.markdown("---")
    st.markdown("#### 2. Generated LLM Response & Sampling")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    item = None
    if selected_preset != "Custom Input":
        idx = preset_options.index(selected_preset) - 1
        item = qa_presets[idx]
        if col_btn1.button("✨ Load Factual Response (LOW Risk)"):
            st.session_state.response_input = item["factual"]
            st.session_state.sample1 = f"Verified records state that {item['factual']}"
            st.session_state.sample2 = f"As noted in the reference evidence: {item['factual']}"
            st.rerun()
        if col_btn2.button("⚠️ Load Partially Supported (MEDIUM Risk)"):
            st.session_state.response_input = item["partial"]
            st.session_state.sample1 = f"While confirmed in part, {item['partial']}"
            st.session_state.sample2 = f"Research suggests that {item['partial']}"
            st.rerun()
        if col_btn3.button("🚨 Load Confident Hallucination (HIGH Risk)"):
            st.session_state.response_input = item["hallucinated"]
            st.session_state.sample1 = f"It is officially certified that {item['hallucinated']}"
            st.session_state.sample2 = f"Authoritative reports verify that {item['hallucinated']}"
            st.rerun()

    if col_btn4.button("🧪 Simulate RAG Draft"):
        draft = generator.generate_response(
            st.session_state.question_input,
            st.session_state.evidence_input,
            variant_type="factual"
        )
        st.session_state.response_input = draft["response"]
        samples = draft.get("sampled_responses", [])
        if samples:
            st.session_state.sample1 = samples[0]
            st.session_state.sample2 = samples[1] if len(samples) > 1 else samples[0]
        st.rerun()

    st.text_area("LLM Generated Answer to Evaluate:", height=100, key="response_input")

    with st.expander("🔬 Advanced: Multi-Sample Completions (for Consistency & Semantic Entropy)", expanded=False):
        st.caption("The system samples 2-3 repeated responses to estimate semantic consistency and entropy (SelfCheckGPT & Farquhar et al.)")
        st.text_input("Sample 1:", key="sample1")
        st.text_input("Sample 2:", key="sample2")

    evaluate_btn = st.button("🚀 Analyze Hallucination Risk Checkpoint", type="primary", use_container_width=True)

    if evaluate_btn or "last_assessment" in st.session_state:
        if evaluate_btn:
            sample1 = st.session_state.get("sample1", "")
            sample2 = st.session_state.get("sample2", "")
            samples = [s for s in [sample1, sample2] if s]
            if not samples:
                samples = generator.generate_rephrased_samples(st.session_state.response_input)
            with st.spinner("Extracting 7 features and predicting risk probability..."):
                assessment = scorer.assess_response(
                    question=st.session_state.question_input,
                    reference_evidence=st.session_state.evidence_input,
                    response=st.session_state.response_input,
                    sampled_responses=samples
                )
                st.session_state["last_assessment"] = assessment
        else:
            assessment = st.session_state["last_assessment"]

        st.markdown("---")
        st.markdown("### 📊 Checkpoint Decision & Risk Assessment")

        col_metric, col_gauge = st.columns([1, 1])

        with col_metric:
            st.markdown(f"""
            <div style="padding: 16px; border-radius: 8px; border: 2px solid {assessment['color']}; background: rgba(0,0,0,0.02);">
                <h3 style="color: {assessment['color']}; margin-top:0;">Risk Category: <b>{assessment['risk_category']}</b></h3>
                <h4>Action: <code>{assessment['action']}</code></h4>
                <p><b>Decision:</b> {assessment['action_desc']}</p>
                <hr style="margin: 10px 0;">
                <p><b>Model:</b> {assessment['model_used']}</p>
                <p><b>Hallucination Probability:</b> <code>{assessment['risk_probability']:.4f} ({assessment['risk_probability']*100:.1f}%)</code></p>
                <p><b>Class Distribution:</b> LOW: {assessment['probabilities']['LOW']:.2f} | MED: {assessment['probabilities']['MEDIUM']:.2f} | HIGH: {assessment['probabilities']['HIGH']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            if assessment["risk_category"] in ["MEDIUM", "HIGH"]:
                st.markdown("#### 🚨 Primary Hallucination Drivers:")
                for d in assessment["risk_drivers"]:
                    st.warning(f"• {d}")

                if st.button("📋 Escalate to Human Review Queue (Step 8)", key="btn_esc"):
                    rev_id = review_manager.enqueue_for_review(assessment)
                    st.success(f"Successfully enqueued response under Ticket ID: `{rev_id}`. Check the Human Review Workstation tab!")

        with col_gauge:
            fig_gauge = render_risk_gauge(assessment["risk_probability"], assessment["risk_category"], assessment["color"])
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 🔍 Breakdown of the 7 Signals (4 Core + 3 Add-on Features)")
        col_bar, col_radar = st.columns([1.2, 0.8])
        with col_bar:
            fig_bar = render_feature_bar_chart(assessment["features"])
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_radar:
            fig_radar = render_radar_chart(assessment["features"])
            st.plotly_chart(fig_radar, use_container_width=True)

        feats = assessment["features"]
        exp_data = [
            {"Signal": "1. Semantic Consistency", "Type": "Core", "Value": f"{feats['semantic_consistency']:.4f}", "Interpretation": "Stability across repeated samplings (SelfCheckGPT)"},
            {"Signal": "2. Evidence Support", "Type": "Core", "Value": f"{feats['evidence_support']:.4f}", "Interpretation": "NLI Entailment probability against reference passage"},
            {"Signal": "3. Contradiction Score", "Type": "Core", "Value": f"{feats['contradiction_score']:.4f}", "Interpretation": "NLI Contradiction probability against evidence"},
            {"Signal": "4. Response Uncertainty", "Type": "Core", "Value": f"{feats['response_uncertainty']:.4f}", "Interpretation": "Semantic entropy and token spread (Farquhar et al.)"},
            {"Signal": "5. Claim-Evidence Gap", "Type": "Add-on (New)", "Value": f"{feats['claim_evidence_gap']:.4f}", "Interpretation": "Topical similarity minus true logical entailment"},
            {"Signal": "6. Confidence-Factuality Gap", "Type": "Add-on (New)", "Value": f"{feats['confidence_factuality_gap']:.4f}", "Interpretation": "Model self-confidence minus actual evidence support"},
            {"Signal": "7. Semantic-Evidence Interaction", "Type": "Add-on (New)", "Value": f"{feats['semantic_evidence_interaction']:.4f}", "Interpretation": "Consistency multiplied by factual grounding"}
        ]
        st.dataframe(exp_data, use_container_width=True)
