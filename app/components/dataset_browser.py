"""
Dataset & Knowledge Corpus Browser Component
Enables interactive exploration of the reference knowledge corpus, QA pairs,
extracted 7 signals, and Sun et al. (2024) error annotations.
Reference: Section 4.1 Dataset & Section 6 Reusable Labelled Dataset
"""

import streamlit as st
import json
import pandas as pd
from config.config import REFERENCE_CORPUS_FILE, DATASET_LABELED_FILE, TRAIN_FEATURES_FILE, TEST_FEATURES_FILE

def render_dataset_browser():
    st.subheader("📚 Reference Knowledge Corpus & Labeled Dataset Browser")
    st.caption("Explore verified evidence chunks, question-answer pairs, extracted signals, and labels.")

    tab_corpus, tab_dataset = st.tabs(["🏛️ Reference Evidence Corpus", "📊 Labeled Evaluation Dataset"])

    with tab_corpus:
        st.markdown("#### Reference Knowledge Passages")
        st.markdown("These passages represent verified, trustworthy ground-truth knowledge used for retrieval and evidence entailment.")
        
        if REFERENCE_CORPUS_FILE.exists():
            with open(REFERENCE_CORPUS_FILE, "r", encoding="utf-8") as f:
                corpus = json.load(f)

            domains = list({c["domain"] for c in corpus})
            selected_domain = st.selectbox("Filter by Scientific Domain:", ["All Domains"] + domains)

            filtered_corpus = [c for c in corpus if selected_domain == "All Domains" or c["domain"] == selected_domain]

            st.write(f"Showing {len(filtered_corpus)} reference document(s):")
            for c in filtered_corpus:
                with st.expander(f"[{c['chunk_id']}] {c['title']} — Domain: {c['domain']}"):
                    st.markdown(f"**Document Content:**")
                    st.info(c["text"])
        else:
            st.warning("Reference corpus file not found.")

    with tab_dataset:
        st.markdown("#### Labeled 7-Feature Evaluation Dataset")
        st.markdown("Includes factual (LOW), partially supported (MEDIUM), and contradicted/hallucinated (HIGH) response variants.")

        if DATASET_LABELED_FILE.exists():
            df = pd.read_csv(DATASET_LABELED_FILE)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                cat_filter = st.selectbox("Filter by Risk Category:", ["All Categories"] + list(df["risk_category"].unique()))
            with col_f2:
                dom_filter = st.selectbox("Filter by Domain:", ["All Domains"] + list(df["domain"].unique()))
            with col_f3:
                tax_filter = st.selectbox("Filter by Error Taxonomy:", ["All Error Types"] + list(df["error_taxonomy"].unique()))

            filtered_df = df.copy()
            if cat_filter != "All Categories":
                filtered_df = filtered_df[filtered_df["risk_category"] == cat_filter]
            if dom_filter != "All Domains":
                filtered_df = filtered_df[filtered_df["domain"] == dom_filter]
            if tax_filter != "All Error Types":
                filtered_df = filtered_df[filtered_df["error_taxonomy"] == tax_filter]

            st.write(f"Showing **{len(filtered_df)}** of {len(df)} records:")

            display_cols = [
                "record_id", "domain", "risk_category", "error_taxonomy",
                "semantic_consistency", "evidence_support", "contradiction_score",
                "response_uncertainty", "claim_evidence_gap", "confidence_factuality_gap",
                "semantic_evidence_interaction"
            ]
            st.dataframe(filtered_df[display_cols], use_container_width=True)

            # Sample deep dive
            st.markdown("---")
            st.markdown("#### 🔎 Inspect Selected Record Text & Grounding")
            selected_rec_id = st.selectbox("Select Record ID to Inspect:", filtered_df["record_id"].tolist())
            if selected_rec_id:
                row = df[df["record_id"] == selected_rec_id].iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Question:**\n{row['question']}")
                    st.markdown(f"**Generated Response:**\n{row['response']}")
                with c2:
                    st.markdown(f"**Reference Evidence:**\n{row['reference_evidence']}")
                    st.markdown(f"**Error Category:** `{row['error_taxonomy']}` | **Risk:** `{row['risk_category']}`")

            # CSV Download
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Labeled Dataset (CSV)",
                data=csv_data,
                file_name="ai_hallucination_dataset_labeled.csv",
                mime="text/csv"
            )
        else:
            st.warning("Labeled dataset not found.")
