"""
Human-in-the-Loop Review Station Component (Step 8)
Interactive triage interface for MEDIUM and HIGH hallucination risk answers.
Enables human experts to review evidence, classify according to Sun et al. (2024) error taxonomy,
approve/reject answers, and log corrections.
Reference: Section 4.4 Step 8, Section 4.5 System Architecture
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.human_loop.review_manager import HumanReviewManager
from config.config import ERROR_TAXONOMY

def render_review_queue():
    st.subheader("🛡️ Human-in-the-Loop Review Workstation (Step 8)")
    st.caption("Human safety checkpoint for responses flagged as MEDIUM or HIGH hallucination risk.")

    manager = HumanReviewManager()
    manager.seed_demo_queue()
    all_items = manager.get_all_reviews()
    pending_items = manager.get_pending_reviews()

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Flagged Items", len(all_items))
    with col_stat2:
        st.metric("Awaiting Human Review", len(pending_items))
    with col_stat3:
        resolved_count = len(all_items) - len(pending_items)
        st.metric("Resolved / Verified", resolved_count)

    tab1, tab2 = st.tabs(["📋 Active Pending Triage Queue", "📜 Verification History & Audit Log"])

    with tab1:
        if not pending_items:
            st.success("✅ Triage queue is empty! No unreviewed high-risk responses pending.")
            st.info("Tip: In the 'Live Detector' tab, test a response that triggers HIGH risk and click 'Escalate to Human Review Queue' to add it here.")
        else:
            st.markdown(f"**Found {len(pending_items)} item(s) awaiting verification:**")
            
            for idx, item in enumerate(pending_items):
                color = "#dc3545" if item["risk_category"] == "HIGH" else "#ffc107"
                with st.expander(
                    f"Ticket [{item['review_id']}] — Risk: {item['risk_category']} ({item['risk_probability']*100:.1f}%) — Question: {item['question'][:60]}...",
                    expanded=(idx == 0)
                ):
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 12px;">
                        <p><b>Ticket ID:</b> <code>{item['review_id']}</code> | <b>Timestamp:</b> {item['timestamp']}</p>
                        <p><b>Flagged Risk Category:</b> <span style="color:{color}; font-weight:bold;">{item['risk_category']}</span> (Probability: {item['risk_probability']:.4f})</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c_q, c_ev = st.columns(2)
                    with c_q:
                        st.markdown("**User Question:**")
                        st.info(item["question"])
                        st.markdown("**Draft LLM Response (Flagged):**")
                        st.warning(item["response"])

                    with c_ev:
                        st.markdown("**Trustworthy Reference Evidence:**")
                        st.success(item["reference_evidence"])
                        st.markdown("**Primary Risk Drivers:**")
                        for d in item.get("risk_drivers", []):
                            st.write(f"• {d}")

                    st.markdown("---")
                    st.markdown("#### Human Reviewer Actions & Annotation")

                    c_act1, c_act2 = st.columns(2)
                    with c_act1:
                        selected_action = st.radio(
                            "Review Decision:",
                            ["REJECT (Suppress response)", "APPROVE (False alarm / safe to show)", "MODIFY (Correct and approve)"],
                            key=f"act_{item['review_id']}"
                        )
                        assigned_taxonomy = st.selectbox(
                            "Sun et al. (2024) Error Taxonomy Tag:",
                            ERROR_TAXONOMY,
                            key=f"tax_{item['review_id']}"
                        )

                    with c_act2:
                        corrected_text = ""
                        if "MODIFY" in selected_action:
                            corrected_text = st.text_area(
                                "Enter Corrected Grounded Response:",
                                value=item["response"],
                                key=f"corr_{item['review_id']}"
                            )
                        reviewer_notes = st.text_area(
                            "Reviewer Audit Notes / Justification:",
                            placeholder="Explain why this response was rejected, approved, or modified...",
                            key=f"notes_{item['review_id']}"
                        )

                    if st.button("💾 Submit Verification Decision", key=f"sub_{item['review_id']}", type="primary"):
                        decision_str = "REJECTED" if "REJECT" in selected_action else ("MODIFIED" if "MODIFY" in selected_action else "APPROVED")
                        manager.submit_review(
                            review_id=item["review_id"],
                            status=decision_str,
                            assigned_error_type=assigned_taxonomy,
                            reviewer_notes=reviewer_notes,
                            corrected_response=corrected_text
                        )
                        st.success(f"Ticket {item['review_id']} resolved as {decision_str}!")
                        st.rerun()

    with tab2:
        if not all_items:
            st.info("No audit history recorded yet.")
        else:
            history_rows = []
            for it in all_items:
                history_rows.append({
                    "Ticket ID": it["review_id"],
                    "Risk": it["risk_category"],
                    "Probability": f"{it['risk_probability']:.2f}",
                    "Status": it["status"],
                    "Taxonomy Tag": it.get("assigned_error_type", "N/A"),
                    "Reviewer Notes": it.get("reviewer_notes", ""),
                    "Resolved At": it.get("resolved_at", "Pending")
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
