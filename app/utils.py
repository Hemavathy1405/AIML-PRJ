"""
Streamlit UI Utilities and Chart Helpers
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def render_risk_gauge(probability: float, category: str, color: str):
    """Generates an attractive Plotly Gauge indicator for hallucination risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Probability: <b>{category}</b>", 'font': {'size': 20, 'color': color}},
        number={'suffix': "%", 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(40, 167, 69, 0.2)'},
                {'range': [35, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(220, 53, 69, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_feature_bar_chart(features: Dict[str, float]):
    """Renders horizontal bar chart breaking down the 7 features."""
    names_map = {
        "semantic_consistency": "1. Semantic Consistency (Core)",
        "evidence_support": "2. Evidence Support (Core)",
        "contradiction_score": "3. Contradiction Score (Core)",
        "response_uncertainty": "4. Response Uncertainty (Core)",
        "claim_evidence_gap": "5. Claim-Evidence Gap (Add-on)",
        "confidence_factuality_gap": "6. Conf-Factuality Gap (Add-on)",
        "semantic_evidence_interaction": "7. Sem-Evidence Inter. (Add-on)"
    }
    labels = [names_map.get(k, k) for k in names_map.keys() if k in features]
    vals = [features[k] for k in names_map.keys() if k in features]
    
    # Colors: Green for safety, Red for risk indicators
    colors = [
        "#20c997", # consistency
        "#28a745", # support
        "#dc3545", # contradiction
        "#fd7e14", # uncertainty
        "#e83e8c", # claim gap
        "#d63384", # confidence gap
        "#17a2b8"  # interaction
    ]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation='h',
        marker=dict(color=colors[:len(vals)]),
        text=[f"{v:.4f}" for v in vals],
        textposition='outside'
    ))
    fig.update_layout(
        title="7 Hallucination Signals (Core vs Add-on Derived Features)",
        xaxis=dict(range=[0, 1.1], title="Signal Magnitude [0.0 - 1.0]"),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def render_radar_chart(features: Dict[str, float]):
    """Generates radar chart comparing signal profile."""
    keys = [
        "semantic_consistency", "evidence_support", "contradiction_score",
        "response_uncertainty", "claim_evidence_gap", "confidence_factuality_gap",
        "semantic_evidence_interaction"
    ]
    labels = [
        "Consistency", "Support", "Contradiction",
        "Uncertainty", "Claim Gap", "Conf Gap", "Interaction"
    ]
    vals = [features.get(k, 0.0) for k in keys]
    # close loop
    vals.append(vals[0])
    labels.append(labels[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=vals,
        theta=labels,
        fill='toself',
        fillcolor='rgba(0, 123, 255, 0.25)',
        line=dict(color='#007bff', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=320,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    return fig
