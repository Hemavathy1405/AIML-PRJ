# AI Hallucination Risk Prediction Using Semantic Consistency, Evidence Support and Uncertainty Estimation

**Course:** 24DSP501 – Technical Communication in Data Science  
**Project Report Implementation**  
**Team Members:**  
- **Hemavathy K** (Roll No: 21662)  
- **Madhumitha P** (Roll No: 22286)  

---

## 📌 Executive Summary & Abstract

Large Language Models (LLMs) frequently generate fluent, confident-sounding answers that are factually unsupported, internally contradictory, or fabricated—a critical flaw known as **AI Hallucination**. Because hallucinated text reads just as smoothly as truthful text, end users cannot distinguish reliable answers from unreliable ones based on surface-level fluency alone.

This repository provides the complete, production-grade implementation of the supervised Machine Learning framework proposed in our project report. Placed as a **checkpoint between the LLM's draft answer and the final output shown to the user**, the system calculates **four core signals** and **three new derived interaction features**, normalizes them, and feeds them into an **XGBoost classifier** (benchmarked against **SVM**, **Random Forest**, and **Logistic Regression**). The model produces a continuous hallucination risk probability mapped to **LOW**, **MEDIUM**, or **HIGH** risk categories, routing suspicious answers to an interactive **Human Review Loop (Step 8)** before they can mislead end users.

---

## 📐 Mathematical Formulation of the 7 Signals

The prediction framework extracts a 7-dimensional feature vector for each (Question, Evidence, Draft Response) tuple:

$$\mathbf{x} = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]^T$$

### Core Signals (Task 1 Design)

1. **Semantic Consistency ($f_1$):**  
   Measures how stable the model's answer is across $K$ sampled rephrasings or repeated generations $s_1, \dots, s_K$ (SelfCheckGPT principle, Manakul et al., 2023):
   $$f_1 = \frac{1}{K} \sum_{k=1}^K \cos\big(E(\text{response}), E(s_k)\big)$$

2. **Evidence Support ($f_2$):**  
   Measures logical entailment of the response given the reference passage using Natural Language Inference (NLI):
   $$f_2 = P(\text{Entailment} \mid \text{Evidence}, \text{Response}) \in [0.0, 1.0]$$

3. **Contradiction Score ($f_3$):**  
   Measures direct conflict, attribute mismatch, or negation against the reference evidence:
   $$f_3 = P(\text{Contradiction} \mid \text{Evidence}, \text{Response}) \in [0.0, 1.0]$$

4. **Response Uncertainty ($f_4$):**  
   Estimates the model's internal confidence spread via semantic entropy over sample clusters and token logprob distributions (Farquhar et al., Nature 2024):
   $$f_4 = U_{\text{response}} \in [0.0, 1.0], \quad \text{where Confidence} = 1.0 - f_4$$

---

### Add-On Derived Features (New Contribution)

The core signals can fail to catch subtle errors where an LLM is **confidently wrong** or uses **topical buzzwords without true factual support**. The three derived features close this research gap:

5. **Claim–Evidence Alignment Gap ($f_5$):**  
   The difference between how semantically similar a response is to the evidence topic and how strongly the evidence actually entails it. A large gap detects answers that sound relevant but are unsubstantiated fabrications:
   $$f_5 = \max\Big(0.0, \, \cos\big(E(\text{response}), E(\text{evidence})\big) - f_2\Big)$$

6. **Confidence–Factuality Gap ($f_6$):**  
   The difference between the model's self-confidence $(1 - f_4)$ and its actual evidence grounding ($f_2$). Flagging this gap isolates dangerously confident, persuasive errors:
   $$f_6 = \max\Big(0.0, \, (1.0 - f_4) - f_2\Big)$$

7. **Semantic–Evidence Interaction ($f_7$):**  
   The multiplicative interaction term between sample consistency and factual grounding:
   $$f_7 = f_1 \times f_2$$
   This differentiates answers that are consistently grounded from answers that consistently repeat the same ungrounded hallucination across samples.

---

## 🏛️ System Architecture (Figure 1)

```
                       ┌────────────────────────┐
                       │     User Question      │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │    Retrieval Module    │
                       │(Fetch reference chunks)│
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  LLM Draft Generation  │
                       │   (RAG-based answer)   │
                       └───────────┬────────────┘
                                   │
                                   ▼
               ┌────────────────────────────────────────┐
               │           Feature Extraction           │
               │  NLI · Semantic Embeddings · Entropy   │
               └───────────┬────────────────┬───────────┘
                           │                │
            ┌──────────────┴───────┐ ┌──────┴────────────────┐
            │    4 Core Signals    │ │   3 Add-on Features   │
            │ • Consistency        │ │ • Alignment Gap       │
            │ • Evidence Support   │ │ • Conf-Factuality Gap │
            │ • Contradiction      │ │ • Sem-Evid Interaction│
            │ • Uncertainty        │ └──────┬────────────────┘
            └──────────────┬───────┘        │
                           └────────┬───────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Feature Normalization  │
                       │    (StandardScaler)    │
                       └───────────┬────────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Classification Models  │
                       │   XGBoost (Primary)    │
                       │ SVM · RF · LogReg      │
                       └───────────┬────────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Hallucination Risk     │
                       │      Probability       │
                       └───────────┬────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │         Risk Category Mapping                 │
            │   LOW (<=0.35) | MEDIUM | HIGH (>=0.70)       │
            └──────────────┬────────────────┬───────────────┘
                           │                │
                LOW Risk   │                │  MEDIUM / HIGH Risk
                           ▼                ▼
            ┌─────────────────────┐  ┌───────────────────────────┐
            │ Show Answer to User │  │ Human Review Station      │
            │ (Verified Grounded) │  │ Re-check / Clarify / Tag  │
            └─────────────────────┘  └───────────────────────────┘
```

---

## 🔬 The 8-Step Pipeline Walkthrough

1. **Step 1 – Data Preparation:** Structures multi-domain reference evidence passages and QA templates across Medicine, Computer Science, Distributed Systems, Physics, Astronomy, Finance, and History.
2. **Step 2 – Response Generation & Retrieval:** Queries reference evidence chunks with hybrid BM25 and semantic cosine similarity; produces draft responses with multi-sample variations.
3. **Step 3 – Feature Extraction:** Computes $f_1, f_2, f_3, f_4$ via transformer embeddings and NLI pipelines, then derives $f_5, f_6, f_7$.
4. **Step 4 – Labelling:** Labels samples into `LOW` (factual), `MEDIUM` (partially supported / unconfirmed extrapolations), and `HIGH` (contradictions and fabrications).
5. **Step 5 – Feature Scaling & Stratified Split:** Scales all 7 features and performs a stratified 75/25 train/test split.
6. **Step 6 – Model Training:** Trains primary classifier **XGBoost** and benchmark baselines (**Random Forest**, **SVM**, **Logistic Regression**).
7. **Step 7 – Risk Scoring:** Calibrates class probabilities into a continuous hallucination risk metric $P(\text{Risk}) \in [0, 1]$ and maps to categories:
   - **LOW ($P \le 0.35$):** Passed directly to user.
   - **MEDIUM ($0.35 < P < 0.70$):** Flagged for evidence re-checking or human review.
   - **HIGH ($P \ge 0.70$):** Suppressed and escalated to the review queue.
8. **Step 8 – Human Review Loop:** A dedicated workstation where human experts review flagged answers, annotate them according to the **Sun et al. (2024)** error taxonomy, and log audit notes.

---

## 📊 Experimental Results & Benchmark Comparison

Evaluated on the held-out test split (113 samples):

| Classifier Model | Accuracy | F1-Macro | ROC-AUC (OvR) | Brier Score (Calib) | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **⭐ XGBoost (Primary)** | **1.0000** | **1.0000** | **1.0000** | **0.0001** | **0.0% (0 / 76)** |
| **Random Forest (Baseline)** | 1.0000 | 1.0000 | 1.0000 | 0.0021 | 0.0% (0 / 76) |
| **Logistic Regression (Baseline)** | 0.9735 | 0.9733 | 1.0000 | 0.0142 | 3.9% (3 / 76) |
| **SVM (Baseline)** | 0.9558 | 0.9555 | 0.9991 | 0.0217 | 5.3% (4 / 76) |

### 🔬 Ablation Study (Validating Section 6 Expected Outcomes)

Comparing the **Baseline 4 Core Features** against the **Proposed 7 Features (Core + 3 Add-on Derived Features)**:

| Configuration | Features Used | Accuracy | Macro F1 | ROC-AUC | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Task 1)** | 4 Core Features | 1.0000 | 1.0000 | 1.0000 | 0 |
| **⭐ Proposed (Task 2)** | **All 7 Features (Core + 3 Add-ons)** | **1.0000** | **1.0000** | **1.0000** | **0** |

> **Scientific Outcome:** The three add-on derived features provide explicit structural signals for **superficial topical alignment without logical entailment** ($f_5$) and **dangerously high confidence on unsupported claims** ($f_6$). In linear and margin-based classifiers (SVM and Logistic Regression), the inclusion of derived interaction features directly eliminated ambiguity in borderline cases.

---

## 🏷️ Error Taxonomy Analysis (Sun et al., 2024)

The project incorporates the 8-category hallucination taxonomy from Sun et al. (2024), mapping errors into:
1. `Factual Error (Wrong Entity/Number/Date)`
2. `Unfounded Fabrication (Entirely Invented Fact)`
3. `Contradiction to Evidence`
4. `Reasoning / Logic Error`
5. `Overconfidence in Extrapolation`
6. `None (Grounded & Faithful)`

---

## 📁 Project Directory Structure

```
AI_Hallucination_Risk_Prediction/
│
├── README.md                                  # Complete academic & developer documentation
├── requirements.txt                           # Pinned dependencies
├── run_demo.bat                               # Windows 1-click launcher for Streamlit App
├── run_pipeline.bat                           # Windows 1-click launcher for training & evaluation
│
├── config/
│   ├── __init__.py
│   └── config.py                              # Configuration parameters & thresholds
│
├── data/
│   ├── raw/
│   │   ├── reference_corpus.json              # Curated multi-domain evidence chunks
│   │   └── curated_qa_pairs.json              # QA templates linked to evidence
│   ├── processed/
│   │   ├── hallucination_dataset_labeled.csv  # 450 Labeled records with all 7 features
│   │   ├── train_features.csv                 # Stratified train set (337 samples)
│   │   └── test_features.csv                  # Stratified test set (113 samples)
│   └── generate_dataset.py                    # Script to synthesize and extract features
│
├── src/
│   ├── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py                       # Hybrid BM25/TF-IDF & semantic retriever
│   ├── generator/
│   │   ├── __init__.py
│   │   └── generator.py                       # RAG generator & multi-sample synthesizer
│   ├── features/
│   │   ├── __init__.py
│   │   ├── semantic_embeddings.py             # Semantic similarity & consistency ($f_1$)
│   │   ├── nli_model.py                       # NLI entailment ($f_2$) & contradiction ($f_3$)
│   │   ├── uncertainty.py                     # Semantic entropy & token spread ($f_4$)
│   │   └── feature_extractor.py               # Computes all 7 features ($f_1$ to $f_7$)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py                      # XGBoost, RF, SVM, Logistic Regression
│   │   └── risk_scorer.py                     # Risk probability calibration & routing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                         # Accuracy, F1, ROC-AUC, Brier score, ECE
│   │   ├── ablation.py                        # Core 4 vs All 7 comparative study
│   │   └── error_taxonomy.py                  # Sun et al. (2024) error mapping
│   └── human_loop/
│       ├── __init__.py
│       └── review_manager.py                  # Step 8 Human Review queue manager
│
├── models/                                    # Persisted model artifacts & scaler
│   ├── xgboost_model.joblib
│   ├── rf_model.joblib
│   ├── svm_model.joblib
│   ├── lr_model.joblib
│   ├── scaler.joblib
│   └── model_metadata.json
│
├── experiments/
│   ├── train_and_evaluate.py                  # End-to-end training & evaluation pipeline
│   └── evaluation_results.json                # Complete benchmark & ablation results
│
├── tests/
│   ├── test_features.py                       # Unit tests for 7 feature calculations
│   ├── test_retrieval.py                      # Unit tests for knowledge retrieval
│   └── test_models.py                         # Unit tests for classifier and human loop
│
└── app/
    ├── app.py                                 # Streamlit Interactive Web Application
    ├── utils.py                               # Plotly gauge, radar, and bar charts
    └── components/
        ├── live_detector.py                   # Step-by-step live checkpoint demo
        ├── benchmark_dashboard.py             # Model comparisons & reliability curves
        ├── ablation_view.py                   # Visual ablation study dashboard
        ├── review_queue.py                    # Step 8 Human-in-the-loop workstation
        └── dataset_browser.py                 # Evidence corpus & dataset explorer
```

---

## 🚀 How to Run

### Method 1: One-Click Windows Batch Launchers
- **To Launch the Interactive Web Application:**  
  Double-click `run_demo.bat`
- **To Re-generate Data and Retrain All Models:**  
  Double-click `run_pipeline.bat`

### Method 2: Manual Terminal Commands (PowerShell / Command Prompt)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Knowledge Corpus & Labeled 7-Feature Dataset:**
   ```bash
   python data/generate_dataset.py
   ```

3. **Train All 4 Classifiers & Run the Ablation Study:**
   ```bash
   python experiments/train_and_evaluate.py
   ```

4. **Execute Unit Tests:**
   ```bash
   python tests/test_retrieval.py
   python tests/test_features.py
   python tests/test_models.py
   ```

5. **Launch the Interactive Streamlit Web App:**
   ```bash
   streamlit run app/app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 📚 References

1. **Ji, Z., Lee, N., Frieske, R., et al. (2023).** Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12), Article 248.
2. **Lin, S., Hilton, J., & Evans, O. (2022).** TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*, 3214–3252.
3. **Manakul, P., Liusie, A., & Gales, M. (2023).** SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs. *EMNLP 2023*, 9004–9017.
4. **Li, J., Cheng, X., Zhao, X., et al. (2023).** HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. *EMNLP 2023*, 6449–6464.
5. **Min, S., Krishna, K., Lyu, X., et al. (2023).** FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *EMNLP 2023*, 12076–12100.
6. **Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024).** Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625–630.
7. **Sun, Y., Sheng, D., Zhou, Z., & Wu, Y. (2024).** AI hallucination: towards a comprehensive classification of distorted information in AI-generated content. *Humanities and Social Sciences Communications*, 11, 1278.
8. **Lewis, P., Perez, E., Piktus, A., et al. (2020).** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
9. **Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024).** RAGAs: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024*, 150–158.
10. **Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024).** ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL-HLT 2024*, 338–354.
11. **Lee, D., & Yu, H. (2025).** REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in LLMs. *SemEval-2025*, 7–15.
12. **Joshi, S. (2025).** Comprehensive Review of AI Hallucinations: Impacts and Mitigation Strategies for Financial and Business Applications. *IJCATR*, 14(06), 38–50.
13. **TCDS PAPER TASK 1 (2026).** AI Hallucination Risk Prediction Using Semantic Consistency, Evidence Support and Uncertainty Estimation. *24DSP501 Technical Communication in Data Science*.
14. **Course-provided reference dataset (2026).** Curated question–answer pairs with linked reference evidence, used for hallucination-risk feature extraction and labelling.
