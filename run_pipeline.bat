@echo off
title AI Hallucination Risk Prediction - Pipeline & Training
echo ======================================================================
echo Running Dataset Generation, Model Training, Benchmarking, and Ablation
echo Course: 24DSP501 Technical Communication in Data Science
echo ======================================================================
cd /d "%~dp0"
echo [1/3] Generating Reference Corpus and Labeled Dataset...
python data\generate_dataset.py
echo.
echo [2/3] Training Classifiers (XGBoost, RF, SVM, LR) and Running Ablation...
python experiments\train_and_evaluate.py
echo.
echo [3/3] Running Unit Test Suite...
python tests\test_retrieval.py
python tests\test_features.py
python tests\test_models.py
echo.
echo ======================================================================
echo Pipeline execution completed successfully!
echo ======================================================================
pause
