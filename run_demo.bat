@echo off
title AI Hallucination Risk Prediction - Interactive Streamlit App
echo ======================================================================
echo Launching AI Hallucination Risk Prediction Interactive Application...
echo Course: 24DSP501 Technical Communication in Data Science
echo Authors: Hemavathy K (21662) & Madhumitha P (22286)
echo ======================================================================
cd /d "%~dp0"
python -m streamlit run app\app.py
pause
