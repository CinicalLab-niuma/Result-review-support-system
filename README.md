# Result-review-support-system
AI-Driven Interception of Lab Artifacts
# Pheno-Calcium-RRSS: Phenotype-Driven Machine Learning for Pseudohypercalcemia

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-success.svg)](https://shap.readthedocs.io/)

This repository contains the core source code and analytical pipeline for the research project: **"A Phenotype-Driven Machine Learning Framework for Intercepting Pseudohypercalcemia: Towards an Explainable Results Review Support System"**. 

It aims to provide an innovative, biochemical-test-free solution for early warning and error interception, paving the way for the "Unmanned Clinical Laboratory" and "Intelligent Quality Control" (IQC).

## 🌟 Key Highlights
- **Identification of Occult Preanalytical Errors**: Systematically elucidated pseudohypercalcemia triggered by complex physicochemical interactions between specific blood collection tubes (Company A) and a particular biochemical analyzer (Hitachi 7600 A). Such artifacts perfectly evade routine internal quality control (IQC) and external quality assessment (EQA).
- **Purely Phenotype-Driven**: Accurately intercepts false-positive laboratory reports utilizing only 11 core clinical phenotypes mined from Electronic Medical Records (EMRs), eliminating the need for additional expensive biochemical confirmatory tests.
- **Optimal Predictive Model**: Constructed and evaluated six mainstream machine learning algorithms (XGBoost, LightGBM, CatBoost, Random Forest, SVM, and Logistic Regression). The **XGBoost** framework demonstrated superior predictive performance via rigorous stratified 5-fold cross-validation.
- **Explainable AI (XAI)**: Integrated the game theory-based **SHAP** algorithm to precisely quantify feature contributions. Global explanations flawlessly align with the real-world epidemiology of emergency departments (e.g., the high-frequency impact of malignancy and weakness), while local waterfall plots acutely safeguard individual clinical diagnostic principles (e.g., identifying primary hyperparathyroidism).
- **Clinical Utility**: Validated substantial net clinical benefit in practical medical scenarios through Decision Curve Analysis (**DCA**).

## 📂 Repository Structure

```text
├── data/
│   ├── dummy_data.csv          # Dummy dataset for code reproduction (de-identified/randomized)
│   └── README_data.md          # Data dictionary and variable descriptions
├── src/
│   ├── data_preprocessing.py   # Data cleaning, zero-imputation, and class imbalance handling
│   ├── model_training_cv.py    # Training and stratified 5-fold cross-validation of 6 models
│   ├── shap_analysis.py        # Generation of SHAP global and local interpretation plots
│   └── dca_evaluation.py       # Decision Curve Analysis (DCA) evaluation code
├── results/
│   ├── figures/                # Output directory for ROC curves, SHAP plots, and DCA curves
│   └── model_performance.csv   # Summary metrics (AUC, Acc, Sens, Spec) for all models
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
