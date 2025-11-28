# CAU-ML-PROJECT4
Hull Tactical Market Prediction - CAU Machine Learning Project 4

This repository contains the full implementation, analysis, and reproducible code for Project 4 — Hull Tactical Market Prediction, completed as part of the Machine Learning course at Chung-Ang University.

The objectives of the project are:

Predict S&P 500 market-forward excess returns.

Convert predictions into valid portfolio allocation weights between 0 and 2.

Maintain model volatility at or below 1.2 times the benchmark volatility.

Submit predictions through Kaggle’s competition server.

2. Repository Structure
notebooks/
    NotebookA_Research.ipynb
    NotebookB_Inference.ipynb

report/
    ML_Project4_Report.pdf

submissions/
    submission.csv

models/
    final_lgbm_model.pkl

README.md

3. Project Overview

This project implements the complete pipeline required for the Hull Tactical Market Prediction task.

3.1 Exploratory Data Analysis

Examination of dataset structure and feature groups (D, E, I, M, P, S, V).

Detection and visualization of missing values.

Analysis of the target variable distribution and trends over time.

3.2 Baseline Models

Momentum-based baseline using 5-day rolling averages.

Lag-feature Random Forest baseline using 1–5 day lags.

Evaluation through MSE, RMSE, MAE, and directional accuracy.

3.3 Feature Engineering

Creation of lag features (1–5 days).

Rolling mean and rolling volatility features.

Drawdown computation.

Identification of high-volatility regimes.

Additional diagnostics, including Sharpe-like statistics.

3.4 Improved Model

Full-feature LightGBM model with exclusion of non-predictive columns.

Five-fold TimeSeriesSplit cross-validation.

Cross-validated RMSE and directional accuracy evaluation.

Feature importance analysis.

3.5 Portfolio Weight Mapping

All model predictions are transformed into portfolio weights using:

tanh-based transformation → shifted → clipped to [0, 2]

3.6 Volatility Constraint

Strategy returns are rescaled when:

model_volatility > 1.2 × benchmark_volatility

3.7 Final Kaggle Submission

LightGBM retrained on the full dataset.

Generation of the final submission file (submission.csv).

Validation using Kaggle’s evaluation server.

4. Reproducibility Instructions
4.1 Requirements

This project uses the standard Kaggle Python environment, including:

numpy
pandas
polars
joblib
lightgbm
scikit-learn
matplotlib
seaborn

No additional installations are required when running inside Kaggle.

4.2 Running Notebook A

Open the file:

notebooks/NotebookA_Research.ipynb

Run all cells in order to reproduce:

Exploratory Data Analysis

Feature Engineering

Baseline Models

LightGBM Training

Backtesting

Final submission file generation

4.3 Running Notebook B (Inference Notebook)

Notebook B contains the final model used during Kaggle’s evaluation stage.

File:

notebooks/NotebookB_Inference.ipynb

Notebook B:

Trains the LightGBM model.

Saves the model and feature list.

Defines the prediction function used by the evaluation server.

Starts Kaggle’s default inference server.

To run a local test of the server inside Kaggle:

server.run_local_gateway((KAGGLE_INPUT_PATH,))

5. Results

(The following values are placeholders. Replace them with your actual results.)

Kaggle Score: XX.XXXXX
Validation RMSE: XXXX
Directional Accuracy: XX%
