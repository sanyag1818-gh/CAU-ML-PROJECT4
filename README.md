# CAU Machine Learning Project 4  
## Hull Tactical Market Prediction

This repository contains the full implementation for Project 4 of the Machine Learning course at Chung-Ang University. It includes the complete pipeline for solving the Hull Tactical Market Prediction problem—covering research, feature engineering, model development, backtesting, and Kaggle submission.

---

#  Project Structure

```
CAU-ML-PROJECT4/
├── notebooks/
│   ├── NotebookA_Research.ipynb        # EDA, FE, model training, backtesting
│   └── NotebookB_Inference.ipynb       # Kaggle submission notebook
│
├── report/
│   └── ML_Project4_Report.pdf
│
├── submissions/
│   └── submission.csv                  # Kaggle-ready submission
│
│
└── README.md
```

---

# 1. Project Overview

- Predict S&P 500 market-forward excess returns  
- Convert predictions → portfolio weights in [0, 2]  
- Satisfy volatility constraint: strategy_vol ≤ 1.2 × benchmark_vol  
- Build a clean, reproducible ML pipeline for Kaggle

---

# 2. Exploratory Data Analysis (Notebook A)

- Feature-group inspection: D, E, I, M, P, S, V  
- Missing value structure  
- Target distribution  
- Long-term return behavior  
- Regime transitions  

---

# 3. Baseline Models

| Model | MSE | RMSE | MAE | Direction Accuracy |
|-------|--------|--------|--------|---------------------|
| Momentum (5-day) | 0.000136 | 0.01166 | 0.008368 | 48.14% |
| Random Forest | 0.000126 | 0.011207 | 0.007943 | 48.06% |

---

# 4. Feature Engineering

- Lag returns (1–5 days)  
- Rolling means (5, 21)  
- Volatility  
- Drawdown magnitude  
- High-volatility indicator  
- Sharpe-like metric  

---

# 5. LightGBM Model

Time-Series CV Results:

| Fold | MSE | RMSE | MAE |
|------|-----------|-----------|-----------|
| 1 | 0.000157 | 0.012539 | 0.009654 |
| 2 | 0.000118 | 0.010873 | 0.008075 |
| 3 | 0.000208 | 0.014435 | 0.010636 |
| 4 | 0.000085 | 0.009206 | 0.006910 |
| 5 | 0.000163 | 0.012770 | 0.009117 |

Avg CV RMSE ≈ 0.0120

---

# 6. Backtesting Results

**Volatility**  
- Strategy: 0.013341  
- Benchmark: 0.011117  
- Ratio: 1.200  

**Drawdowns**  
- Strategy: −0.3856  
- Benchmark: −0.3258  

---

# 7. Final LightGBM Configuration

```
learning_rate: 0.05
n_estimators: 900
num_leaves: 100
max_depth: 10
objective: regression
metric: rmse
n_jobs: -1
random_state: 42
```

---

# 8. Portfolio Weight Mapping

- tanh normalization  
- shift to positive  
- clip to [0, 2]  

---

# 9. Volatility Constraint

strategy_vol / benchmark_vol ≤ 1.2  
Final ratio achieved: 1.200

---

# 10. Kaggle Submission

Generates:  
- submission.csv  

---

# 11. Final Results Summary

| Metric | Value |
|--------|-------|
| RMSE | 0.013119 |
| MAE | 0.009543 |
| Direction Accuracy | 49.09% |
| Volatility Ratio | 1.200 |
| Sharpe-like | 0.0121 |
| Max Drawdown | −0.3856 |

---

# Flowchart

```mermaid
flowchart TD
    A[Load Data] --> B[EDA]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Backtesting]
    E --> F[Weight Mapping]
    F --> G[Kaggle Submission]
```
