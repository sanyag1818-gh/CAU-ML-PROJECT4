# 📈 CAU Machine Learning Project 4
## Hull Tactical Market Prediction

This repository contains the complete implementation for Project 4 of the 
Machine Learning course at Chung-Ang University. The project covers the 
full pipeline required to solve the Hull Tactical Market Prediction problem, 
including research, feature engineering, model building, evaluation, and 
Kaggle submission.

---

## 📁 Project Structure
CAU-ML-PROJECT4/
├── notebooks/
│   ├── NotebookA_Research.ipynb        # Full research workflow: EDA, FE, training, backtesting
│   └── NotebookB_Inference.ipynb       # Kaggle inference server notebook
│
├── report/
│   └── ML_Project4_Report.pdf          # Final project report
│
├── submissions/
│   └── submission.csv                  # Kaggle-ready submission file
│
├── models/
│   └── final_lgbm_model.pkl            # Saved LightGBM model + feature list
│
└── README.md                           # Project documentation


## 🔍 1. Project Overview

The objectives of this project are:

- Predict market-forward excess returns for the S&P 500.
- Convert predictions into valid portfolio allocation weights (0–2).
- Ensure strategy volatility ≤ **1.2 ×** benchmark volatility.
- Submit final predictions to Kaggle’s evaluation server.

## 📊 2. Exploratory Data Analysis (Notebook A)

- Examination of feature groups: **D, E, I, M, P, S, V**
- Detection of missing values and data gaps
- Visualizing return distributions and long-term market structure
- Checking target variable behavior over decades

## 🔧 3. Baseline Models

### Momentum Baseline  
- **MSE:** 0.000136  
- **MAE:** 0.008368  
- **Direction Accuracy:** 48.14%

### Random Forest Baseline (Lag Features 1–5)
- **MSE:** 0.000126  
- **RMSE:** 0.011207  
- **MAE:** 0.007943  
- **Direction Accuracy:** 48.06%

## 🧠 4. Feature Engineering

- 1–5 day lag returns
- Rolling mean (5-day and 21-day)
- Rolling volatility window
- Drawdown computation per timestamp
- High-volatility regime indicator
- Sharpe-like metric calculation

## 🌲 5. Improved LightGBM Model

### Time-Series Cross Validation (5 folds)

| Fold | MSE       | RMSE      | MAE      |
|------|-----------|-----------|----------|
| 1    | 0.000157  | 0.012539  | 0.009654 |
| 2    | 0.000118  | 0.010873  | 0.008075 |
| 3    | 0.000208  | 0.014435  | 0.010636 |
| 4    | 0.000085  | 0.009206  | 0.006910 |
| 5    | 0.000163  | 0.012770  | 0.009117 |

**Average CV RMSE ≈ 0.0120**

(Used as the main model evaluation metric.)

---

## 📉 6. Validation Backtesting Results (Notebook A)

### Volatility
- **Strategy volatility:** 0.013341  
- **Benchmark volatility:** 0.011117  
- **Volatility ratio:** **1.200** (exactly meets the constraint)

### Sharpe-like Metric
- **Strategy Sharpe-like:** 0.0121  
- **Benchmark Sharpe-like:** 0.0186  

### Drawdowns
- **Max strategy drawdown:** −0.3856  
- **Max benchmark drawdown:** −0.3258  

These results match realistic behavior in noisy financial time series 
and confirm correct implementation of the required constraints.

##  7. Final Model Configuration
learning_rate: 0.05
n_estimators: 900
num_leaves: 100
max_depth: 10
objective: regression
metric: rmse
n_jobs: -1
random_state: 42

## 📐 8. Portfolio Weight Mapping

All predictions are converted into portfolio allocation weights using:

1. tanh-based normalization  
2. shifting to positive range  
3. clipping to **[0, 2]**

This satisfies Kaggle’s submission rules.

---

## ⚖️ 9. Volatility Constraint

After weight generation, strategy returns are rescaled when:


In validation, the final ratio was:



---

## 📤 10. Kaggle Submission

Notebook A and B generate:

- `submission.csv`  
- Saved LightGBM model  
- Feature lists  
- Inference server for Kaggle API testing  

All outputs are Kaggle-compliant.

---

## 🏁 11. Final Results Summary

### Validation (Notebook A)
- **Final validation RMSE:** 0.013119  
- **Final validation MAE:** 0.009543  
- **Direction Accuracy:** 49.09%  

### Backtest Metrics
- **Volatility ratio:** 1.200  
- **Strategy Sharpe-like:** 0.0121  
- **Max Drawdown:** −0.3856  

### Kaggle Output
- Successfully generated **submission.csv**  
- Fully compliant inference pipeline  





This project fulfills all requirements of CAU ML Project 4:
- Baseline → Feature Engineering → Model → Backtest → Submission  
- Time-series proper split  
- Volatility constraint implemented  
- Kaggle submission validated  





