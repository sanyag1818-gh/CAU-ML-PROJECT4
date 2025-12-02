# Bitcoin Portfolio Strategy - Bonus Project

## Overview
This project implements a portfolio allocation strategy for Bitcoin (BTC) based on LSTM-predicted excess returns.

## Project Structure
- btc_portfolio_strategy.ipynb: Main notebook
- btc.csv: Raw data
- DATASET.md: Dataset documentation
- README.md: This file

## Requirements
- pandas, numpy, matplotlib
- scikit-learn
- tensorflow>=2.10.0

## How to Run
1. Ensure btc.csv is in the same directory
2. Open the main notebook
3. Run all cells sequentially

## Key Features
- **Target**: Predicts 5-day excess returns
- **Features**: 12 technical indicators
- **Model**: Multi-layer LSTM with dropout
- **Strategy**: Dynamic weight allocation (0-2)
- **Constraint**: Portfolio volatility ≤ 1.2× benchmark

## Results Summary
- Sharpe Ratio: 1.437
- Cumulative Return: 5.62%
- Max Drawdown: -5.26%
- Volatility Ratio: 1.2×
