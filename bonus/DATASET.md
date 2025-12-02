# Bitcoin Price Dataset Card

## Data Source
- **Source**: Yahoo Finance
- **URL**: https://finance.yahoo.com/quote/BTC-USD/
- **Collection Method**: Manual download from Yahoo Finance
- **License**: Educational/Personal Use (Yahoo Finance ToS)
- **Date Range**: 2014-09-17 to present
- **Frequency**: Daily OHLCV data

## Columns
- **Date**: Trading date
- **Open**: Opening price (USD)
- **High**: Highest price (USD)
- **Low**: Lowest price (USD)
- **Close**: Closing price (USD)
- **Adj Close**: Adjusted closing price (USD)
- **Volume**: Trading volume (BTC)

## Data Quality
- No missing values in selected period
- Total samples after preprocessing: 500
- Train/Test split: 80/20

## Preprocessing Steps
1. Calculated daily returns and excess returns
2. Computed technical indicators (MA, volatility, price ranges)
3. Created target variable: 5-day future excess returns
4. Removed rows with NaN values
5. Normalized features using MinMaxScaler
