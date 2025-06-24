# Seasonal Commodity Rotation Strategy

This project implements a **Seasonal Commodity Rotation Strategy** using Python and pandas. The strategy rotates among different commodity ETFs based on seasonal patterns, momentum, trend filters, and risk controls to maximize returns and manage risk.

## Overview

The strategy analyzes historical price data for multiple commodity ETFs (provided in `seasonal_cmdt_rotation.xlsx`) and applies a set of rules to select the most promising ETFs each month. The portfolio is dynamically allocated based on these rules, with built-in risk management and transaction cost simulation.

## How the Strategy Works

### 1. Data Loading and Preparation
- Loads price data for each ETF from the Excel file (each sheet = 1 ETF).
- Prices are resampled to monthly and daily frequencies for analysis.

### 2. Indicator Calculation
- **Monthly Returns:** Calculates monthly percentage returns for each ETF.
- **Seasonal Matrix:** Computes the average return for each ETF in each calendar month (e.g., average January return).
- **6-Month Moving Average:** Used as a trend filter on monthly data.
- **200-Day Moving Average:** Calculated on daily data, then resampled to monthly for trend filtering.

### 3. Strategy Logic (Monthly Loop)
For each month (after the first year of data):
- **Seasonality:** Looks ahead to the next month's average return for each ETF.
- **Momentum Filter:** Only ETFs with a positive return in the previous month are eligible.
- **Trend Filters:** Only ETFs trading above both their 6-month and 200-day moving averages are eligible.
- **Selection:** Among eligible ETFs, selects the top N (default: 3) with the highest expected seasonality for the next month, and allocates equally among them.
- **Drawdown Control:** If the strategy's drawdown exceeds a set threshold (default: -10%), the portfolio moves to cash (earning a 3% annualized rate) for the next month.
- **Cash Allocation:** If no ETF passes the filters, or if drawdown control is triggered, the portfolio is allocated to cash for that month.
- **Transaction Cost:** Each time the portfolio changes, a transaction cost (default: 0.1% per ETF traded) is subtracted from returns.

### 4. Performance Evaluation
- Calculates the cumulative return of the strategy.
- Computes key performance metrics:
  - **CAGR** (Compound Annual Growth Rate)
  - **Annualized Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**

## Parameters
You can adjust the following parameters at the top of `main.py`:
- `TOP_N`: Number of ETFs to select each month
- `DRAWDOWN_THRESHOLD`: Drawdown level to trigger cash allocation
- `TRANSACTION_COST`: Cost per ETF traded
- `CASH_RATE`: Annualized cash return when in cash

## Usage
1. Place your ETF price data in `seasonal_cmdt_rotation.xlsx` (one sheet per ETF, with columns: Date, Price).
2. Run the script:
   ```bash
   python main.py
   ```
3. View the performance plots and metrics in the output.

## Notes
- The strategy logic can be easily modified to select a different number of ETFs, use different filters, or adjust risk controls.
- Make sure your Excel file is formatted correctly (dates and prices).

---

*This project is for educational purposes and does not constitute financial advice.* 