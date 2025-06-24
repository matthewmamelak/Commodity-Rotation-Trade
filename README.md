# Seasonal Commodity Rotation Strategy

This project implements a **Seasonal Commodity Rotation Strategy** using Python. The strategy rotates among different commodity ETFs based on seasonal patterns, momentum, trend filters, and risk controls.

## Overview

The strategy analyzes historical price data for multiple commodity ETFs (provided in `seasonal_cmdt_rotation.xlsx`) and applies a set of rules to select the most promising ETFs each month. The portfolio is dynamically allocated based on these rules, with built-in risk management, transaction cost simulation, and automatic parameter calibration.

## How the Strategy Works

### 1. Data Loading and Preparation
- Loads price data for each ETF from the Excel file (each sheet = 1 ETF).
- Prices are resampled to monthly and daily frequencies for analysis.

### 2. Indicator Calculation
- **Monthly Returns:** Calculates monthly percentage returns for each ETF.
- **Seasonal Matrix:** Computes the average return for each ETF in each calendar month (e.g., average January return).
- **Moving Averages:**
  - The strategy tests a grid of short-term (3–12 months) and long-term (100–300 days) moving average windows for trend filtering.
  - For each window pair, it calculates the 6-month (or other short-term) and 200-day (or other long-term) moving averages for each ETF.

### 3. Automatic Parameter Calibration
- The model performs a grid search over all combinations of short and long moving average windows.
- For each combination, it runs the full backtest and computes the Sharpe ratio.
- The window pair with the highest Sharpe ratio is selected as optimal.
- The strategy is then rerun using these optimal moving average parameters for all further analysis and plots.

### 4. Monthly Portfolio Selection Logic
For each month (after the first year of data, to allow for moving averages):
- **Seasonality:** Looks ahead to the next month's average return for each ETF.
- **Momentum Filter:** Only ETFs with a positive return in the previous month are eligible.
- **Trend Filters:** Only ETFs trading above both their optimal short-term and long-term moving averages are eligible.
- **Selection:** Among eligible ETFs, selects the top N (default: 3) with the highest expected seasonality for the next month, and allocates equally among them.
- **Drawdown Control:** If the strategy's drawdown exceeds a set threshold (default: -10%), the portfolio moves to cash (earning a 3% annualized rate) for the next month.
- **Cash Allocation:** If no ETF passes the filters, or if drawdown control is triggered, the portfolio is allocated to cash for that month.
- **Transaction Cost:** Each time the portfolio changes, a transaction cost (default: 0.1% per ETF traded) is subtracted from returns.

### 5. Performance Evaluation
- Calculates the cumulative return of the strategy over time.
- Computes key performance metrics:
  - **CAGR** (Compound Annual Growth Rate)
  - **Annualized Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**

### 6. Visualization
- Plots the cumulative return of the strategy over time using the optimal moving average calibration.
- Shows which ETFs are selected each month.
- Plots the drawdown (underwater plot) of the strategy.
- Compares the strategy's cumulative return to an equal-weighted benchmark of all ETFs.
- Displays a heatmap of Sharpe ratios for all tested moving average window pairs.
- Plots cumulative returns for the top 3 moving average pairs, alongside the optimal one, for comparison.
- Prints performance metrics to the console.

## Buying, Selling, and Money Simulation

This strategy simulates a real investment process:

- **Buying and Selling:**
  - Each month, the strategy "buys" the top N ETFs that pass all filters, allocating the portfolio equally among them.
  - If an ETF held last month is no longer in the top N, the strategy "sells" it and reallocates to the new selection.
  - If no ETFs pass the filters, or if drawdown control is triggered, the strategy "sells" all ETFs and moves the portfolio to cash.
  - Transaction costs are subtracted for each ETF bought or sold, mimicking real-world trading costs.

- **Money Simulation:**
  - The strategy starts with a hypothetical portfolio (e.g., $1 or 100%).
  - All returns, allocations, and trades are tracked as if real money were being invested.
  - The cumulative return shows how the portfolio would have grown over time, including the effects of buying, selling, and transaction costs.
  - When the strategy is "in cash," it earns a fixed interest rate (e.g., 3% annualized).


## Parameters
You can adjust the following parameters at the top of `main.py`:
- `TOP_N`: Number of ETFs to select each month
- `DRAWDOWN_THRESHOLD`: Drawdown level to trigger cash allocation
- `TRANSACTION_COST`: Cost per ETF traded
- `CASH_RATE`: Annualized cash return when in cash
- `short_ma_options` and `long_ma_options`: Ranges of moving average windows to grid search



---
