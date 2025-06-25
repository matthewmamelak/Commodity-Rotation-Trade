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
  - For each window pair, it calculates the short-term and long-term moving averages for each ETF.

### 3. Automatic Parameter Calibration
- The model performs a grid search over all combinations of short and long moving average windows, the number of ETFs to select each month (TOP_N), and the drawdown threshold for risk control.
- For each combination, it runs the full backtest and computes the Sharpe ratio.
- The parameter set with the highest Sharpe ratio is selected as optimal.
- The strategy is then rerun using these optimal parameters for all further analysis and plots.

### 4. Monthly Portfolio Selection Logic
For each month (after the first year of data, to allow for moving averages):
- **Seasonality:** Looks ahead to the next month's average return for each ETF.
- **Momentum Filter:** Only ETFs with a positive return in the previous month are eligible.
- **Trend Filters:** Only ETFs trading above both their optimal short-term and long-term moving averages are eligible.
- **Selection:** Among eligible ETFs, selects the top N (optimized via grid search) with the highest expected seasonality for the next month, and allocates equally among them.
- **Drawdown Control:** If the strategy's drawdown exceeds a set threshold (optimized via grid search), the portfolio moves to cash (earning a 3% annualized rate) for the next month.
- **Cash Allocation:** If no ETF passes the filters, or if drawdown control is triggered, the portfolio is allocated to cash for that month.
- **Transaction Cost:** Each time the portfolio changes, a transaction cost (default: 0.1% per ETF traded) is subtracted from returns.

### 5. Performance Evaluation
- Calculates the cumulative return of the strategy over time.
- Computes key performance metrics:
  - **CAGR** (Compound Annual Growth Rate)
  - **Annualized Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**

## In-Sample and Out-of-Sample Testing
- **In-sample testing** (see `in_sample_test.py`): The strategy is calibrated and evaluated on the entire dataset, including grid search for optimal parameters and full performance/visualization.
- **Out-of-sample testing** (see `out_of_sample_test.py`): The dataset is split into a training set (for parameter calibration) and a test set (for evaluation). This script demonstrates how the strategy performs on unseen data, with separate plots and statistics for the out-of-sample period.

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
You can adjust the following parameters at the top of the scripts:
- `TOP_N_options`: List of number of ETFs to select each month (grid search)
- `DRAWDOWN_THRESHOLD_options`: List of drawdown levels to trigger cash allocation (grid search)
- `TRANSACTION_COST`: Cost per ETF traded
- `CASH_RATE`: Annualized cash return when in cash
- `short_ma_options` and `long_ma_options`: Ranges of moving average windows to grid search