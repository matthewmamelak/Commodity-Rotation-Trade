# Seasonal Commodity Rotation Strategy

This project implements a **Seasonal Commodity Rotation Strategy** using Python and pandas. The strategy is designed to rotate among different commodity ETFs based on seasonal patterns, momentum, and trend filters to maximize returns and manage risk.

## Overview

The strategy analyzes historical price data for multiple commodity ETFs (provided in `seasonal_cmdt_rotation.xlsx`) and applies a set of rules to select the most promising ETF each month. The selection is based on:
- **Seasonality**: Average monthly returns for each ETF.
- **Momentum**: Positive 1-month return filter.
- **Trend**: 6-month and 200-day moving average filters.

Only the top ETF that passes all filters is selected each month, and the portfolio is fully allocated to it.

## How It Works

### 1. Data Loading and Preparation
- The script loads price data for each ETF from the Excel file (each sheet = 1 ETF).
- Prices are resampled to monthly and daily frequencies for analysis.

### 2. Indicator Calculation
- **Monthly Returns**: Calculates monthly percentage returns for each ETF.
- **Seasonal Matrix**: Computes the average return for each ETF in each calendar month (e.g., average January return).
- **6-Month Moving Average**: Used as a trend filter on monthly data.
- **200-Day Moving Average**: Calculated on daily data, then resampled to monthly for trend filtering.

### 3. Strategy Logic (Monthly Loop)
For each month (after the first year of data):
- **Seasonality**: Looks ahead to the next month's average return for each ETF.
- **Momentum Filter**: Only ETFs with a positive return in the previous month are eligible.
- **Trend Filters**: Only ETFs trading above both their 6-month and 200-day moving averages are eligible.
- **Selection**: Among eligible ETFs, selects the one with the highest expected seasonality for the next month.
- **Portfolio**: Fully allocates to the selected ETF for the month.

### 4. Performance Evaluation
- Calculates the cumulative return of the strategy.
- Computes key performance metrics:
  - **CAGR** (Compound Annual Growth Rate)
  - **Annualized Volatility**
  - **Sharpe Ratio**
  - **Maximum Drawdown**

### 5. Visualization
- Plots the cumulative return of the strategy over time.
- Prints performance metrics to the console.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib

## Usage
1. Place your ETF price data in `seasonal_cmdt_rotation.xlsx` (one sheet per ETF, with columns: Date, Price).
2. Run the script:
   ```bash
   python main.py
   ```
3. View the performance plot and metrics in the output.

## Notes
- The strategy logic can be easily modified to select more than one ETF or to use different filters.
- Make sure your Excel file is formatted correctly (dates and prices).

---

*This project is for educational purposes and does not constitute financial advice.* 