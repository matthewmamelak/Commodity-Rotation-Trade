import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === STEP 1: Load and Prepare Data ===
file_path = "seasonal_cmdt_rotation.xlsx"  # Make sure this Excel file is in the same folder

# Load Excel file (each sheet = 1 ETF)
xls = pd.ExcelFile(file_path)
monthly_prices = {}
daily_prices = {}

for sheet in xls.sheet_names:
    df = xls.parse(sheet).dropna(how='all', axis=1)
    df.columns = ['Date', 'Price'] + df.columns[2:].tolist()
    df = df[['Date', 'Price']].dropna()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price']).sort_values('Date')
    df = df.set_index('Date')

    # Store monthly and daily separately
    monthly_prices[sheet] = df['Price'].resample('M').last().to_frame(name=sheet)
    daily_prices[sheet] = df['Price'].to_frame(name=sheet)

# Combine all ETFs into a single monthly and daily DataFrame
combined_prices = pd.concat(monthly_prices.values(), axis=1).dropna(how='all')
combined_daily = pd.concat(daily_prices.values(), axis=1).dropna(how='all')

# === STEP 2: Compute Indicators ===
monthly_returns = combined_prices.pct_change().dropna()
seasonal_matrix = monthly_returns.groupby(monthly_returns.index.month).mean()
seasonal_matrix.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 6-month trend filter (monthly)
ma6 = combined_prices.rolling(window=6).mean()

# 200-day moving average filter (daily, resampled monthly)
daily_ma200 = combined_daily.rolling(window=200).mean()
monthly_ma200 = daily_ma200.resample('M').last()

# === STEP 3: Strategy Logic ===
portfolio_returns = []
portfolio_weights = []
dates = []

for date in monthly_returns.index[12:]:
    current_month = date.month
    prev_month = date - pd.DateOffset(months=1)
    if prev_month not in monthly_returns.index:
        continue

    # Look ahead to next month's seasonality
    next_month = (current_month % 12) + 1
    seasonal_scores = seasonal_matrix.iloc[next_month - 1]

    # 1M momentum filter
    trailing_returns = monthly_returns.loc[prev_month]
    eligible = trailing_returns[trailing_returns > 0].index

    # 200-day MA & 6M trend filter
    above_ma = combined_prices.columns[
        (combined_prices.loc[date] > monthly_ma200.loc[date]) &
        (combined_prices.loc[date] > ma6.loc[date])
    ]

    # Combine all filters
    valid_etfs = eligible.intersection(above_ma)
    filtered_scores = seasonal_scores[valid_etfs]

    # Select top 1 ETF
    top_etfs = filtered_scores.sort_values(ascending=False).head(1).index
    weights = pd.Series(0, index=monthly_returns.columns)
    if len(top_etfs) > 0:
        weights[top_etfs] = 1.0

    # Compute return
    ret = (weights * monthly_returns.loc[date]).sum()
    portfolio_returns.append(ret)
    portfolio_weights.append(weights)
    dates.append(date)

# === STEP 4: Format and Evaluate ===
returns_series = pd.Series(portfolio_returns, index=dates, name='Strategy Return')
weights_df = pd.DataFrame(portfolio_weights, index=dates)
cumulative_return = (1 + returns_series).cumprod()

# Performance metrics
cagr = cumulative_return.iloc[-1]**(12 / len(cumulative_return)) - 1
volatility = returns_series.std() * np.sqrt(12)
sharpe = (returns_series.mean() * 12) / volatility
max_dd = (cumulative_return / cumulative_return.cummax() - 1).min()

# === STEP 5: Plot ===
plt.figure(figsize=(10, 5))
plt.plot(cumulative_return, label='Strategy Cumulative Return')
plt.title('Seasonal Commodity Rotation Strategy (Enhanced)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print stats
print("\nPerformance Metrics:")
print(f"CAGR: {cagr:.4f}")
print(f"Annualized Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
