import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

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

# === STRATEGY PARAMETERS ===
TOP_N = 3  # Number of ETFs to select each month
DRAWDOWN_THRESHOLD = -0.10  # Max drawdown before going to cash
TRANSACTION_COST = 0.001  # 0.1% per ETF traded
CASH_RATE = 0.03  # 3% annualized

# === GRID SEARCH FOR MOVING AVERAGE WINDOWS ===
short_ma_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # months
long_ma_options = [100, 150, 200, 250, 300]  # days
results = {}
top_cumrets = {}

for short_ma, long_ma in itertools.product(short_ma_options, long_ma_options):
    # Recompute moving averages
    ma_short = combined_prices.rolling(window=short_ma).mean()
    daily_ma_long = combined_daily.rolling(window=long_ma).mean()
    monthly_ma_long = daily_ma_long.resample('M').last()

    portfolio_returns = []
    portfolio_weights = []
    dates = []
    prev_weights = pd.Series(0, index=monthly_returns.columns)
    dd_flag = False
    cumulative_return = 1.0
    cumulative_returns_list = []

    for i, date in enumerate(monthly_returns.index[12:]):
        current_month = date.month
        prev_month = date - pd.DateOffset(months=1)
        if prev_month not in monthly_returns.index:
            continue
        if len(cumulative_returns_list) > 0:
            running_max = max(cumulative_returns_list)
            drawdown = (cumulative_returns_list[-1] / running_max) - 1
            if dd_flag:
                ret = (1 + CASH_RATE) ** (1/12) - 1
                weights = pd.Series(0, index=monthly_returns.columns)
                dd_flag = False
                portfolio_returns.append(ret)
                portfolio_weights.append(weights)
                dates.append(date)
                cumulative_return *= (1 + ret)
                cumulative_returns_list.append(cumulative_return)
                prev_weights = weights.copy()
                continue
            elif drawdown <= DRAWDOWN_THRESHOLD:
                dd_flag = True
        next_month = (current_month % 12) + 1
        seasonal_scores = seasonal_matrix.iloc[next_month - 1]
        trailing_returns = monthly_returns.loc[prev_month]
        eligible = trailing_returns[trailing_returns > 0].index
        above_ma = combined_prices.columns[
            (combined_prices.loc[date] > monthly_ma_long.loc[date]) &
            (combined_prices.loc[date] > ma_short.loc[date])
        ]
        valid_etfs = eligible.intersection(above_ma)
        filtered_scores = seasonal_scores[valid_etfs]
        top_etfs = filtered_scores.sort_values(ascending=False).head(TOP_N).index
        weights = pd.Series(0, index=monthly_returns.columns)
        if len(top_etfs) > 0:
            weights[top_etfs] = 1.0 / len(top_etfs)
        if weights.sum() == 0:
            ret = (1 + CASH_RATE) ** (1/12) - 1
        else:
            ret = (weights * monthly_returns.loc[date]).sum()
        n_traded = (weights != prev_weights).sum()
        tc = n_traded * TRANSACTION_COST if n_traded > 0 else 0
        ret -= tc
        portfolio_returns.append(ret)
        portfolio_weights.append(weights)
        dates.append(date)
        cumulative_return *= (1 + ret)
        cumulative_returns_list.append(cumulative_return)
        prev_weights = weights.copy()
    if len(portfolio_returns) > 1:
        returns_series = pd.Series(portfolio_returns, index=dates, name='Strategy Return')
        volatility = returns_series.std() * np.sqrt(12)
        sharpe = (returns_series.mean() * 12) / volatility if volatility > 0 else 0
        results[(short_ma, long_ma)] = sharpe
        top_cumrets[(short_ma, long_ma)] = pd.Series(cumulative_returns_list, index=dates)

# Find best Sharpe
best_pair = max(results, key=results.get)
best_short, best_long = best_pair
print(f"\nOptimal MA windows: {best_short} months (short), {best_long} days (long) | Sharpe: {results[best_pair]:.3f}")

# === RERUN STRATEGY WITH OPTIMAL WINDOWS ===
ma6 = combined_prices.rolling(window=best_short).mean()
daily_ma200 = combined_daily.rolling(window=best_long).mean()
monthly_ma200 = daily_ma200.resample('M').last()

# === STEP 3: Strategy Logic (Enhanced) ===
portfolio_returns = []
portfolio_weights = []
dates = []
prev_weights = pd.Series(0, index=monthly_returns.columns)
dd_flag = False  # Drawdown control flag

def monthly_cash_return():
    return (1 + CASH_RATE) ** (1/12) - 1

cumulative_return = 1.0
cumulative_returns_list = []

for i, date in enumerate(monthly_returns.index[12:]):
    current_month = date.month
    prev_month = date - pd.DateOffset(months=1)
    if prev_month not in monthly_returns.index:
        continue

    # Drawdown control: if last month's drawdown exceeded threshold, go to cash
    if len(cumulative_returns_list) > 0:
        running_max = max(cumulative_returns_list)
        drawdown = (cumulative_returns_list[-1] / running_max) - 1
        if dd_flag:
            # Stay in cash for this month, reset flag
            ret = monthly_cash_return()
            weights = pd.Series(0, index=monthly_returns.columns)
            dd_flag = False
            portfolio_returns.append(ret)
            portfolio_weights.append(weights)
            dates.append(date)
            cumulative_return *= (1 + ret)
            cumulative_returns_list.append(cumulative_return)
            prev_weights = weights.copy()
            continue
        elif drawdown <= DRAWDOWN_THRESHOLD:
            # Trigger drawdown flag for next month
            dd_flag = True

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

    # Select top N ETFs
    top_etfs = filtered_scores.sort_values(ascending=False).head(TOP_N).index
    weights = pd.Series(0, index=monthly_returns.columns)
    if len(top_etfs) > 0:
        weights[top_etfs] = 1.0 / len(top_etfs)

    # Compute return
    if weights.sum() == 0:
        # No ETF selected, invest in cash
        ret = monthly_cash_return()
    else:
        ret = (weights * monthly_returns.loc[date]).sum()

    # Transaction cost: sum of absolute weight changes * cost
    n_traded = (weights != prev_weights).sum()
    tc = n_traded * TRANSACTION_COST if n_traded > 0 else 0
    ret -= tc

    portfolio_returns.append(ret)
    portfolio_weights.append(weights)
    dates.append(date)
    cumulative_return *= (1 + ret)
    cumulative_returns_list.append(cumulative_return)
    prev_weights = weights.copy()

# === STEP 4: Format and Evaluate ===
returns_series = pd.Series(portfolio_returns, index=dates, name='Strategy Return')
weights_df = pd.DataFrame(portfolio_weights, index=dates)
cumulative_return = pd.Series(cumulative_returns_list, index=dates)

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

# === Additional Visuals ===

# 1. ETF Selection Over Time
selected_etfs = [w.idxmax() if w.max() > 0 else 'None' for w in portfolio_weights]
plt.figure(figsize=(10, 2))
plt.plot(dates, selected_etfs, marker='o', linestyle='-', color='tab:blue')
plt.title('ETF Selected Each Month')
plt.xlabel('Date')
plt.ylabel('Selected ETF')
plt.yticks(rotation=45)
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.show()

# 2. Underwater (Drawdown) Plot
running_max = cumulative_return.cummax()
drawdown = (cumulative_return / running_max) - 1
plt.figure(figsize=(10, 3))
plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
plt.title('Strategy Drawdown (Underwater Plot)')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.tight_layout()
plt.show()

# 3. Comparison to Equal-Weighted Benchmark
benchmark_returns = monthly_returns.mean(axis=1)
benchmark_cum = (1 + benchmark_returns).cumprod().loc[cumulative_return.index]
plt.figure(figsize=(10, 5))
plt.plot(cumulative_return, label='Strategy')
plt.plot(benchmark_cum, label='Equal-Weighted Benchmark', linestyle='--')
plt.title('Strategy vs. Equal-Weighted Benchmark')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === PLOT HEATMAP OF SHARPE RATIOS ===
sharpe_matrix = np.zeros((len(short_ma_options), len(long_ma_options)))
for i, s in enumerate(short_ma_options):
    for j, l in enumerate(long_ma_options):
        sharpe_matrix[i, j] = results.get((s, l), np.nan)
plt.figure(figsize=(8, 6))
sns.heatmap(sharpe_matrix, annot=True, fmt=".2f", xticklabels=long_ma_options, yticklabels=short_ma_options, cmap="YlGnBu")
plt.title("Sharpe Ratio Heatmap (Short MA months vs Long MA days)")
plt.xlabel("Long MA (days)")
plt.ylabel("Short MA (months)")
plt.tight_layout()
plt.show()

# === PLOT CUMULATIVE RETURNS FOR TOP 3 PAIRS ===
top3 = sorted(results, key=results.get, reverse=True)[:3]
plt.figure(figsize=(10, 5))
for pair in top3:
    plt.plot(top_cumrets[pair], label=f"{pair[0]}m/{pair[1]}d (Sharpe={results[pair]:.2f})")
plt.plot(cumulative_return, label="Optimal (Selected)", linewidth=2, color="black")
plt.title("Cumulative Returns for Top 3 MA Pairs")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print stats
print("\nPerformance Metrics:")
print(f"CAGR: {cagr:.4f}")
print(f"Annualized Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")
