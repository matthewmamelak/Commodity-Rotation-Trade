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
    monthly_prices[sheet] = df['Price'].resample('M').last().to_frame(name=sheet)
    daily_prices[sheet] = df['Price'].to_frame(name=sheet)

combined_prices = pd.concat(monthly_prices.values(), axis=1).dropna(how='all')
combined_daily = pd.concat(daily_prices.values(), axis=1).dropna(how='all')

# === STEP 2: Split Data into Training and Testing ===
split_ratio = 0.7  # 70% train, 30% test
split_idx = int(len(combined_prices) * split_ratio)
split_date = combined_prices.index[split_idx]

train_prices = combined_prices.loc[:split_date]
test_prices = combined_prices.loc[split_date:]
train_daily = combined_daily.loc[:split_date]
test_daily = combined_daily.loc[split_date:]

# Helper function to run grid search and strategy
def run_strategy(prices, daily, grid_search=True, best_params=None, results_dict=None):
    monthly_returns = prices.pct_change().dropna()
    seasonal_matrix = monthly_returns.groupby(monthly_returns.index.month).mean()
    seasonal_matrix.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Parameters
    TOP_N_options = [1, 2, 3, 4, 5]
    DRAWDOWN_THRESHOLD_options = [-0.05, -0.10, -0.15, -0.20]
    TRANSACTION_COST = 0.001
    CASH_RATE = 0.03
    short_ma_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    long_ma_options = [100, 150, 200, 250, 300]
    results = {} if results_dict is None else results_dict
    top_cumrets = {}

    if grid_search:
        for short_ma, long_ma, TOP_N, DRAWDOWN_THRESHOLD in itertools.product(short_ma_options, long_ma_options, TOP_N_options, DRAWDOWN_THRESHOLD_options):
            ma_short = prices.rolling(window=short_ma).mean()
            daily_ma_long = daily.rolling(window=long_ma).mean()
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
                above_ma = prices.columns[
                    (prices.loc[date] > monthly_ma_long.loc[date]) &
                    (prices.loc[date] > ma_short.loc[date])
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
                results[(short_ma, long_ma, TOP_N, DRAWDOWN_THRESHOLD)] = sharpe
                top_cumrets[(short_ma, long_ma, TOP_N, DRAWDOWN_THRESHOLD)] = pd.Series(cumulative_returns_list, index=dates)
        best_params = max(results, key=results.get)
        return best_params, results, top_cumrets
    else:
        # Use best_params to run strategy on test set
        short_ma, long_ma, TOP_N, DRAWDOWN_THRESHOLD = best_params
        ma_short = prices.rolling(window=short_ma).mean()
        daily_ma_long = daily.rolling(window=long_ma).mean()
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
                    ret = (1 + 0.03) ** (1/12) - 1
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
            above_ma = prices.columns[
                (prices.loc[date] > monthly_ma_long.loc[date]) &
                (prices.loc[date] > ma_short.loc[date])
            ]
            valid_etfs = eligible.intersection(above_ma)
            filtered_scores = seasonal_scores[valid_etfs]
            top_etfs = filtered_scores.sort_values(ascending=False).head(TOP_N).index
            weights = pd.Series(0, index=monthly_returns.columns)
            if len(top_etfs) > 0:
                weights[top_etfs] = 1.0 / len(top_etfs)
            if weights.sum() == 0:
                ret = (1 + 0.03) ** (1/12) - 1
            else:
                ret = (weights * monthly_returns.loc[date]).sum()
            n_traded = (weights != prev_weights).sum()
            tc = n_traded * 0.001 if n_traded > 0 else 0
            ret -= tc
            portfolio_returns.append(ret)
            portfolio_weights.append(weights)
            dates.append(date)
            cumulative_return *= (1 + ret)
            cumulative_returns_list.append(cumulative_return)
            prev_weights = weights.copy()
        returns_series = pd.Series(portfolio_returns, index=dates, name='Strategy Return')
        weights_df = pd.DataFrame(portfolio_weights, index=dates)
        cumulative_return = pd.Series(cumulative_returns_list, index=dates)
        cagr = cumulative_return.iloc[-1]**(12 / len(cumulative_return)) - 1
        volatility = returns_series.std() * np.sqrt(12)
        sharpe = (returns_series.mean() * 12) / volatility
        max_dd = (cumulative_return / cumulative_return.cummax() - 1).min()
        return returns_series, weights_df, cumulative_return, cagr, volatility, sharpe, max_dd

# === STEP 3: In-Sample (Training) Grid Search ===
print("\n=== In-Sample (Training) Grid Search ===")
best_params, train_results, train_cumrets = run_strategy(train_prices, train_daily, grid_search=True)
print(f"Best Params (train): {best_params}")

# === STEP 4: Out-of-Sample (Testing) Performance ===
print("\n=== Out-of-Sample (Testing) Performance ===")
test_returns, test_weights, test_cum, test_cagr, test_vol, test_sharpe, test_maxdd = run_strategy(test_prices, test_daily, grid_search=False, best_params=best_params)
print(f"Test CAGR: {test_cagr:.4f}")
print(f"Test Volatility: {test_vol:.4f}")
print(f"Test Sharpe: {test_sharpe:.4f}")
print(f"Test Max Drawdown: {test_maxdd:.4f}")

# === STEP 5: Plot Out-of-Sample Results ===
plt.figure(figsize=(10, 5))
plt.plot(test_cum, label='Out-of-Sample Cumulative Return')
plt.title('Out-of-Sample Cumulative Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('oos_cumulative_return.png')
plt.show()

plt.figure(figsize=(8, 4))
test_returns.hist(bins=30, alpha=0.7)
plt.title('Histogram of Monthly Returns (Out-of-Sample)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('oos_histogram_returns.png')
plt.show()

# Out-of-sample drawdown plot
running_max = test_cum.cummax()
drawdown = (test_cum / running_max) - 1
plt.figure(figsize=(10, 3))
plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.4)
plt.title('Out-of-Sample Drawdown (Underwater Plot)')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.tight_layout()
plt.savefig('oos_drawdown.png')
plt.show()

# Out-of-sample ETF selection frequency
selection_counts = (test_weights > 0).sum()
plt.figure(figsize=(8, 4))
selection_counts.sort_values().plot(kind='bar')
plt.title('Out-of-Sample ETF Selection Frequency')
plt.ylabel('Number of Months Selected')
plt.xlabel('ETF')
plt.tight_layout()
plt.savefig('oos_etf_selection_frequency.png')
plt.show() 