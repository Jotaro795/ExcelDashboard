import scipy.optimize
import yfinance as yf
import datetime
import pandas as pd
import os
import numpy as np
import scipy
from datetime import datetime, timedelta

# Set date range (last 5 years)
end_date = datetime.today() 
start_date = end_date - timedelta(days=5*365)

# Function to fetch data for a single ticker
def fetch_data_for_ticker(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']]
    data['Ticker'] = ticker
    data['Daily Return'] = data['Close'].pct_change()
    return data

# List of tickers
tickers = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "SUNPHARMA.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
    "TATAELXSI.NS",
    "PERSISTENT.NS",
    "TRENT.NS",
    "POLYCAB.NS",
    "ASTRAL.NS",
    "ZOMATO.NS",
    "IRB.NS",
    "JKCEMENT.NS",
    "TTML.NS",
    "TATAPOWER.NS",
    "ADANIGREEN.NS"
]

# Define folder for CSV files and ensure it exists
folder_name = 'stock_prices'
os.makedirs(folder_name, exist_ok=True)

# Download and save data for each ticker
for ticker in tickers:
    ticker_data = fetch_data_for_ticker(ticker)
    file_name = os.path.join(folder_name, f'{ticker}_portfolio_prices.csv')
    ticker_data.to_csv(file_name, index=False)
    df = pd.read_csv(file_name)
    df.to_csv(file_name, index=False)
    print(f"File '{file_name}' has been updated with the latest data for {ticker}.")

# Remove the 2nd line from each CSV file (if needed)
folder_path = "stock_prices"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        updated_lines = [lines[0]] + lines[2:] if len(lines) > 1 else lines
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(updated_lines)
        print(f"Updated: {file_name} (Removed 2nd line)")

# Download combined closing prices data
data = yf.download(tickers, start_date, end_date)['Close']
returns = data.ffill().pct_change().dropna()
expected_returns = returns.mean()
covariance_matrix = returns.cov()

risk_free_rate = 0.073

# Portfolio performance function
def portfolio_performance(weights, expected_returns, covariance_matrix, risk_free_rate):
    portfolio_return = np.sum(weights * expected_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Optimize portfolio to maximize Sharpe ratio
def optimize_portfolio(expected_returns, covariance_matrix, risk_free_rate):
    num_assets = len(expected_returns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    result = scipy.optimize.minimize(
        lambda weights: -portfolio_performance(weights, expected_returns, covariance_matrix, risk_free_rate)[2],
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x

optimal_weights = optimize_portfolio(expected_returns, covariance_matrix, risk_free_rate)
optimal_weights = np.round(optimal_weights, 2)
optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize to ensure sum=1

portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(optimal_weights, expected_returns, covariance_matrix, risk_free_rate)
print(f"Portfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2}")

# --- NEW CODE: Calculate individual and portfolio metrics with additional info ---

# Function to fetch sector and market cap details using yfinance
def get_stock_info(ticker):
    info = yf.Ticker(ticker).info
    sector = info.get("sector", "Unknown")
    market_cap = info.get("marketCap", 0)  # in USD (or local currency unit)
    # Convert market cap to Crores if needed; here we assume the value is in a comparable unit
    # You may adjust thresholds based on your market; these are example thresholds:
    if market_cap > 5e11:       # e.g., > 500 billion
        size_category = "Large Cap"
    elif market_cap >= 1e11:    # e.g., between 100 billion and 500 billion
        size_category = "Mid Cap"
    else:
        size_category = "Small Cap"
    return sector, size_category

# Calculate covariance between each stock and the portfolio (annualized)
def calculate_cov_with_portfolio(ticker, covariance_matrix, optimal_weights):
    cov_row = covariance_matrix.loc[ticker]
    return np.dot(cov_row, optimal_weights) * 252  # Annualize

stock_metrics = []
for ticker in tickers:
    stock_returns = returns[ticker]
    mean_return = expected_returns[ticker] * 252
    volatility = stock_returns.std() * np.sqrt(252)
    sharpe_ratio_individual = (mean_return - risk_free_rate) / volatility
    covariance_with_portfolio = calculate_cov_with_portfolio(ticker, covariance_matrix, optimal_weights)
    sector, size_category = get_stock_info(ticker)
    
    stock_metrics.append({
        'Ticker': ticker,
        'Sector': sector,
        'Market Cap Category': size_category,
        'Mean Return': mean_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio_individual,
        'Weight': optimal_weights[tickers.index(ticker)],
        'Covariance with Portfolio': covariance_with_portfolio
    })

# Create DataFrame and save to CSV
metrics_df = pd.DataFrame(stock_metrics)
metrics_df.to_csv('portfolio_metrics.csv', index=False)

# Save the optimal weights to a separate CSV file
pd.DataFrame({'Ticker': tickers, 'Weight': optimal_weights}).to_csv('optimal_weights.csv', index=False)

print("\nNew file created: portfolio_metrics.csv")
print("Contains metrics for all stocks and the optimized portfolio, including Sector and Market Cap Category")

nifty_ticker = "^NSEI"  # Adjust if you want a different Nifty index
nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date)
nifty_data.reset_index(inplace=True)
nifty_data['Daily Return'] = nifty_data['Close'].pct_change()
nifty_data['Cumulative Return'] = (1 + nifty_data['Daily Return']).cumprod() - 1

# Save Nifty data to a CSV file
nifty_csv_file = os.path.join("nifty", "nifty_indices.csv")
nifty_data.to_csv(nifty_csv_file, index=False)
folder_path = "nifty"
for file_name in os.listdir(folder_path):
    if file_name.endswith("indices.csv"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        updated_lines = [lines[0]] + lines[2:] if len(lines) > 1 else lines
        with open(file_path, "w", encoding="utf-8") as file:
            file.writelines(updated_lines)
        print(f"Updated: {file_name} (Removed 2nd line)")
print(f"\nNifty indices data saved to '{nifty_csv_file}'")

cumulative_returns = (1 + returns).cumprod() - 1
returns['Cumulative Return'] = cumulative_returns.mean(axis=1)
returns.to_csv("portfolio_returns.csv", index=True)
volatility_df = pd.DataFrame({"Date": returns.index, "Portfolio Volatility": portfolio_volatility})
volatility_df.to_csv("portfolio_volatility.csv", index=False)

agg_metrics = {
    "Metric": ["Portfolio_Return", "Portfolio_Volatility", "Portfolio_Sharpe_Ratio"],
    "Value": [portfolio_return, portfolio_volatility, sharpe_ratio]
}
agg_metrics_df = pd.DataFrame(agg_metrics)
agg_metrics_df.to_csv("portfolio_agg_metrics.csv", index=False)

target_portfolio_value = 10_00_000

# Create a list to store holdings data
holdings_data = []

# Loop through each ticker to compute holdings based on optimal weights
for i, ticker in enumerate(tickers):
    try:
        # Fetch the latest price for the ticker (using the most recent closing price)
        stock = yf.Ticker(ticker)
        stock_history = stock.history(period="1d")
        if stock_history.empty:
            print(f"Data not available for {ticker}.")
            latest_price = np.nan
        else:
            latest_price = stock_history['Close'].iloc[-1]

        # Calculate dollar allocation for the ticker
        allocation = optimal_weights[i] * target_portfolio_value

        # Calculate the number of shares to hold (round as needed)
        if latest_price and latest_price > 0:
            shares_held = allocation / latest_price
        else:
            shares_held = 0

        holdings_data.append({
            "Ticker": ticker,
            "Latest Price": latest_price,
            "Rupee Allocation": allocation,
            "Shares Held": shares_held
        })
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Convert the holdings data into a DataFrame
holdings_df = pd.DataFrame(holdings_data)
holdings_df.to_csv('portfolio_holdings.csv', index=False)

# Calculate the correlation matrix among the stocks

# Calculate the correlation matrix among the stocks
stock_corr = returns[tickers].corr()

# Calculate portfolio returns as a weighted sum of daily returns
portfolio_returns = returns[tickers].dot(optimal_weights)

# Calculate the correlation of each stock with the portfolio
corr_with_portfolio = returns[tickers].corrwith(portfolio_returns)

# Append the portfolio correlations as a new column in the correlation matrix
stock_corr['Portfolio'] = corr_with_portfolio

# Create a new row for the portfolio correlations (Portfolio with itself is 1)
portfolio_corr_row = pd.concat([corr_with_portfolio, pd.Series({'Portfolio': 1.0})])
portfolio_corr_row.name = 'Portfolio'

# Append the new row to the correlation matrix
combined_corr = pd.concat([stock_corr, portfolio_corr_row], axis=0)


# Save the combined correlation matrix to a CSV file
combined_corr.to_csv('stock_correlation.csv')

print("\nNew file created: stock_correlation.csv")
print("Contains the correlation of each stock with every other stock and with the portfolio.")