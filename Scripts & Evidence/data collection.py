import yfinance as yf
import pandas as pd

# Define the list of tickers for FTSE 100 companies
ftse100_tickers = ["^FTSE"]  # Placeholder for the FTSE 100 index ticker

# Download historical stock data for the past 10 years
data = yf.download(ftse100_tickers, start="2014-01-01", end="2024-01-01", interval="1d")

# Save the data to a CSV file
data.to_csv('data/ftse100_data.csv')
