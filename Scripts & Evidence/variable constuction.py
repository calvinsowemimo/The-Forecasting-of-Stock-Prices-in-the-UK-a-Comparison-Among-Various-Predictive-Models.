import pandas as pd

# Load the data
data = pd.read_csv('data/ftse100_data.csv', index_col='Date', parse_dates=True)

# Calculate daily returns
data['Return'] = data['Adj Close'].pct_change()

# Drop NaN values
data = data.dropna()

# Save the processed data to a CSV file
data.to_csv('data/ftse100_data_processed.csv')

# Display descriptive statistics
descriptive_stats = data.describe()
print(descriptive_stats)

# Save descriptive statistics to a CSV file
descriptive_stats.to_csv('data/descriptive_statistics.csv')
