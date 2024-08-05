import pandas as pd
import statsmodels.api as sm

# Load the processed data
data = pd.read_csv('data/ftse100_data_processed.csv', index_col='Date', parse_dates=True)

# Risk-free rate (example: 1% annualized risk-free rate)
risk_free_rate = 0.01 / 252  # Converted to daily rate

# Market returns (FTSE 100 index returns)
market_returns = data['Return']

# Excess returns
excess_returns = data['Return'] - risk_free_rate

# CAPM formula
X_capm = sm.add_constant(market_returns)
capm_model = sm.OLS(excess_returns, X_capm).fit()

# Save CAPM results
capm_results = pd.DataFrame({
    'Parameter': capm_model.params.index,
    'Value': capm_model.params.values,
    'P-Value': capm_model.pvalues.values
})
capm_results.to_csv('results/capm_results.csv', index=False)

print(capm_model.summary())
