import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the processed data
data = pd.read_csv('data/ftse100_data_processed.csv', index_col='Date', parse_dates=True)

# Fit ARIMA model (p, d, q can be optimized)
arima_model = ARIMA(data['Return'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Make predictions
y_pred_arima = arima_model_fit.forecast(steps=len(data))

# Evaluate the model
mse_arima = mean_squared_error(data['Return'], y_pred_arima)
rmse_arima = np.sqrt(mse_arima)
mae_arima = mean_absolute_error(data['Return'], y_pred_arima)
r2_arima = r2_score(data['Return'], y_pred_arima)

# Save the results
results_arima = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [mse_arima, rmse_arima, mae_arima, r2_arima]
})
results_arima.to_csv('results/arima_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Return'], label='Actual')
plt.plot(data.index, y_pred_arima, label='Predicted')
plt.legend()
plt.title('ARIMA Predictions')
plt.savefig('results/arima_plot.png')
plt.show()
