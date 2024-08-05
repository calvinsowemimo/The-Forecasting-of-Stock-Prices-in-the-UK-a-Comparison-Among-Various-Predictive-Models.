import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the processed data
data = pd.read_csv('data/ftse100_data_processed.csv', index_col='Date', parse_dates=True)

# Scale the data (optional based on the warning)
data['Return'] = data['Return'] * 100

# Fit GARCH model
garch_model = arch_model(data['Return'], vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit(disp='off')

# Make predictions
forecast_horizon = 10  # Example forecast horizon, you can adjust this as needed
y_pred_garch = garch_model_fit.forecast(horizon=forecast_horizon)
y_pred_garch = y_pred_garch.variance.values[-forecast_horizon:].flatten()

# Align the prediction length with the actual data length
y_test = data['Return'][-forecast_horizon:]

# Evaluate the model
mse_garch = mean_squared_error(y_test, y_pred_garch)
rmse_garch = np.sqrt(mse_garch)
mae_garch = mean_absolute_error(y_test, y_pred_garch)
r2_garch = r2_score(y_test, y_pred_garch)

# Save the results
results_garch = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [mse_garch, rmse_garch, mae_garch, r2_garch]
})
results_garch.to_csv('results/garch_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_garch, label='Predicted')
plt.legend()
plt.title('GARCH Predictions')
plt.savefig('results/garch_plot.png')
plt.show()
