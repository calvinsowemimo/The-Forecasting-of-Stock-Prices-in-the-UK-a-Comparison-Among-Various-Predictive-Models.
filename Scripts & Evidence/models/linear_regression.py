import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the processed data
data = pd.read_csv('data/ftse100_data_processed.csv', index_col='Date', parse_dates=True)

# Prepare features and target variable
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Return']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Save the results
results_lr = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [mse_lr, rmse_lr, mae_lr, r2_lr]
})
results_lr.to_csv('results/linear_regression_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_lr, label='Predicted')
plt.legend()
plt.title('Linear Regression Predictions')
plt.savefig('results/linear_regression_plot.png')
plt.show()
