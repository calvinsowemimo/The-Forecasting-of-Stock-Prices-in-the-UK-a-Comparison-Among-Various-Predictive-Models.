import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

# Train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Evaluate the model
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse_svm = np.sqrt(mse_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

# Save the results
results_svm = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [mse_svm, rmse_svm, mae_svm, r2_svm]
})
results_svm.to_csv('results/svm_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_svm, label='Predicted')
plt.legend()
plt.title('SVM Predictions')
plt.savefig('results/svm_plot.png')
plt.show()
