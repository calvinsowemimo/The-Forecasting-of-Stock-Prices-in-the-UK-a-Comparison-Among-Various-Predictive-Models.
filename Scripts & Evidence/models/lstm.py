import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the processed data
data = pd.read_csv('data/ftse100_data_processed.csv', index_col='Date', parse_dates=True)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Return']])

# Preparing the data for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X, y = create_dataset(data_scaled, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Splitting into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Building the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Making predictions
y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
y_test_scaled = scaler.inverse_transform([y_test])

# Evaluate the model
mse_lstm = mean_squared_error(y_test_scaled[0], y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(y_test_scaled[0], y_pred_lstm)
r2_lstm = r2_score(y_test_scaled[0], y_pred_lstm)

# Save the results
results_lstm = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
    'Value': [mse_lstm, rmse_lstm, mae_lstm, r2_lstm]
})
results_lstm.to_csv('results/lstm_results.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test_scaled[0], label='Actual')
plt.plot(y_pred_lstm, label='Predicted')
plt.legend()
plt.title('LSTM Predictions')
plt.savefig('results/lstm_plot.png')
plt.show()
