import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_market_calendars as mcal
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Download Stock Data (Example: Reliance - NSE)
stock_symbol = 'RELIANCE.NS'  # NSE-listed stock
data = yf.download(stock_symbol, start='2000-01-01', end='2025-02-27')

# Use multiple features for prediction
features = data[['Close', 'Open', 'High', 'Low', 'Volume']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(features)

# Create sequences for LSTM
sequence_length = 60
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting 'Close' price
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, sequence_length)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build Optimized LSTM Model
model = Sequential([
    Input(shape=(sequence_length, X.shape[2])),
    Bidirectional(LSTM(units=100, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(units=50, return_sequences=False)),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# Early Stopping & Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train Model
epochs = 200  # Increased epochs for better learning
batch_size = 64  # Experimenting with different batch sizes
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Make Predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), features.shape[1] - 1)))))[:, 0]  # Convert back to original scale
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), features.shape[1] - 1)))))[:, 0]

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, y_pred)

# Compute Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Compute Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test_actual, y_pred)

# Print results
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Get corresponding dates for test data
test_dates = data.index[split_index + sequence_length:]

# Plot Results with Date Labels
plt.figure(figsize=(14, 5))
plt.plot(test_dates, y_test_actual, label='Actual Price')
plt.plot(test_dates, y_pred, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.legend()
plt.title(f'{stock_symbol} Stock Price Prediction using Optimized LSTM')
plt.show()

# Real-time Prediction
def predict_next_day(last_60_days, model, scaler):
    last_60_days_scaled = scaler.transform(last_60_days)
    X_input = np.array([last_60_days_scaled])
    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(np.hstack((predicted_price, np.zeros((1, features.shape[1] - 1)))))[:, 0]
    return predicted_price[0]

# Get last 60 days of data for real-time prediction
last_60_days = features[-60:]
predicted_next_price = predict_next_day(last_60_days, model, scaler)

# Get next trading day's date (skip weekends & NSE holidays)
nse = mcal.get_calendar('XNSE')  # NSE Market Calendar
holidays = nse.holidays().holidays  # List of holidays


next_day = data.index[-1] + pd.Timedelta(1,"D")
while next_day.weekday() >= 5 or next_day in holidays:  # Skip weekends & holidays
    next_day += pd.Timedelta(days=1)

print(f'Predicted stock price for {next_day.date()}: {predicted_next_price:.2f} INR')