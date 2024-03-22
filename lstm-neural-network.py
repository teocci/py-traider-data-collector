import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# Load your Bitcoin price data (replace with your actual data)
df = pd.read_csv('data/bitcoin_price_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Normalize data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['price']])

# Create sequences for LSTM
sequence_length = 30  # Adjust as needed
X, y = [], []
for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled[i: i + sequence_length])
    y.append(df_scaled[i + sequence_length])

X, y = np.array(X), np.array(y)

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(Input(shape=(sequence_length, 1)))
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test)

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual BTC Price')
plt.plot(y_pred_actual, label='Predicted BTC Price')
plt.legend()
plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.show()

# Estimate BTC price at the halving event (April 2024)
predicted_price_halving = y_pred_actual[-1][0]
print(f"Predicted BTC price at the halving event (April 2024): ${predicted_price_halving:.2f}")
