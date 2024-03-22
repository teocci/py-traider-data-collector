import numpy as np
import pandas as pd
from keras.layers import Input, LSTM, Dense
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
sequence_length = 31
X, y = [], []
for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled[i: i + sequence_length])
    y.append(df_scaled[i + sequence_length])

X, y = np.array(X), np.array(y)

# Initialize an empty array to store predictions
all_predictions = []

# Run the LSTM estimator 1000 times
for _ in range(10):
    # Split into train and test sets (randomly)
    train_size = int(0.8 * len(X))
    np.random.shuffle(X)
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
    all_predictions.append(y_pred_actual[-1][0])

# Calculate the average prediction
average_prediction = np.mean(all_predictions)
print(f"Average BTC price prediction: ${average_prediction:.2f}")
