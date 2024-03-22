import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load your Bitcoin price data (replace with your actual data)
df = pd.read_csv('data/bitcoin_price_data.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Fit ARIMA model
model = ARIMA(df['price'], order=(5, 1, 0))  # Example order (p, d, q)
model_fit = model.fit()

# Make predictions
forecast_steps = 31  # Adjust as needed
forecast = model_fit.forecast(steps=forecast_steps)

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(df.index[-forecast_steps:], forecast, label='ARIMA Forecast')
plt.plot(df.index[-forecast_steps:], df['price'].iloc[-forecast_steps:], label='Actual BTC Price')
plt.legend()
plt.title('Bitcoin Price Forecast using ARIMA')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.show()

# Print the forecasted values
print("Forecasted BTC prices:")
print(forecast)
