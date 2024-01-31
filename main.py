import pandas as pd
import requests


def parse_interval(interval) -> str:
    match interval:
        case 'hourly':
            return 'h'
        case 'daily':
            return 'D'
        case 'weekly':
            return 'W'
        case _:
            raise ValueError(f"interval not supported {interval}")


def fetch_and_process_data(url, interval):
    # Fetch the data
    response = requests.get(url)
    data = response.json()

    # Create a DataFrame and convert timestamps
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['datetime'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices.set_index('datetime', inplace=True)

    rule = parse_interval(interval)
    resampled_data = prices.resample(rule).mean()

    return resampled_data


# Base URL for CoinGecko API
base_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd"

# Short-Term Analysis (1-3 days, hourly data)
short_term_url = f"{base_url}&days=3"
short_term_data = fetch_and_process_data(short_term_url, 'hourly')
short_term_data.to_csv('data/short_term_data.csv')

# Medium-Term Analysis (2-4 weeks, daily data)
medium_term_url = f"{base_url}&days=28&interval=daily"
medium_term_data = fetch_and_process_data(medium_term_url, 'daily')
medium_term_data.to_csv('data/medium_term_data.csv')

# Long-Term Analysis (3-6 months, weekly data)
long_term_url = f"{base_url}&days=180"
long_term_data = fetch_and_process_data(long_term_url, 'weekly')
long_term_data.to_csv('data/long_term_data.csv')
