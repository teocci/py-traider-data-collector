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
            raise ValueError(f'interval not supported {interval}')


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


def fetch_historical_btc_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',  # Specify the currency (USD in this case)
        'days': '365',  # Historical data for the last 365 days
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Create a DataFrame
    df = pd.DataFrame(data['prices'], columns=['datetime', 'price'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)

    # Save to CSV
    df.to_csv('data/bitcoin_price_data.csv')
    print('Bitcoin price data saved to bitcoin_price_data.csv')

    return df


# Base URL for CoinGecko API
base_url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd'

# Short-Term Analysis (1-3 days, hourly data)
short_term_url = f'{base_url}&days=3'
short_term_data = fetch_and_process_data(short_term_url, 'hourly')
short_term_data.to_csv('data/short_term_data.csv')

# Medium-Term Analysis (2-4 weeks, daily data)
medium_term_url = f'{base_url}&days=28&interval=daily'
medium_term_data = fetch_and_process_data(medium_term_url, 'daily')
medium_term_data.to_csv('data/medium_term_data.csv')

# Long-Term Analysis (3-6 months, weekly data)
long_term_url = f'{base_url}&days=180'
long_term_data = fetch_and_process_data(long_term_url, 'weekly')
long_term_data.to_csv('data/long_term_data.csv')

fetch_historical_btc_data()
