import requests


def get_bitcoin_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin"
    response = requests.get(url)
    data = response.json()
    return data


bitcoin_data = get_bitcoin_data()
print(bitcoin_data)
