import pandas as pd
import requests
from datetime import datetime, timedelta


def get_historical_klines(symbol, interval, start_str, end_str=None):
    """
    params:
        'symbol' - currency, e.g. 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT'
        'interval' - one of these values '1m' (1 minute), '3m' (3 minutes), '5m' (5 minutes), '15m' (15 minutes), '30m' (30 minutes), '1h' (1 hour), '2h' (2 hours),
                                        '4h' (4 hours), '6h' (6 hours), '8h' (8 hours), '12h' (12 hours), '1d' (1 day), '3d' (3 days), '1w' (1 week), '1M' (1 month).
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_str,
    }
    if end_str:
        params['endTime'] = end_str
    response = requests.get(url, params=params)
    data = response.json()

    # Parse the data into a DataFrame
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
        'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
        'Taker buy quote asset volume', 'Ignore'
    ]
    df = pd.DataFrame(data, columns=columns)

    # Convert the timestamp columns to datetime
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms') #.dt.strftime('%Y-%m-%d %H:%M:%S') # pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms') #.dt.strftime('%Y-%m-%d %H:%M:%S')

    # Set the index to the open time
    # df.set_index('Open time', inplace=True)
    df.rename(
        columns={
            "Open time": "Open_time",
            "Close time": "Close_time",
            "Quote asset volume": "Quote_asset_volume",
            "Number of trades": "Number_of_trades",
            "Taker buy base asset volume": "Taker_buy_base_asset_volume",
            "Taker buy quote asset volume": "Taker_buy_quote_asset_volume"
        },
        inplace=True
    )

    df.drop(['Ignore'], axis=1, inplace=True)
    df['Number_of_trades'] = df['Number_of_trades'].apply(lambda v: str(v))
    df['Currency'] = symbol
    df = df.fillna('0')

    return df


# Example usage
"""
# Define currency
symbol = 'BTCUSDT'

# Define interval
interval = '4h'  # daily data #

# Define startTime (as YYYY-mm-dd)
date_string = "2006-01-01"
date_object = datetime.strptime(date_string, "%Y-%m-%d")
# convert the datetime object to milliseconds
start_str = int(date_object.timestamp() * 1000)

btc_data = get_historical_klines(symbol, interval, start_str)
print(f"Data for interval: {btc_data.index.min()}-{btc_data.index.max()}")
btc_data.tail()
"""