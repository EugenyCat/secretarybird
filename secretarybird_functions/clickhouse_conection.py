import os
from dotenv import load_dotenv
import clickhouse_connect
from secretarybird_functions.get_data_from_binance_api import get_historical_klines
import pandas as pd
import time

load_dotenv()


# 1. Init db connection
CH_HOST = os.getenv('CH_HOST')
CH_PORT = os.getenv('CH_PORT')
CH_USER = os.getenv('CH_USER')
CH_PASSWORD = os.getenv('CH_PASSWORD')

#client = Client(host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASSWORD, database='CRYPTO_1h')
client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASSWORD, database='CRYPTO_1h')


# 2. Insert data
for year in range(2010, 2025):
    for month in range(1, 13):
        start_of_month = pd.Timestamp(year, month, 1)
        end_of_month = pd.Timestamp(year, month, pd.Timestamp(year, month, 1).days_in_month, 23, 59, 59)


        start_of_month_ms = int(start_of_month.timestamp() * 1000)
        end_first_half_ms = int(pd.Timestamp(year, month, 15, 23, 59, 59, 59).timestamp() * 1000)

        btc_data_part_1 = get_historical_klines(symbol='BTCUSDT', interval='1h', start_str=start_of_month_ms, end_str=end_first_half_ms)

        start_second_half_ms = int(pd.Timestamp(year, month, 16).timestamp() * 1000)
        end_of_month_ms = int(end_of_month.timestamp() * 1000)

        btc_data_part_2 = get_historical_klines(symbol='BTCUSDT', interval='1h', start_str=start_second_half_ms, end_str=end_of_month_ms)

        print(f'--- {start_of_month}-{end_of_month}')
        # Construct the insert query
        query = ("INSERT INTO CRYPTO_1h.bitcoin ( \
                 Open_time, \
                 Open, \
                 High, \
                 Low, \
                 Close, \
                 Volume, \
                 Close_time, \
                 Quote_asset_volume, \
                 Number_of_trades, \
                 Taker_buy_base_asset_volume, \
                 Taker_buy_quote_asset_volume, \
                 Currency\
                 ) \
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )

        data_tuples = [list(x) for x in pd.concat([btc_data_part_1, btc_data_part_2]).to_numpy()]

        remove_request = f"ALTER TABLE CRYPTO_1h.bitcoin DELETE WHERE Open_time >= '{start_of_month}' AND Open_time <= '{end_of_month}'"
        client.query(remove_request)

        client.insert('CRYPTO_1h.bitcoin', data_tuples,
                      column_names=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                                    'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                                    'Taker_buy_quote_asset_volume', 'Currency'])

        time.sleep(0.5)
