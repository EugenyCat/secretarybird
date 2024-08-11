import logging
from pipeline import ExtractBase
from database.ClickHouseFuncs import insert_data, remove_data, create_table, create_db
import os
import requests
import pandas as pd
import random
import time
import datetime
from dateutil.parser import parse as datetime_parser
import json
import pytz


class ETLBinance(ExtractBase):

    def __init__(self):
        super().__init__()
        self.api_link = os.getenv('BINANCE_API_LINK')
        self.possible_values_for_params = json.load(
            open(f"{os.getenv('JSON_EXTRACT_CURRENCIES_SETTINGS')}", 'r')
        )['binance_api']


    def validate_parameters(self, input_params):
        # Validate required parameters
        required_params = ['currency', 'interval']
        missing_params = [
            param
            for param in required_params
            if param not in input_params or not input_params[param]
        ]

        if missing_params:
            error = {
                "status": "error",
                "message": f"Missing or invalid parameters: {', '.join(missing_params)}",
            }
            return None, error

        # Validate params
        try:
            database = os.getenv('PREFIX_CRYPTO_CURRENCIES')

            currency = str(input_params['currency'])
            if currency not in self.possible_values_for_params['currency']:
                raise Exception(f"Currency value {currency} isn't contented in currencies.json file")

            interval = str(input_params['interval'])
            if interval not in self.possible_values_for_params['interval']:
                raise Exception(f"Interval value {interval} isn't contented in currencies.json file")

            start_date = datetime_parser(input_params.get('start') or self.define_start_time(database, currency, interval)).replace(tzinfo=pytz.UTC)

            end_date = datetime_parser(input_params.get('end') or datetime.datetime.now().replace(hour=23, minute=59, second=59).strftime('%Y-%m-%d %H:%M:%S')).replace(tzinfo=pytz.UTC)
            if end_date.time() == datetime.time(0, 0, 0):
                end_date = end_date.replace(hour=23, minute=59, second=59)



        except Exception as e:
            logging.info(f"{str(e)}")
            error = {"status": "error", "message": str(e)}
            return None, error

        return {
            'start': start_date,
            'end': end_date,
            'currency': currency,
            'interval': interval,
            'database': database
        }, None


    async def fetch(self, session, url, params): #TODO: rewrite everything into asyns requests with sertificate ssl
        async with session.get(url, params=params) as response:
            return await response.text()

    def extract_data_from_binance_api(self, currency, interval, start_str, end_str):
        """
        params:
            'currency' - currency, e.g. 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT'
            'interval' - one of these values '1m' (1 minute), '3m', '5m', '15m', '30m', '1h' (1 hour), '2h', '4h', '6h', '8h', '12h', '1d' (1 day), '3d', '1w', '1M',
            'start_str' -
            'end_str' -
        return:
            pandas dataframe
        """

        # Set params for request to API Binance
        params = {
            'symbol': currency,
            'interval': interval,
            'startTime': start_str,
            'endTime': end_str
        }

        """
        async with session.get(self.api_link, params=params) as response:
            return await response.text()
        """

        # Getting data
        response = requests.get(self.api_link, params=params)
        data = response.json()

        # Parse the data into a DataFrame
        columns = [
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
            'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
            'Taker_buy_quote_asset_volume', 'Ignore'
        ]
        df_currency_time_series = pd.DataFrame(data, columns=columns)

        return df_currency_time_series


    def transform_data_from_binance_api(self, df_binance_time_series: pd.DataFrame, currency, interval):

        # Convert the timestamp columns to datetime
        df_binance_time_series['Open_time'] = pd.to_datetime(df_binance_time_series['Open_time'], unit='ms')
        df_binance_time_series['Close_time'] = pd.to_datetime(df_binance_time_series['Close_time'], unit='ms')

        # Drop unnecessary fields and add some new info fields
        df_binance_time_series.drop(['Ignore'], axis=1, inplace=True)
        df_binance_time_series['Number_of_trades'] = df_binance_time_series['Number_of_trades'].apply(
            lambda v: str(v))
        df_binance_time_series['Currency'] = currency
        df_binance_time_series['Interval'] = interval
        df_binance_time_series['Source'] = 'binance_api'

        return df_binance_time_series



    def load_and_insert_data_into_ch_db(self, input_params):

        start = input_params['start']
        end = input_params['end']
        currency = input_params['currency']
        interval = input_params['interval']
        database = input_params['database']

        data_ranges = self.get_list_of_intervals(interval, start, end)
        data_ranges_ms = self.convert_intervals_into_ms(data_ranges)

        for i in range(len(data_ranges)):
            start_1, end_1, start_2, end_2 = data_ranges[i]
            start_ms_1, end_ms_1, start_ms_2, end_ms_2 = data_ranges_ms[i]


            # for load daya for 1 day (usually for a couple of hours)
            if end_ms_1 == start_ms_2 == self.null_datetime_ms:
                extracted_data = self.extract_data_from_binance_api(currency=currency, interval=interval,
                                                                           start_str=start_ms_1,
                                                                           end_str=end_ms_2)

                transformed_data_part = self.transform_data_from_binance_api(
                    df_binance_time_series=extracted_data,
                    currency=currency, interval=interval)

                data_tuples = [list(x) for x in transformed_data_part.to_numpy()]

            # for load data more than 1 day (usually for a long period)
            else:
                extracted_data_part_1 = self.extract_data_from_binance_api(currency=currency, interval=interval,
                                                                           start_str=start_ms_1,
                                                                           end_str=end_ms_1)

                transformed_data_part_1 = self.transform_data_from_binance_api(df_binance_time_series=extracted_data_part_1,
                                                                               currency=currency, interval=interval)

                extracted_data_part_2 = self.extract_data_from_binance_api(currency=currency, interval=interval,
                                                                           start_str=start_ms_2,
                                                                           end_str=end_ms_2)

                transformed_data_part_2 = self.transform_data_from_binance_api(df_binance_time_series=extracted_data_part_2,
                                                                               currency=currency, interval=interval)

                data_tuples = [list(x) for x in pd.concat([transformed_data_part_1, transformed_data_part_2]).to_numpy()]


            if len(data_tuples) > 0:
                # TODO: 1) hide scripts

                # Create db if not exists
                create_db(self.db_session, database, interval)
                """
                create_db = f"CREATE DATABASE IF NOT EXISTS {database}"
                self.db_session.query(create_db)
                """

                # Create table if not exists
                create_table(self.db_session, database, currency, interval)
                """
                create_tab_request = f"CREATE TABLE IF NOT EXISTS {database}.{currency.lower()}  ( \
	                                                Currency String, \
	                                                Interval String, \
                                                    Open_time DateTime, \
                                                    Open Float64, \
                                                    High Float64, \
                                                    Low Float64, \
                                                    Close Float64, \
                                                    Volume Float64, \
                                                    Close_time DateTime, \
                                                    Quote_asset_volume Float64, \
                                                    Number_of_trades UInt32, \
                                                    Taker_buy_base_asset_volume Float64, \
                                                    Taker_buy_quote_asset_volume Float64, \
                                                    Source String DEFAULT 'binance_api'\
                                    ) ENGINE = MergeTree() \
                                    PARTITION BY toYYYYMM(Open_time) \
                                    ORDER BY (Open_time);"
                self.db_session.query(create_tab_request)
                """

                # Remove data for the same period
                remove_data(self.db_session, database, currency, interval, start_1, end_2)

                # Insert operation
                insert_data(self.db_session, database, currency, interval, data_tuples)

                time.sleep(random.uniform(0.19, 0.53))

        return 'insert_data_into_ch_db finishes successfully'


    def run_etl(self, input_params):

        self.db_session = self.ch_connection_obj.get_session()

        # Validate params
        validated_params, error = self.validate_parameters(input_params)
        logging.info(f'validate_parameters is okey')

        if error:
            return None, error

        try:
            result = self.load_and_insert_data_into_ch_db(validated_params)

            return {"status": "success", "result": result}, None
        except Exception as e:
            logging.info(f"Code 500: error process 'load_and_insert_data_into_ch_db' {e}")
            message = f"Code 500: error process 'load_and_insert_data_into_ch_db'"
            return None, {"status": "success", "message": message}
        finally:
            self.ch_connection_obj.close_session()
