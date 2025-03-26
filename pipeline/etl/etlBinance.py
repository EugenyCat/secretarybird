from pipeline.helpers.setup import ETLProtocol, ConfigurationBuilder
from pipeline.helpers.utils import ETLUtils
from pipeline.helpers.db_funcs_etl import DBFuncsETL
from pipeline.helpers.telegram_notifier import TelegramNotifier
from system_files.constants.constants import DEFAULT_DATA_START_LOAD
import os
import requests
import pandas as pd
import random
import time
import datetime
from dateutil.parser import parse as datetime_parser
import json
import pytz
import logging


class ETLBinance(ETLProtocol, ConfigurationBuilder):
    """
        ETLBinance - Extract-Transform-Load process for Binance API
    """

    __SOURCE_NAME = 'binance_api'

    def __init__(self):
        # Initialize the parent classes
        ConfigurationBuilder.__init__(self)  # Call the initialization method for ConfigurationBuilder

        # Init TelegramNotifier
        self.notifier = TelegramNotifier()

        # api_link - binance api url
        self.__api_link = os.getenv('BINANCE_API_LINK')

        # path to json file that contains settings for etl
        self.load_params = json.load(
            open(f"{os.getenv('JSON_EXTRACT_CURRENCIES_SETTINGS')}", 'r')
        )[self.__SOURCE_NAME]


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
                'status': 'error',
                'message': f"Missing or invalid parameters: {', '.join(missing_params)}",
            }
            return None, error

        # Validate params
        try:
            database = self.load_params['database']

            currency = str(input_params['currency'])
            if currency not in self.load_params['currency']:
                raise Exception(f'[etl_manager/etlBinance.py] Currency value {currency} isn\'t contented in currencies.json file')

            interval = str(input_params['interval'])
            if interval not in self.load_params['interval']:
                raise Exception(f'[etl_manager/etlBinance.py] Interval value {interval} isn\'t contented in currencies.json file')

            (   # self ETLBinance
                self.set_database(database)
                .set_currency(currency)
                .set_interval(interval)
            )

            # Set etl_utils: various ETL helper functions
            self.etl_utils = (
                ETLUtils()
                .set_client_session(self.db_client_session)
                .set_database(self.database)
                .set_currency(self.currency)
                .set_interval(self.interval)
            )

            # Set db_funcs: various database operations
            self.db_funcs = (
                DBFuncsETL()
                .set_client_session(self.db_client_session)
                .set_database(self.database)
                .set_currency(self.currency)
                .set_interval(self.interval)
            )

            # Get the start time from input parameters or define it based on database data
            try:
                max_data = self.db_funcs.get_extreme_date()
                # Add 1 second to avoid duplicate data entries
                start_time_raw = (max_data + datetime.timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logging.warning(
                    f"[etl_manager/etlBinance.py] Error of getting "
                    f"the max data from {self.database}.{self.currency.lower()}_{self.interval}: {e}"
                )
                # Get the default start time value
                start_time_raw = DEFAULT_DATA_START_LOAD

            # Parse the raw start time string into a datetime object and Set the timezone to UTC
            start_date = datetime_parser(start_time_raw).replace(tzinfo=pytz.UTC)

            # Get the end time from input parameters or use the current date
            end_time_raw = input_params.get('end') or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Parse the raw end time string into a datetime object and Set the timezone to UTC
            end_date = datetime_parser(end_time_raw).replace(tzinfo=pytz.UTC)
            # Set end time to 23:59:59 for the same day
            end_date = end_date.replace(hour=23, minute=59, second=59)

            (
                # Set start time and end time for self ETLBinance
                self.set_start(start_date)
                .set_end(end_date)
            )

            (   # Set start time and end time for etl_utils
                self.etl_utils
                .set_start(self.start)
                .set_end(self.end)
            )

        except Exception as e:
            logging.info(f'{str(e)}')
            error = {'status': 'error', 'message': str(e)}
            return None, error

        return {'status': 'success', 'message': 'input params are validated'}, None


    def extract_data(self) -> (dict or None, None or dict):
        # Getting data
        response_crypto_prices = (requests.get(self.__api_link, params={
            'symbol': self.currency.upper(),
            'interval': self.interval,
            'startTime': int(self.start.timestamp() * 1000),
            'endTime': int(self.end.timestamp() * 1000)
        })).json()

        # If success
        if isinstance(response_crypto_prices, list) and len(response_crypto_prices) > 0:
            logging.info(f'{self.__str__()}: the `extract_data` process finishes successfully.')
            return {'status': 'success', 'result': response_crypto_prices}, None

        # Else error message
        error_message = f'{self.__str__()}: the `extract_data` process error [NO DATA EXTRACTED].'
        logging.warning(error_message)
        return None, {'status': 'error', 'message': error_message}


    def transform_data(self, extracted_data: list) -> (dict or None, None or dict):
        # Parse the data into a DataFrame
        columns = [
            'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
            'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
            'Taker_buy_quote_asset_volume', 'Ignore'
        ]
        df_crypto_prices = pd.DataFrame(extracted_data, columns=columns)

        # Convert the timestamp columns to datetime
        df_crypto_prices['Open_time'] = pd.to_datetime(df_crypto_prices['Open_time'], unit='ms')
        df_crypto_prices['Close_time'] = pd.to_datetime(df_crypto_prices['Close_time'], unit='ms')

        # Drop unnecessary fields and add some new info fields
        df_crypto_prices.drop(['Ignore'], axis=1, inplace=True)
        df_crypto_prices['Number_of_trades'] = df_crypto_prices['Number_of_trades'].apply(
            lambda v: str(v))
        df_crypto_prices['Currency'] = self.currency
        df_crypto_prices['Interval'] = self.interval
        df_crypto_prices['Source'] = self.__SOURCE_NAME

        data_tuples = [list(x) for x in df_crypto_prices.to_numpy()]

        if len(data_tuples) > 0:
            logging.info(f'{self.__str__()}: the `transform_data` process finishes successfully.')
            return {'status': 'success', 'result': data_tuples}, None

        # Else error message
        error_message = f'{self.__str__()}: the `transform_data` process error [NO DATA].'
        logging.warning(error_message)
        return None, {'status': 'error', 'message': error_message}


    def load_data(self, prepared_data: list) -> (dict or None, None or dict):

        if len(prepared_data) > 0:

            # Create db if not exists
            self.db_funcs.create_db()

            # Create table if not exists
            self.db_funcs.create_table()

            # Remove data for the same period
            self.db_funcs.remove_data()

            # Insert operation
            self.db_funcs.insert_data(prepared_data)

            time.sleep(random.uniform(0.02, 0.2))

            message = f'{self.__str__()}: the `load_data` process finishes successfully.'
            logging.info(f'{message}')
            return {'status': 'success', 'message': f'{message}'}, None

        message = f'{self.__str__()}: the `load_data` process [ERROR]: No data to insert.'
        logging.warning(f'{message}')
        return None, {'status': 'error', 'message': f'{message}'}


    def run_etl(self, input_params):
        # Create db connection
        self.set_client_session(self.clickhouse_conn.get_client_session())

        # Validate params
        validated_params, error = self.validate_parameters(input_params)

        if error:
            self.notifier.notify(error['message'])
            return None, error

        logging.info(f'The validate_parameters process finished successfully: {self.__str__()}')

        try:

            data_ranges = self.etl_utils.get_list_of_intervals()

            for i in range(len(data_ranges)):
                start, end = data_ranges[i]

                self.set_start(start) \
                    .set_end(end)

                self.db_funcs.set_start(start) \
                    .set_end(end)

                # 1. Extract
                extract_response, error = self.extract_data()
                if error:
                    #self.notifier.notify(error['message'])
                    continue

                # 2. Transform
                transformed_response, error = self.transform_data(extract_response['result'])
                if error:
                    self.notifier.notify(error['message'])
                    continue

                # 3. Load
                result, error = self.load_data(transformed_response['result'])

                if error:
                    self.notifier.notify(error['message'])
                    return None, error

            return {"status": "success", "result": f'{self.__str__()} finished successfully'}, None
        except Exception as e:
            message = f"Code 500: `run_etl` process [ERROR]: {self.__str__()}"
            self.notifier.notify(message)
            logging.error(f"{message}: {e}")
            return None, {"status": "error", "message": message}
        finally:
            self.clickhouse_conn.close_session()
            self.etl_utils = None
            self.db_funcs = None

    def __str__(self):
        return f'<[etl_manager/etlBinance.py] {self.currency}_{self.interval}: {self.start}-{self.end}>'