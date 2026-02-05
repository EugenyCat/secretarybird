from typing import Protocol
from pipeline.database.clickHouseConnection import ClickHouseConnection
from functools import wraps
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd


def retry(retries=5, backoff_factor=0.1):
    """
        A decorator to automatically retry a function call on failure.

        This decorator wraps a function and retries it up to a specified number
        of times if it raises an exception. The retries occur with an increasing
        delay between attempts, calculated as `backoff_factor` plus an additional
        delay that increases with each retry attempt.

        Parameters:
            retries (int): The maximum number of retry attempts (default is 5).
            backoff_factor (float): The initial delay between retries in seconds
                                    (default is 0.1).

        Returns:
            function: The wrapped function that will be retried on failure.

        Raises:
            Exception: If all retry attempts fail, the last exception is raised.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(backoff_factor + (0.5 * attempt))
                    else:
                        raise e

        return wrapper

    return decorator


class ConfigurationBuilder:
    """
        A builder class for setting various configuration parameters using a fluent interface pattern.
        Each setter method allows chaining by returning the instance itself.
    """

    def __init__(self):
        # ClickHouse connection object
        self.clickhouse_conn = ClickHouseConnection()
        self.db_client_session = None
        self.db_sqlalchemy_session = None

        # TS params
        self.database = None
        self.currency = None
        self.interval = None
        self.processing_stage = None
        self.start = None
        self.end = None

        # ML and NN params
        self.modelname = None
        self.model = None
        self.data = None


    def set_client_session(self, db_client_session=None):
        self.db_client_session = db_client_session
        return self

    def set_sqlalchemy_session(self, db_sqlalchemy_session=None):
        self.db_sqlalchemy_session = db_sqlalchemy_session
        return self

    def set_start(self, start=None):
        self.start = start
        return self

    def set_end(self, end=None):
        self.end = end
        return self

    def set_currency(self, currency=None):
        self.currency = currency.lower()
        return self

    def set_interval(self, interval=None):
        self.interval = interval
        return self

    def set_processing_stage(self, processing_stage=None):
        if processing_stage not in ['raw', 'transformed']:
            # todo: pass and return error
            pass
        self.processing_stage = processing_stage
        return self

    def set_database(self, database=None):
        self.database = database
        return self

    def set_modelname(self, modelname=None):
        self.modelname = modelname
        return self

    def set_model(self, model=None):
        self.model = model
        return self

    def set_data(self, X=None, y=None):
        self.data = {'X': X, 'y': y}
        return self

    def get_ts_id(self) -> str:
        """
        Generates a time series ID based on currency and interval.

        Returns:
            str: The time series ID in the format "currency_interval"

        Raises:
            ValueError: If currency or interval is not set.
        """
        # Validate that currency and interval are set
        if not self.currency or not self.interval:
            missing = []
            if not self.currency:
                missing.append('currency')
            if not self.interval:
                missing.append('interval')
            raise ValueError(f"Cannot generate ts_id: missing {', '.join(missing)}")

        return f"{self.currency}_{self.interval}"