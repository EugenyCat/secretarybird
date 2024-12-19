from typing import Protocol
from etl_pipeline.database.clickHouseConnection import ClickHouseConnection
from functools import wraps
import time

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


class ETLProtocol(Protocol):
    """
        pattern Abstract Factory
        ETLProtocol - ETL process interface from different sources
    """

    def validate_parameters(self, input_params: dict) -> (dict, int):
        """
            Validate required parameters
        """
        ...

    def extract_data(self) -> (dict or None, None or dict):
        """
            Extract data from source
        """
        ...

    def transform_data(self, extracted_data: list) -> (dict or None, None or dict):
        """
            Transform raw data into a Pandas DataFrame with the correct fieldnames and datatypes.
        """
        ...

    def load_data(self, transformed_data: list) -> (dict or None, None or dict):
        """
            Insert data into ClickHouse db
        """
        ...

    def run_etl(self, input_params: dict) -> (dict, dict):
        """
            Orchestrates the entire ETL process: extraction, transformation, and loading of data.
        """
        ...


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

    def set_database(self, database=None):
        self.database = database
        return self

    def set_modelname(self, modelname=None):
        self.modelname = modelname
        return self

    def set_model(self, model=None):
        self.model = model
        return self

    def set_data(self, data=None):
        self.data = data
        return self