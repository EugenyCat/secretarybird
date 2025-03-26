from pipeline.helpers.setup import ConfigurationBuilder
from pipeline.helpers.db_funcs_etl import DBFuncsETL
from pipeline.helpers.telegram_notifier import TelegramNotifier
import logging
import os
import json

class BaseManager(ConfigurationBuilder):
    """
        BaseManager provides common initialization functionality for managing
        database sessions and configurations for both backup and data quality managers.

        This class should be inherited by classes that need to:
        - Initialize a database session.
        - Set up database configurations.
        - Manage DBFuncs instance for database-related operations.
    """

    # Variables for generating awesome alerts
    line_str = '⤵\n'
    line_split_messages = '\n✦•·················•✦•·················•✦\n'

    def __init__(self, source_name, use_extended_timeout=False):
        """
            Initializes the BaseManager instance, setting up a database connection,
            initializing database utility functions, and loading configuration settings.

            Args:
                source_name (str): The name of the source configuration to be used.
                use_extended_timeout (bool): Flag to indicate if a longer connection timeout
                                             should be set for the database connection.
        """
        super().__init__()  # Call the parent class initializer

        # Load API configurations from a JSON file specified by an environment variable
        try:
            config_path = os.getenv('JSON_EXTRACT_CURRENCIES_SETTINGS')
            with open(config_path) as config_file:
                config_data = json.load(config_file)
                self.__api_configurations = config_data.get(source_name, {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading API configurations: {e}")

        # Set an extended connection timeout if specified and set the session
        if use_extended_timeout:
            self.clickhouse_conn.set_connection_timeout()
        self.set_client_session(self.clickhouse_conn.get_client_session())

        # Set the database name from the loaded configurations
        self.set_database(self.__api_configurations['database'])

        # Initialize db_funcs to handle various database operations
        self.db_funcs = (
            DBFuncsETL()  # Create an instance of DBFuncsETL
            .set_client_session(self.db_client_session)
            .set_database(self.database)
        )

        # Initialize the TelegramNotifier for sending notifications about backup status
        self.notifier = TelegramNotifier()


    def get_api_configurations(self):
        return self.__api_configurations


    def get_ts_tables(self):
        """
            todo: add docstring
        """
        return self.db_funcs.get_table_names()