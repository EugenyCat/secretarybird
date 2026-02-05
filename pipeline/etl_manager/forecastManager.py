from pipeline.etl_manager.baseManager import BaseManager
import subprocess
import logging
from system_files.constants.constants import CH_BACKUPS_PATH, CH_OLD_BACKUPS_SUBDIRECTORY


class ForecastManager(BaseManager):
    """
        todo: write docstring
    """

    def __init__(self, source_name, use_extended_timeout=False):
        """
            Initializes the ClickHouseBackupManager instance, setting up the backup management
            environment and ensuring required directories are in place.

            During initialization, this method performs the following actions:
            - Calls the parent class initializer to set up the base manager with configurations
              and database connection settings.
            - Ensures that the backup subdirectory exists within the ClickHouse Docker container.

            Args:
                source_name (str): The name of the source configuration to be used for the backup management.
                use_extended_timeout (bool): Optional flag to set an extended connection timeout
                                             if longer database operations are expected.
        """

        super().__init__(source_name)  # Call the parent class initializer


    def get_train_data(self, tablename: str):
        currency, interval = tablename.split('_')

        self.db_funcs.set_currency(currency).set_interval(interval)

        return self.db_funcs.get_data()

