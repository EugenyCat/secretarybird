from pipeline.helpers.setup import ConfigurationBuilder, retry
from pipeline.db_schema_manager.models.orm_models import TimeSeriesDefinition
from pipeline.db_schema_manager.models.ts_schema_factories import CryptoRawSchemaFactory, CryptoTransformedSchemaFactory
import logging
import functools

class DBFuncsBase(ConfigurationBuilder):
    """
        Base class for ClickHouse database operations, focused on configuration queries
        and schema management.

        This class provides methods for accessing key configuration tables like TimeSeriesDefinition
        and for retrieving and caching schema information for time series tables.
    """

    # Class-level schema cache
    _schema_cache = {}

    # Set of system/technical databases
    _tech_db = {'INFORMATION_SCHEMA', 'default', 'information_schema', 'system'}

    def __init__(self):
        super().__init__()

    def validate_ts_parameters(self):
        """
        Validates that the required time series parameters are set.

        Raises:
            ValueError: If any of the required parameters (processing_stage, currency, interval) are missing.
        """
        missing_params = []

        if not self.database:
            missing_params.append('database')
        if not self.processing_stage:
            missing_params.append('processing_stage')
        if not self.currency:
            missing_params.append('currency')
        if not self.interval:
            missing_params.append('interval')


        if missing_params:
            raise ValueError(f"Missing required time series parameters: {', '.join(missing_params)}")

    @functools.lru_cache(maxsize=128)
    def get_table_columns(self):
        """
            Get the table columns from the appropriate schema factory based on database and processing_stage.
            Uses functools.lru_cache for efficient caching of schema information.

            Returns:
                List[Tuple[str, str]]: A list of tuples defining the column names and data types for the table.

            Raises:
                ValueError: If required parameters are missing or if schema generation fails.
        """
        # Validate required parameters
        self.validate_ts_parameters()

        # Generate a cache key based on the current parameters including database
        cache_key = (self.database, self.processing_stage, self.currency, self.interval)

        # Check if the schema is already in the cache
        if cache_key in self.__class__._schema_cache:
            return self.__class__._schema_cache[cache_key]

        try:
            # Map database+processing_stage combinations to factory classes
            factory_map = {
                ('CRYPTO', 'raw'): CryptoRawSchemaFactory,
                ('CRYPTO', 'transformed'): CryptoTransformedSchemaFactory,

                # Add mappings for future database types here:
                # ('STOCKS', 'raw'): StocksRawSchemaFactory,
                # ('STOCKS', 'transformed'): StocksTransformedSchemaFactory,
            }

            # Normalize database name for lookup
            db_key = self.database.upper()

            # Get the appropriate factory class based on database and processing_stage
            factory_key = (db_key, self.processing_stage)
            factory_class = factory_map.get(factory_key)

            if not factory_class:
                raise ValueError(
                    f"No schema factory found for {self.database} with {self.processing_stage} processing stage")

            # Instantiate the factory
            factory = factory_class()

            # Generate a ts_id
            ts_id = self.get_ts_id()

            # Create the columns of the schema
            columns_schema = factory.create_schema(ts_id, self.processing_stage)

            # Cache the columns of the schema
            self.__class__._schema_cache[cache_key] = columns_schema

            # Return the columns of the schema
            return columns_schema
        except ImportError as e:
            logging.error(f"Schema factory modules not available: {e}")
            raise ValueError("Schema factory modules could not be imported. Schema generation failed.")
        except Exception as e:
            logging.error(f"Error getting schema from factory: {e}")
            raise ValueError(
                f"Failed to generate schema for {self.database}.{self.get_ts_id()}_{self.processing_stage}: {str(e)}")


    @classmethod
    def clear_schema_cache(cls):
        """Clear the schema cache."""
        cls._schema_cache = {}


    def validate_data_against_schema(self, data):
        """
            # TODO: do we need this method ? maybe deprecated
            Validates that the provided data matches the expected schema.

            Args:
                data: List of data rows to validate

            Returns:
                bool: True if the data is valid according to the schema

            Raises:
                ValueError: If the data doesn't match the schema
        """
        if not data:
            return True

        columns = self.get_table_columns()
        expected_fields = len(columns)

        # Check if all rows have the correct number of fields
        for i, row in enumerate(data):
            if len(row) != expected_fields:
                raise ValueError(f"Row {i} has {len(row)} fields, expected {expected_fields}")

        return True


    @retry()
    def get_time_series_definitions(self, source_name, only_active=True):
        """
        Retrieves time series definitions from the database for the specified source.

        Args:
            source_name (str): The name of the data source (e.g., 'binance_api')
            only_active (bool): Flag to filter only active time series

        Returns:
            list: A list of TimeSeriesDefinition objects

        Raises:
            Exception: If an error occurs while executing the database query
        """
        try:
            # Build the query
            query = self.db_sqlalchemy_session.query(TimeSeriesDefinition)
            query = query.filter(TimeSeriesDefinition.source == source_name)

            if only_active:
                query = query.filter(TimeSeriesDefinition.is_active == True)

            # Execute the query and get the results
            time_series = query.all()

            return time_series

        except Exception as e:
            logging.error(f"Error retrieving time series definitions: {e}")
            raise
        finally:
            # Close the session
            self.db_sqlalchemy_session.close()