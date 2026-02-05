from pipeline.helpers.setup import retry
from pipeline.helpers.db_funcs_base import DBFuncsBase
import pandas as pd

class DBFuncsSQL(DBFuncsBase):
    """
        A class for handling ClickHouse database operations.

        This class provides methods for managing and manipulating ClickHouse tables, including:
        - Creating and dropping tables with schema awareness
        - Inserting and removing data
        - Working with different processing stages (raw or transformed)

        It extends DBFuncsBase which provides schema management functionality.
    """

    # This variable reflects how many `the unit` is contained in `the upper unit`.
    __valid_amount_of_rows = {
        "1h": 24,
        "12h": 2,
        "1d": 1,
        "3d": [9, 10],
        "1w": 4,
        "1M": 1
    }


    def get_tablename(self):
        """
            Generate and return the table name based on the current database, currency, interval and processing_stage.

            This method constructs the table name using the format:
            "{database}.{currency}_{interval}_{processing_stage}", where `database`, `currency`, `interval` and
            `processing_stage` are attributes of the current instance.

            Returns:
                str: The formatted table name.
        """
        self.validate_ts_parameters()

        return f'{self.database}.{self.currency}_{self.interval}_{self.processing_stage}'


    def get_table_names(self):
        """
            todo: add docstring
        """
        if not self.database:
            raise ValueError("Database name is not set")

        select_table_names = f"SELECT name FROM system.tables where database = '{self.database}'"
        response = self.db_client_session.query(select_table_names)
        return [table_name[0] for table_name in response.result_rows]


    @retry()
    def create_db(self):
        """
            Creates a database in ClickHouse if it does not already exist.

            This method constructs a SQL `CREATE DATABASE IF NOT EXISTS` statement
            using the `self.database` attribute to specify the database name.

            The operation is executed via the `db_client_session.query` method, which
            interacts with the ClickHouse server to create the database.

            If the database creation fails, the `retry` decorator will automatically
            attempt to retry the operation according to the specified retry policy.

            Raises:
                Exception: If all retry attempts fail, the last exception is raised.
        """
        if not self.database:
            raise ValueError("Database name is not set")

        create_db = f"CREATE DATABASE IF NOT EXISTS {self.database}"
        self.db_client_session.query(create_db)


    @retry()
    def create_table(self, suffix=''):
        """
        Creates a ClickHouse table with an optional suffix if it does not already exist.

        This method constructs a SQL `CREATE TABLE IF NOT EXISTS` statement based on the provided table schema,
        database, currency, and interval. The table will use the `MergeTree` engine, partitioned by the `Open_time`
        field in a YYYYMM format and ordered by `Open_time`.

        The table schema is dynamically generated from the `self.table_columns` attribute, which contains the field names
        and their corresponding data types. The optional `suffix` parameter allows for appending a string to the table name,
        enabling the creation of versioned or state-specific tables.

        If the table creation fails, the `retry` decorator will automatically attempt to retry the operation according to
        the specified retry policy.

        Args:
            suffix (str): An optional suffix to append to the table name, useful for versioning or different table states.

        Raises:
            Exception: If all retry attempts fail, the last exception is raised.
        """
        # This will validate parameters and raise errors if needed
        table_columns = self.get_table_columns()

        if not table_columns:
            raise ValueError(f"Failed to get table columns for {self.get_tablename()}")

        table_schema = ', '.join(f'{fieldname} {datatype}' for fieldname, datatype in table_columns)
        create_tab_request = f"CREATE TABLE IF NOT EXISTS {self.get_tablename()}{suffix}  (\
                                    {table_schema} \
                                    ) ENGINE = MergeTree() \
                                    PARTITION BY toYYYYMM(Open_time) \
                                    ORDER BY (Open_time);"

        self.db_client_session.query(create_tab_request)


    def drop_table(self, suffix=''):
        """
            Drop a ClickHouse table if it exists.

            This method constructs and executes a SQL `DROP TABLE IF EXISTS` statement to remove the specified table
            from the database. The optional `suffix` parameter allows for specifying a particular table version or state
            by appending a suffix to the table name.

            Args:
                suffix (str): An optional suffix to append to the table name, useful for versioning or different table states.
        """
        drop_request = f"""
                        DROP TABLE IF EXISTS {self.get_tablename()}{suffix};
                    """
        self.db_client_session.query(drop_request)


    @retry()
    def insert_data(self, data_tuples: list):
        """
            Inserts a batch of data into the specified ClickHouse table.

            This method attempts to insert the provided data tuples into a table
            formatted with the current database, currency, and interval. It uses
            the `self.db_client_session.insert` method to perform the insertion, with column
            names specified by `self.__column_names`.

            Parameters:
                data_tuples (list of tuples): The data to be inserted into the table.

            Raises:
                SomeException: If the insertion fails, retries according to the retry policy.
        """
        # Get column names dynamically from the schema
        table_columns = self.get_table_columns()

        self.db_client_session.insert(
            f'{self.get_tablename()}',
            data_tuples,
            column_names=[column[0] for column in table_columns]
        )

    @retry()
    def remove_data(self):
        """
            Removes data from a ClickHouse table for a specified time period.

            This function constructs and executes an SQL `ALTER TABLE DELETE` query
            to remove records from the specified ClickHouse table where the `Open_time`
            falls within the given start and end datetime range.

            Raises:
                Exception: If the query fails, the `retry` decorator will handle the retry logic.
        """
        if not self.start or not self.end:
            raise ValueError("Start and end times must be set for remove_data operation")

        remove_request = f"""
                            ALTER TABLE {self.get_tablename()} DELETE 
                            WHERE Open_time >= '{self.start.strftime('%Y-%m-%d %H:%M:%S')}' 
                            AND Open_time <= '{self.end.strftime('%Y-%m-%d %H:%M:%S')}'
                          """
        self.db_client_session.query(remove_request)


    def get_databases(self, filter_name: str = None):
        """
            Retrieves a list of existing databases from ClickHouse, excluding system and technical databases.

            This method queries the ClickHouse server to fetch all available databases and filters out
            technical/system databases like 'INFORMATION_SCHEMA', 'default', 'information_schema', and 'system'.
            If a filter name is provided, it will return only databases matching the filter.

            Args:
                filter_name (str, optional): The name or pattern to filter the databases. If provided,
                                              only databases matching this filter will be returned.

            Returns:
                list: A list of database names (str) that are available in ClickHouse, excluding technical databases.
                       If filter_name is provided, returns databases matching the filter.
        """
        # Base query to get all databases
        query = 'SHOW DATABASES'

        # Append LIKE clause if filter_name is provided
        if filter_name:
            query += f" LIKE '{filter_name}'"

        # Execute the query
        all_databases = self.db_client_session.query(query).result_rows

        # Return databases that are not in the technical databases list
        return [db[0] for db in all_databases if db[0] not in self._tech_db]


    def create_backup_database(self):
        """
            Creates a backup of the specified database.

            This method constructs a SQL backup request for the database associated
            with the instance. The backup is saved as a .zip file in the 'backups'
            directory, with a name that includes the database name followed by '-backup.zip'.

            Returns:
                str: The name of the created backup file.
        """

        # Construct the name for the backup file using the database name
        backup_name = f'{self.database}-backup.zip'

        # Create the SQL command to backup the database to the specified disk location
        backup_request = f"BACKUP DATABASE {self.database} TO Disk('backups', '{backup_name}')"

        # Execute the backup command using the current database session
        self.db_client_session.command(backup_request)

        # Return the name of the created backup file for reference
        return backup_name


    def restore_database(self):
        """
            Restores the database from its backup.

            This method restores the database by constructing the path to the backup file and
            executing a RESTORE command. The backup is assumed to be stored as a `.zip` file
            in the 'backups' directory. The function uses the current database session to
            execute the restore query.
        """

        # Construct the name for the backup file using the database name
        backup_name = f'{self.database}-backup.zip'

        # Define the backup path in the format required by ClickHouse (assuming it's stored on disk)
        backup_path = f"Disk('backups', '{backup_name}')"

        # Create the RESTORE SQL command to restore the database from the backup path
        restore_query = f"RESTORE DATABASE {self.database} FROM {backup_path}"

        # Execute the restore command using the current database session
        self.db_client_session.command(restore_query)


    @retry()
    def get_data(self):
        """
            Retrieve all data from the specified ClickHouse table.

            This method constructs and executes a SQL `SELECT *` query to fetch all records from the table
            specified by the `get_tablename` method. The query results are then converted into a Pandas DataFrame.

            The function is decorated with `@retry()` to handle transient errors during execution, such as
            database connectivity issues, by automatically retrying the query a specified number of times
            before giving up.

            Returns:
                pd.DataFrame: A DataFrame containing all columns and rows from the specified ClickHouse table.
                The DataFrame columns are named according to the columns in the ClickHouse table.
        """
        ch_response = self.db_client_session.query(f"""
                                    SELECT *
                                    FROM {self.get_tablename()}
                                    ORDER BY Open_time
                                """)

        return pd.DataFrame(ch_response.result_rows, columns=ch_response.column_names)


    @retry()
    def get_extreme_date(self, method='max'):
        """
            Retrieves the extreme (maximum or minimum) date from the Open_time field in the database.

            This method constructs and executes a SQL query to obtain either the maximum or minimum
            value of the `Open_time` field from the specified table in the database. The table is
            identified by combining the database name, currency, and interval attributes.

            Parameters:
                method (str): Specifies whether to retrieve the 'max' (default) or 'min' date.
                              This value is inserted directly into the SQL query.

            Returns:
                datetime: The maximum or minimum `Open_time` value from the specified table,
                          depending on the method parameter.

            Raises:
                Exception: If the query fails, the `retry` decorator will handle the retry logic
                           according to the specified retry policy.
        """
        self.validate_ts_parameters()

        end_time = self.db_client_session.query(f"""
                    SELECT {method}(Open_time) 
                    FROM {self.get_tablename()}
                """).result_rows[0][0]

        return end_time


    def get_where_condition(self):
        """
            Generates the SQL WHERE condition based on the specified interval.

            This method constructs a WHERE clause for an SQL query that filters
            records based on the `interval` attribute. The generated condition is
            used to determine the acceptable number of rows (`rows_amount`) in a
            database table, considering different time units such as hours, days,
            weeks, or months.

            The method handles different scenarios:
            - `1h`, `12h`, `1d`:
            - `3d`
            - `1w`, `1M`

            If the interval is not recognized, an exception is raised.

            Returns:
                str: The generated WHERE clause as a string.

            Raises:
                Exception: If the `interval` does not match any of the expected values,
                           an exception is thrown indicating an unexpected error.
        """
        row_num_coef = self.__valid_amount_of_rows[f'{self.interval}']

        # when the `the upper unit` is a day
        if self.interval in ['1h', '12h', '1d']:
            where_condition = f"""
                            WHERE   (rows_amount < {row_num_coef * 31} AND month_name in [1, 3, 5, 7, 8, 10, 12]) 
                                    OR 
                                    (rows_amount < {row_num_coef * 30} AND month_name in [4, 6, 9, 11])
                                    OR 
                                    (rows_amount < {row_num_coef * 29} AND month_name = 2 AND is_leap = 1)
                                    OR 
                                    (rows_amount < {row_num_coef * 28} AND month_name = 2 AND is_leap = 0)
                    """
        # when the `the upper unit` is a month (depends on February)
        elif self.interval == '3d':
            where_condition = f"""
                            WHERE   (rows_amount < {row_num_coef[0]} AND month_name = 2) 
                                    OR 
                                    (rows_amount < {row_num_coef[1]} AND month_name != 2)
                    """
        # when the `the upper unit` is a month
        elif self.interval in ['1w', '1M']:
            where_condition = f"""
                            WHERE   rows_amount < {row_num_coef}
                    """
        else:
            raise Exception("[Error] Unexpected error in get_where_condition process.")

        return where_condition


    def _build_get_records_count_request(self):
        """
            Build SQL query to get the current lacks in CH db
        """
        return f"""
                SELECT month_where_lacks,
                        '{self.currency}_{self.interval}' as "ts_id",
                        rows_amount
                FROM (
                    SELECT 
                        toStartOfMonth(Open_time) AS month_where_lacks,
                        MONTH(Open_time) AS month_name,
                        IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 28, 0,
                            IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 29, 1,
                                NULL 
                        )) AS is_leap,
                        count(*) AS rows_amount
                    FROM {self.get_tablename()}
                    WHERE Open_time < toStartOfMonth(today())
                            and Open_time >= (
                                SELECT toStartOfMonth(addMonths(MIN(Open_time), 1))
                                FROM {self.get_tablename()}
                            )
                    GROUP BY month_where_lacks, month_name, is_leap
                    ORDER BY rows_amount
                )
                {self.get_where_condition()}
            """


    @retry()
    def get_lacks(self):
        """
            Retrieve and return a DataFrame containing the current lacks in the ClickHouse database.

            This method constructs an SQL query to identify months with missing or incomplete data
            in the ClickHouse database for a given currency and interval. The SQL query is built
            using the `_build_get_records_count_request` method, which incorporates the `where_condition`
            directly within the query based on the specified interval.

            The function is decorated with `@retry()` to handle transient errors during execution,
            such as database connectivity issues, by automatically retrying the query a specified
            number of times before giving up.

            Returns:
                pd.DataFrame: A DataFrame containing the following columns:
                    - month_where_lacks: The starting date of the month, where data is lacking.
                    - interval: The interval (e.g., '1h', '12h') being analyzed.
                    - currency: The currency being analyzed.
                    - rows_amount: The number of rows of data available for the corresponding month.
        """
        get_records_count_request = self._build_get_records_count_request()

        ch_response_df = pd.DataFrame(
            self.db_client_session.query(get_records_count_request).result_rows,
            columns=['month_where_lacks', 'ts_id', 'rows_amount']
        )

        return ch_response_df


    @retry()
    def get_amount_of_rows_for_currency_and_interval(self):
        """
            Retrieve the count of rows in the specified table for the current currency and interval.

            This method constructs and executes an SQL query to count the number of rows in the
            table corresponding to the `currency` and `interval` attributes within the `database`.

            Returns:
                int: The count of rows in the table for the current currency and interval.
        """
        # Calculate the amount of rows for the currency and interval
        query_count_open_time = f"""
                            SELECT count(Open_time) 
                            FROM {self.get_tablename()}
                        """
        return self.db_client_session.query(query_count_open_time).result_rows[0][0]


    @retry()
    def define_doubles_in_tables(self):
        """
            Identify and retrieve duplicate records in the table for the current currency and interval.

            This method sends a query to the ClickHouse database to identify duplicate entries based on
            the `Open_time` column. It groups the data by `Open_time` and returns the records where
            duplicates exist (i.e., where the count of `Open_time` is greater than 1). The result is
            returned as a pandas DataFrame.

            Returns:
                pd.DataFrame: A DataFrame containing the duplicate records with the following columns:
                    - Open_time: The timestamp of the duplicated entries.
                    - count: The number of times the `Open_time` appears in the table.
        """
        # Get data from db
        ch_response = self.db_client_session.query(f"""
                                    SELECT Open_time, COUNT(*) AS count
                                    FROM {self.get_tablename()}
                                    GROUP BY Open_time
                                    HAVING count > 1
                                """)

        return pd.DataFrame(ch_response.result_rows, columns=ch_response.column_names)


    @retry()
    def remove_duplicates(self):
        """
            Remove duplicate records from the ClickHouse table and update the table with cleaned data.

            This method performs a series of operations to clean duplicate records from the table:

            1. Drop the existing table with duplicates: Calls `drop_table` with the suffix `_no_duplicates` to remove the table if it already exists.
            2. Create a new table for cleaned data: Calls `create_table` with the suffix `_no_duplicates` to create a new table structure without duplicates.
            3. Generate a query to insert cleaned data: Constructs and executes an `INSERT INTO` query to populate the new table with unique records, aggregated by `Open_time`.
            4. Rename tables: Renames the current table to an old version and the cleaned table to the original table name.
            5. Drop the old table: Calls `drop_table` with the suffix `_old` to remove the old table after renaming.

            The function ensures that the final table contains only unique records by grouping and aggregating data based on the `Open_time` column.

            Raises:
                Exception: If any of the table operations fail, the last exception is raised.
        """
        # Drop table
        self.drop_table(suffix='_no_duplicates')

        # Create table
        self.create_table(suffix='_no_duplicates')

        # Get column names dynamically from the schema
        table_columns = self.get_table_columns()

        # Generate column list for the SELECT statement
        columns = ",\n".join(
            [f"any({col_name}) as {col_name}" for col_name, _ in table_columns if col_name != 'Open_time'])

        # Insert table
        insert_tab_request = f"""
            INSERT INTO {self.get_tablename()}_no_duplicates
            SELECT
                Open_time,
                {columns}
            FROM {self.get_tablename()}
            GROUP BY 
                Open_time
            """

        self.db_client_session.query(insert_tab_request)

        # Rename tables
        rename_request = f"""
            RENAME TABLE 
                {self.get_tablename()} TO {self.get_tablename()}_old, 
                {self.get_tablename()}_no_duplicates TO {self.get_tablename()};
        """
        self.db_client_session.query(rename_request)

        # Drop table with suffix='_old'
        self.drop_table(suffix='_old')