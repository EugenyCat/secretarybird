from pipeline.db_schema_manager.models.ts_schema_factories import (
    CryptoRawSchemaFactory,
    CryptoTransformedSchemaFactory
)
from pipeline.db_schema_manager.models.orm_schema_extractor import SchemaExtractor
import pipeline.db_schema_manager.models.orm_models as orm_module
import logging
import re


# todo: should we add a detector for database changes (moving from one to another), data type changes, something else?

# todo: how to avoid SQL queries and use ORM and Query Builder instead

class SchemaComparator:
    """
        Component for comparing schemas between code and the database.
        Detects added, removed, and changed columns.
    """

    # todo: needs to be registered every time, move to a single place, some unified config
    _supported_databases = ["'CRYPTO'"]  # Add other databases as needed

    # Create schema factory instances
    _factories = {
        'CRYPTO_raw': CryptoRawSchemaFactory(),
        'CRYPTO_transformed': CryptoTransformedSchemaFactory()
    }

    def compare_code_and_db_schemas(self, client_session):
        """
        Compares schemas in code and in the DB for all tables in all supported databases

        Args:
            client_session: ClickHouse connection session

        Returns:
            dict: Dictionary with schema differences
        """
        differences = {}

        # Get all tables from the database
        tables = self._get_all_database_tables(client_session)

        # Process raw tables
        for table_info in tables['raw']:
            database = table_info['database']
            table_name = table_info['table_name']
            currency = table_info['currency']
            interval = table_info['interval']

            # Build schema key and ts_id
            schema_key = f"{database}_raw"
            ts_id = f"{currency}_{interval}"

            # Skip if no matching factory
            if schema_key not in self._factories:
                logging.warning(f"Schema factory not found for {schema_key}, skipping table {table_name}")
                continue

            # Get current table schema from DB
            describe_query = f"""
            DESCRIBE {database}.{table_name}
            """
            result = client_session.query(describe_query)

            db_schema = [(row[0], row[1]) for row in result.result_rows]

            # Get schema from code via the corresponding factory
            factory = self._factories[schema_key]
            code_schema = factory.create_schema(ts_id, 'raw')

            # Compare schemas
            self._compare_and_update_differences(
                differences, schema_key, database, 'raw', '',
                table_name, db_schema, code_schema
            )

        # Process transformed tables
        for table_info in tables['transformed']:
            database = table_info['database']
            table_name = table_info['table_name']
            currency = table_info['currency']
            interval = table_info['interval']

            # Build schema key and ts_id
            schema_key = f"{database}_{interval}_transformed"
            ts_id = f"{currency}_{interval}"

            # For transformed, consider separate schemas for different intervals
            if schema_key not in self._factories and f"{database}_transformed" in self._factories:
                schema_key = f"{database}_transformed"

            # Skip if no matching factory
            if schema_key not in self._factories:
                logging.warning(f"Schema factory not found for {schema_key}, skipping table {table_name}")
                continue

            # Get current table schema from DB
            describe_query = f"""
            DESCRIBE {database}.{table_name}
            """
            result = client_session.query(describe_query)

            db_schema = [(row[0], row[1]) for row in result.result_rows]

            # Get schema from code via the corresponding factory
            factory = self._factories[schema_key]
            # For transformed schemas, pass only ts_id
            code_schema = factory.create_schema(ts_id)

            # Compare schemas
            self._compare_and_update_differences(
                differences, schema_key, database, 'transformed', interval,
                table_name, db_schema, code_schema
            )

        return differences

    def _compare_and_update_differences(self, differences, schema_key, database,
                                        processing_stage, interval, table_name,
                                        db_schema, code_schema):
        """
        Helper method for comparing schemas and updating the differences dictionary

        Args:
            differences (dict): Dictionary for accumulating differences
            schema_key (str): Schema key
            database (str): Database name
            processing_stage (str): Processing stage (raw/transformed)
            interval (str): Data interval
            table_name (str): Table name
            db_schema (list): Table schema in the DB
            code_schema (list): Table schema in the code
        """
        # Convert schemas to dictionaries for easier comparison
        db_columns = {col[0]: col[1] for col in db_schema}
        code_columns = {col[0]: col[1] for col in code_schema}

        # Find differences
        added_columns = {col: dtype for col, dtype in code_columns.items()
                         if col not in db_columns}

        changed_columns = {col: (db_columns[col], dtype) for col, dtype in code_columns.items()
                           if col in db_columns and db_columns[col] != dtype}

        removed_columns = {col: dtype for col, dtype in db_columns.items()
                           if col not in code_columns}

        # If there are differences, add them to the result
        if added_columns or changed_columns or removed_columns:
            if schema_key not in differences:
                differences[schema_key] = {
                    "database": database,
                    "processing_stage": processing_stage,
                    "interval": interval,
                    "is_new": 0,
                    "added_columns": added_columns,
                    "changed_columns": changed_columns,
                    "removed_columns": removed_columns,
                    "affected_tables": []
                }
            else:
                # Merge differences if the schema already exists in results
                differences[schema_key]["added_columns"].update(added_columns)
                differences[schema_key]["changed_columns"].update(changed_columns)
                differences[schema_key]["removed_columns"].update(removed_columns)

            # Add table to affected list if not already present
            if table_name not in differences[schema_key]["affected_tables"]:
                differences[schema_key]["affected_tables"].append(table_name)

    def schemas_are_different(self, schema1, schema2):
        """
        Checks whether two schemas are different

        Args:
            schema1 (list): First schema as a list of tuples
            schema2 (list): Second schema as a list of tuples

        Returns:
            bool: True if schemas are different, False if identical
        """
        try:
            # Check input data types for debugging
            logging.debug(f"Schema1 type: {type(schema1)}, Schema2 type: {type(schema2)}")

            # Convert to list of tuples if a dict is provided
            if isinstance(schema1, dict):
                schema1 = [(k, v) for k, v in schema1.items()]
            if isinstance(schema2, dict):
                schema2 = [(k, v) for k, v in schema2.items()]

            if len(schema1) != len(schema2):
                logging.debug(f"Different schema length: {len(schema1)} vs {len(schema2)}")
                return True

            # Convert to dictionaries for easy comparison
            dict1 = {col[0]: col[1] for col in schema1}
            dict2 = {col[0]: col[1] for col in schema2}

            # Compare columns
            keys1 = set(dict1.keys())
            keys2 = set(dict2.keys())
            if keys1 != keys2:
                logging.debug(f"Different columns: {keys1 - keys2} / {keys2 - keys1}")
                return True

            # Compare column types
            for key in dict1:
                if dict1[key] != dict2[key]:
                    logging.debug(f"Different types for {key}: {dict1[key]} vs {dict2[key]}")
                    return True

            logging.debug("Schemas are identical")
            return False
        except Exception as e:
            logging.error(f"Error while comparing schemas: {e}")
            # In case of error, treat schemas as different for safety
            return True

    def apply_schema_changes(self, current_schema, changes, is_upgrade=True):
        """
        Applies schema changes to the current schema

        Args:
            current_schema (list): Current schema as a list of tuples (column_name, data_type)
            changes (dict): Schema changes description
            is_upgrade (bool): True for upgrade migration, False for rollback

        Returns:
            list: Modified schema
        """
        try:
            # Convert to a unified format (list of tuples)
            if isinstance(current_schema, dict):
                current_schema = [(k, v) for k, v in current_schema.items()]

            # Convert list of tuples to dict for easier processing
            schema_dict = {col[0]: col[1] for col in current_schema}

            logging.debug(f"Current schema (before changes): {schema_dict}")
            logging.debug(f"Changes to apply: {changes}")

            if is_upgrade:
                # Apply changes during forward migration

                # Remove columns
                if "removed_columns" in changes and changes["removed_columns"]:
                    logging.debug(f"Removing columns: {list(changes['removed_columns'].keys())}")
                    for col in changes["removed_columns"]:
                        if col in schema_dict:
                            del schema_dict[col]

                # Add columns
                if "added_columns" in changes and changes["added_columns"]:
                    logging.debug(f"Adding columns: {list(changes['added_columns'].keys())}")
                    for col, dtype in changes["added_columns"].items():
                        schema_dict[col] = dtype

                # Change column types
                if "changed_columns" in changes and changes["changed_columns"]:
                    logging.debug(f"Changing column types: {list(changes['changed_columns'].keys())}")
                    for col, (old_type, new_type) in changes["changed_columns"].items():
                        if col in schema_dict:
                            schema_dict[col] = new_type
            else:
                # Apply changes during rollback migration

                # Restore removed columns
                if "removed_columns" in changes and changes["removed_columns"]:
                    logging.debug(f"Restoring removed columns: {list(changes['removed_columns'].keys())}")
                    for col, dtype in changes["removed_columns"].items():
                        schema_dict[col] = dtype

                # Remove added columns
                if "added_columns" in changes and changes["added_columns"]:
                    logging.debug(f"Removing added columns: {list(changes['added_columns'].keys())}")
                    for col in changes["added_columns"]:
                        if col in schema_dict:
                            del schema_dict[col]

                # Revert to old column types
                if "changed_columns" in changes and changes["changed_columns"]:
                    logging.debug(f"Reverting to old column types: {list(changes['changed_columns'].keys())}")
                    for col, (old_type, new_type) in changes["changed_columns"].items():
                        if col in schema_dict:
                            schema_dict[col] = old_type

            logging.debug(f"Modified schema (after changes): {schema_dict}")

            # Convert dict back to list of tuples
            return [(col, dtype) for col, dtype in schema_dict.items()]
        except Exception as e:
            logging.error(f"Error while applying schema changes: {e}")
            logging.exception("Detailed error information:")
            return current_schema  # Return original schema in case of error

    def _get_all_database_tables(self, client_session):
        """
        Retrieves all tables from the database and groups them by type (raw/transformed)

        Args:
            client_session: ClickHouse connection session

        Returns:
            dict: Dictionary with tables grouped by processing type
        """
        # Query all tables from supported databases
        query = f"""
        SELECT database, name 
        FROM system.tables 
        WHERE database IN ({', '.join(self._supported_databases)})
        """

        result = client_session.query(query)

        # Dictionary to store table information
        tables = {
            'raw': [],
            'transformed': []
        }

        # Regular expression to parse table name
        pattern = r'^(.+)_(.+)_(raw|transformed)$'

        for row in result.result_rows:
            database, table_name = row

            # Extract table information using regex
            match = re.match(pattern, table_name)
            if match:
                currency = match.group(1)
                interval = match.group(2)
                processing_stage = match.group(3)

                # Add table to the corresponding group
                table_info = {
                    'database': database,
                    'table_name': table_name,
                    'currency': currency,
                    'interval': interval
                }

                tables[processing_stage].append(table_info)

        logging.info(f"Found {len(tables['raw'])} raw tables and {len(tables['transformed'])} transformed tables")
        return tables

    def get_all_orm_models(self):
        """
        Retrieves all ORM models from the orm_models.py module

        Returns:
            dict: Dictionary {table_name: model_class}
        """
        try:
            # Use extractor to get all models
            return SchemaExtractor.get_all_orm_models(orm_module)
        except ImportError as e:
            logging.error(f"Error importing ORM models: {e}")
            return {}

    def compare_orm_models_with_db(self, orm_models, client_session):
        """
        Compares ORM model definitions with current schemas in the database

        Args:
            orm_models (dict): Dictionary {table_name: model_class}
            client_session: ClickHouse connection session

        Returns:
            dict: Dictionary with schema differences
        """
        differences = {}

        # Create InfraKernel database if it does not exist
        try:
            create_db_query = "CREATE DATABASE IF NOT EXISTS InfraKernel"
            client_session.query(create_db_query)
        except Exception as e:
            logging.warning(f"Failed to create InfraKernel database: {e}")

        # Create ConfigKernel database if it does not exist
        try:
            create_db_query = "CREATE DATABASE IF NOT EXISTS ConfigKernel"
            client_session.query(create_db_query)
        except Exception as e:
            logging.warning(f"Failed to create ConfigKernel database: {e}")

        # For each ORM model, check the existence of the corresponding table
        for table_name, model_class in orm_models.items():
            # Determine database from ORM model
            database = SchemaExtractor.get_database_for_model(model_class)
            schema_key = f"{database}.{table_name}"

            # Get schema definition from ORM model
            orm_schema = SchemaExtractor.extract_simple_schema(model_class)

            # Convert list of tuples to dict for easier comparison
            orm_columns = {col[0]: col[1] for col in orm_schema}

            # Initialize differences entry
            differences[schema_key] = {
                "database": database,
                "table_name": table_name,
                "is_new": 0,
                "added_columns": {},
                "removed_columns": {},
                "changed_columns": {},
                "affected_tables": [table_name]
            }

            # Check table existence
            check_table_query = f"""
            SELECT 1 FROM system.tables 
            WHERE database = '{database}' AND name = '{table_name}'
            """
            check_result = client_session.query(check_table_query)

            # If table does not exist, mark it as new
            if not check_result.result_rows:
                differences[schema_key]["is_new"] = 1
                differences[schema_key]["added_columns"] = orm_columns
                logging.info(f"Table {schema_key} does not exist in the database, marking as new")
                continue

            # Get current table schema from DB
            try:
                describe_query = f"DESCRIBE {database}.{table_name}"
                result = client_session.query(describe_query)
                db_schema = [(row[0], row[1]) for row in result.result_rows]
                db_columns = {col[0]: col[1] for col in db_schema}
            except Exception as e:
                logging.warning(f"Error retrieving schema for {schema_key}: {e}")
                continue

            # Find added columns (present in ORM, missing in DB)
            added_columns = {col: dtype for col, dtype in orm_columns.items() if col not in db_columns}

            # Find removed columns (present in DB, missing in ORM)
            removed_columns = {col: dtype for col, dtype in db_columns.items() if col not in orm_columns}

            # Dictionary of equivalent types for ClickHouse
            equivalent_types = {
                'Int32': ['UInt32'],  # Int32 is equivalent to UInt32 for migrations
                'UInt32': ['Int32'],
                'UInt8': ['Bool'],
                'Bool': ['UInt8'],
                'Nullable(UInt8)': ['Bool', 'UInt8'],
                'Nullable(DateTime)': ['DateTime'],
                'DateTime': ['Nullable(DateTime)'],
                'Nullable(String)': ['String'],
                'String': ['Nullable(String)'],
                # Other equivalent types as needed
            }

            # Function to check type equivalence
            def types_are_equivalent(type1, type2):
                if type1 == type2:
                    return True
                if type1 in equivalent_types and type2 in equivalent_types[type1]:
                    return True
                if type2 in equivalent_types and type1 in equivalent_types[type2]:
                    return True
                # Nullable check
                if type1.startswith('Nullable(') and type1[9:-1] == type2:
                    return True
                if type2.startswith('Nullable(') and type2[9:-1] == type1:
                    return True
                return False

            # Find changed column types
            changed_columns = {
                col: (db_columns[col], orm_columns[col])
                for col in set(orm_columns) & set(db_columns)
                if not types_are_equivalent(db_columns[col], orm_columns[col])
            }

            # Update differences entry
            differences[schema_key]["added_columns"] = added_columns
            differences[schema_key]["removed_columns"] = removed_columns
            differences[schema_key]["changed_columns"] = changed_columns

            # If there are no differences, remove the entry
            if not added_columns and not removed_columns and not changed_columns:
                del differences[schema_key]

        return differences
