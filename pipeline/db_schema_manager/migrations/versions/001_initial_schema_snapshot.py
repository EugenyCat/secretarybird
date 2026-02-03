"""
    Initial schema snapshot
    Date: 2025-04-18
    """

from datetime import datetime
import json
import re
import sys
import importlib

description = "Initial schema snapshot"
depends_on = None

def get_all_database_tables(client_session):
    """
    Retrieves all tables from the database and groups them by type (raw/transformed)
    """
    # Query all tables from supported databases
    supported_databases = ["'CRYPTO'"]
    query = f"""
    SELECT database, name 
    FROM system.tables 
    WHERE database IN ({', '.join(supported_databases)})
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

        # Extract table information using regular expression
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

    print(f"Found {len(tables['raw'])} raw tables and {len(tables['transformed'])} transformed tables")
    return tables

def get_table_schema(client_session, database, table_name):
    """
    Retrieves the table schema from the database
    """
    describe_query = f"""
    DESCRIBE {database}.{table_name}
    """
    result = client_session.query(describe_query)

    # Convert result to list of tuples
    schema = [(row[0], row[1]) for row in result.result_rows]

    return schema

def register_schema(client_session, schema_id, schema_definition, changelog="Initial schema"):
    """
    Registers the schema in the registry
    """
    # Convert schema definition to JSON string
    schema_json = json.dumps(schema_definition)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if a schema with this ID already exists
    query = f"""
    SELECT current_version FROM InfraKernel.schemas 
    WHERE schema_id = '{schema_id}'
    """

    result = client_session.query(query)

    if not result.result_rows:
        # Create a new schema record and the first version
        insert_schema = f"""
        INSERT INTO InfraKernel.schemas 
        (schema_id, current_version, created_at, updated_at)
        VALUES 
        ('{schema_id}', 1, '{current_time}', '{current_time}')
        """

        insert_version = f"""
        INSERT INTO InfraKernel.schema_versions 
        (schema_id, version, schema_definition, changelog, created_at)
        VALUES 
        ('{schema_id}', 1, '{schema_json}', '{changelog}', '{current_time}')
        """

        client_session.query(insert_schema)
        client_session.query(insert_version)

        print(f"Created new schema: {schema_id}")
    else:
        # Get current version
        current_version = result.result_rows[0][0]
        new_version = current_version + 1

        # Create a new schema version
        insert_version = f"""
        INSERT INTO InfraKernel.schema_versions 
        (schema_id, version, schema_definition, changelog, created_at)
        VALUES 
        ('{schema_id}', {new_version}, '{schema_json}', '{changelog}', '{current_time}')
        """

        # Update current version in the schemas table
        update_schema = f"""
        UPDATE InfraKernel.schemas 
        SET current_version = {new_version}, updated_at = '{current_time}' 
        WHERE schema_id = '{schema_id}'
        """

        client_session.query(insert_version)
        client_session.query(update_schema)

        print(f"Updated schema: {schema_id} (version {new_version})")

def register_all_current_schemas(client_session):
    """
    Registers all current schemas in the registry
    """
    # Retrieve all tables
    tables = get_all_database_tables(client_session)

    # Register raw table schemas
    # Group tables by database to create shared schemas
    raw_schemas = {}
    for table_info in tables['raw']:
        database = table_info['database']
        schema_id = f"{database}_raw"

        if schema_id not in raw_schemas:
            raw_schemas[schema_id] = {
                'database': database,
                'table': table_info['table_name']
            }

    for schema_id, info in raw_schemas.items():
        # Get schema of the first table for this database and type
        schema = get_table_schema(client_session, info['database'], info['table'])

        # Register the schema
        register_schema(client_session, schema_id, schema, f"Initial raw schema for {info['database']}")

    # Register transformed table schemas
    # Group by database and interval
    transformed_schemas = {}
    for table_info in tables['transformed']:
        database = table_info['database']
        interval = table_info['interval']
        schema_id = f"{database}_{interval}_transformed"

        if schema_id not in transformed_schemas:
            transformed_schemas[schema_id] = {
                'database': database,
                'interval': interval,
                'table': table_info['table_name']
            }

    for schema_id, info in transformed_schemas.items():
        # Get schema of the first table for this database, interval, and type
        schema = get_table_schema(client_session, info['database'], info['table'])

        # Register the schema
        register_schema(client_session, schema_id, schema, f"Initial transformed schema for {info['database']} {info['interval']}")

def create_orm_tables(client_session):
    """
    Creates tables for ORM models in the InfraKernel database
    """
    print("Creating tables for ORM models...")

    # Create databases
    client_session.query("CREATE DATABASE IF NOT EXISTS InfraKernel")
    client_session.query("CREATE DATABASE IF NOT EXISTS ConfigKernel")

    # Correct import of ORM models
    from pipeline.db_schema_manager.models.orm_schema_extractor import SchemaExtractor
    import pipeline.db_schema_manager.models.orm_models as orm_module

    # Use the list of models from __all__
    __all__ = orm_module.__all__
    print(f"Imported ORM models: {__all__}")

    # Dynamically import each model into the current namespace
    for name in __all__:
        globals()[name] = getattr(orm_module, name)

    # Retrieve all ORM models
    orm_models_dict = {}
    for name in __all__:
        model_class = globals()[name]
        if hasattr(model_class, '__tablename__'):
            orm_models_dict[model_class.__tablename__] = model_class
            print(f"Added model: {name} -> {model_class.__tablename__}")

    # Create tables
    for table_name, model_class in orm_models_dict.items():
        database = getattr(model_class, "__clickhouse_database__", "InfraKernel")
        print(f"Creating table {database}.{table_name}...")

        try:
            # Generate SQL for table creation
            create_table_sql = SchemaExtractor.generate_create_table_sql(model_class)

            # Create the table
            client_session.query(create_table_sql)
            print(f"Created table {database}.{table_name}")

            # Register the table schema
            schema_id = f"{database}.{table_name}"
            schema_definition = SchemaExtractor.extract_simple_schema(model_class)
            register_schema(client_session, schema_id, schema_definition, f"Initial schema for {table_name}")

        except Exception as e:
            # On error, continue creating other tables
            print(f"Error creating table {table_name}: {e}")
            import traceback
            traceback.print_exc()

    print("ORM table creation completed")

def upgrade(client_session):
    """
    Apply migration: creates schema registry tables,
    ORM tables, and registers all current schemas
    """
    print("Initializing schema tables, ORM tables, and registering existing schemas...")

    # First, create ORM tables (including schemas and schema_versions tables)
    create_orm_tables(client_session)

    # Then, register time series schemas
    register_all_current_schemas(client_session)

    print("All schema initialization completed successfully.")

def downgrade(client_session):
    """
    Rollback migration: deletes all schema registry tables and ORM tables
    """
    print("WARNING: This will delete all schema registry data and ORM tables!")

    # List of all ORM tables
    orm_tables = [
        "migrations",
        "schema_versions",
        "schemas",
        "time_series_definition",
        "models",
        "model_training",
        "model_quality"
    ]

    # Drop ORM tables
    for table in orm_tables:
        drop_query = f"DROP TABLE IF EXISTS InfraKernel.{table}"
        try:
            client_session.query(drop_query)
            print(f"Dropped table InfraKernel.{table}")
        except Exception as e:
            print(f"Error dropping table {table}: {e}")

    print("All schema registry tables and ORM tables have been deleted.")
