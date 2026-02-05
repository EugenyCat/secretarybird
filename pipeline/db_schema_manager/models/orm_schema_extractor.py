import inspect
import logging
from sqlalchemy import Column, inspect as sa_inspect
from sqlalchemy.ext.declarative import DeclarativeMeta


class SchemaExtractor:
    """
    Extracts schema information from SQLAlchemy ORM models for use
    in the ClickHouse schema migration system.

    The class provides methods for converting ORM models to ClickHouse schema definitions,
    generating SQL for table creation, and retrieving the list of models.
    """

    # Mapping of SQLAlchemy types to ClickHouse types
    TYPE_MAPPING = {
        'Integer': 'UInt32',
        'BigInteger': 'UInt64',
        'SmallInteger': 'UInt16',
        'Float': 'Float64',
        'Numeric': 'Decimal',
        'String': 'String',
        'Text': 'String',
        'Boolean': 'Bool',
        'DateTime': 'DateTime',
        'Date': 'Date',
        'TIMESTAMP': 'DateTime',
        'Enum': 'Enum8',
        'JSON': 'String',
        'JSONB': 'String',
        'UUID': 'UUID',
        'ARRAY': 'Array',
    }

    @classmethod
    def get_database_for_model(cls, model_class):
        database = "ConfigKernel"  # default value
        if hasattr(model_class, '__table_args__'):
            table_args = model_class.__table_args__
            if isinstance(table_args, dict) and 'schema' in table_args:
                database = table_args['schema']
        return database

    @classmethod
    def extract_schema(cls, model_class):
        """
        Extracts the complete schema definition from an ORM model class

        Args:
            model_class: SQLAlchemy model class

        Returns:
            dict: Dictionary with complete schema definition for ClickHouse
        """
        if not isinstance(model_class, DeclarativeMeta):
            raise TypeError(f"Expected SQLAlchemy model class, got {type(model_class)}")

        # Get database from __table_args__ (standard SQLAlchemy)
        database = None
        if hasattr(model_class, '__table_args__'):
            table_args = model_class.__table_args__
            if isinstance(table_args, dict) and 'schema' in table_args:
                database = table_args['schema']

        # If not specified, use the default value
        if database is None:
            database = 'InfraKernel'  # Default value

        schema = {
            'table_name': model_class.__tablename__,
            'database': database,
            'columns': [],
            'primary_key': [],
            'engine': 'MergeTree()',  # Default engine
            'order_by': [],
            'partition_by': None,
            'settings': {}
        }

        # Extracting column information and their definitions
        for column in model_class.__table__.columns:
            column_name = column.name
            column_type = cls._map_type(column.type)
            nullable = 'Nullable(' + column_type + ')' if column.nullable else column_type

            # Processing default values
            default = None
            if column.default:
                if column.default.is_scalar:
                    # For scalar values
                    default_val = column.default.arg
                    if isinstance(default_val, bool):
                        default = str(int(default_val))
                    elif isinstance(default_val, (int, float)):
                        default = str(default_val)
                    elif isinstance(default_val, str):
                        default = f"'{default_val}'"
                    else:
                        # For other types, we won't specify DEFAULT for now
                        default = None

            schema['columns'].append({
                'name': column_name,
                'type': nullable,
                'default': default
            })

            # Check if the column is part of the primary key
            if column.primary_key:
                schema['primary_key'].append(column_name)

        # Extracting ClickHouse-specific attributes from the model class
        if hasattr(model_class, '__clickhouse_engine__'):
            schema['engine'] = model_class.__clickhouse_engine__

        # Defining order_by
        if hasattr(model_class, '__clickhouse_order_by__') and model_class.__clickhouse_order_by__:
            # Use explicitly defined order_by if it exists
            schema['order_by'] = model_class.__clickhouse_order_by__
        elif schema['primary_key']:
            # Otherwise use primary_key
            schema['order_by'] = schema['primary_key']
        else:
            # If there is neither order_by nor primary_key, use an empty list
            schema['order_by'] = []

        if hasattr(model_class, '__clickhouse_partition_by__'):
            schema['partition_by'] = model_class.__clickhouse_partition_by__

        if hasattr(model_class, '__clickhouse_settings__'):
            schema['settings'] = model_class.__clickhouse_settings__

        return schema

    @classmethod
    def _map_type(cls, sqlalchemy_type):
        """
        Converts SQLAlchemy type to ClickHouse type

        Args:
            sqlalchemy_type: SQLAlchemy data type

        Returns:
            str: Corresponding ClickHouse type
        """
        type_name = sqlalchemy_type.__class__.__name__

        if type_name in cls.TYPE_MAPPING:
            clickhouse_type = cls.TYPE_MAPPING[type_name]

            # Processing types with parameters (e.g., String(255))
            if hasattr(sqlalchemy_type, 'length') and sqlalchemy_type.length:
                if type_name == 'String':
                    return f"{clickhouse_type}({sqlalchemy_type.length})"

            # Processing Enum
            if type_name == 'Enum':
                values = [f"'{v}'" for v in sqlalchemy_type.enums]
                return f"{clickhouse_type}({', '.join(values)})"

            return clickhouse_type

        # For unknown types, return String
        logging.warning(f"Unknown SQLAlchemy type: {type_name}, using String as fallback")
        return 'String'

    @classmethod
    def extract_simple_schema(cls, model_class):
        """
        Extracts a simplified schema as a list of tuples (column_name, type)

        Args:
            model_class: SQLAlchemy model class

        Returns:
            list: List of tuples [(column_name, type), ...]
        """
        try:
            schema = cls.extract_schema(model_class)
            return [(col['name'], col['type']) for col in schema['columns']]
        except Exception as e:
            logging.error(f"Error extracting simple schema from {model_class.__name__}: {e}")
            return []

    @classmethod
    def generate_create_table_sql(cls, model_class):
        """
        Generates CREATE TABLE SQL query for an ORM model
        Returns the query without CREATE DATABASE for use with ClickHouse HTTP API

        Args:
            model_class: SQLAlchemy model class

        Returns:
            str: SQL query for creating the table
        """
        try:
            schema = cls.extract_schema(model_class)

            # Building column definitions
            columns_sql = []
            for column in schema['columns']:
                column_def = f"{column['name']} {column['type']}"
                if column['default'] is not None:
                    column_def += f" DEFAULT {column['default']}"
                columns_sql.append(column_def)

            # Building PRIMARY KEY
            primary_key_clause = ""
            if schema['primary_key']:
                primary_key_clause = f"PRIMARY KEY ({', '.join(schema['primary_key'])})\n"

            # Building ORDER BY
            order_by_clause = ""
            if schema['order_by']:
                order_by_clause = f"ORDER BY ({', '.join(schema['order_by'])})\n"
            else:
                # If there is no ORDER BY but there is PRIMARY KEY, use it
                if schema['primary_key']:
                    order_by_clause = f"ORDER BY ({', '.join(schema['primary_key'])})\n"
                else:
                    # If there is neither ORDER BY nor PRIMARY KEY, use tuple()
                    order_by_clause = "ORDER BY tuple()\n"

            # Building PARTITION BY
            partition_by_clause = ""
            if schema['partition_by']:
                partition_by_clause = f"PARTITION BY {schema['partition_by']}\n"

            # Building SETTINGS
            settings_clause = ""
            if schema['settings']:
                settings_parts = [f"{k}={v}" for k, v in schema['settings'].items()]
                settings_clause = f"SETTINGS {', '.join(settings_parts)}"

            # Assembling the complete SQL query for table creation WITHOUT creating the database
            columns_def = ',\n    '.join(columns_sql)
            sql = (
                f"CREATE TABLE IF NOT EXISTS {schema['database']}.{schema['table_name']} (\n"
                f"    {columns_def}\n"
                f") ENGINE = {schema['engine']}\n"
                f"{primary_key_clause}"
                f"{order_by_clause}"
                f"{partition_by_clause}"
                f"{settings_clause}"
            )

            return sql.strip()
        except Exception as e:
            logging.error(f"Error generating CREATE TABLE SQL for {model_class.__name__}: {e}")
            raise

    @classmethod
    def get_all_orm_models(cls, module):
        """
        Gets all ORM models from the specified module

        Args:
            module: Module containing ORM models

        Returns:
            dict: Dictionary {table_name: model_class}
        """
        models = {}

        try:
            for name, obj in inspect.getmembers(module):
                if isinstance(obj, DeclarativeMeta) and hasattr(obj, '__tablename__'):
                    models[obj.__tablename__] = obj

            return models
        except Exception as e:
            logging.error(f"Error getting ORM models from module {module.__name__}: {e}")
            return models