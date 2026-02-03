"""
add the new table ts_preprocessing_properties - ORM
Date: 2025-05-03
"""

import json
import sys
from datetime import datetime

description = "add the new table ts_preprocessing_properties - ORM"
depends_on = "001"
auto_generated = True
schema_changes = {
    "ConfigKernel.model_quality": {
        "database": "ConfigKernel",
        "table_name": "model_quality",
        "is_new": 0,
        "added_columns": {},
        "removed_columns": {
            "maeNew": "Nullable(Float64)"
        },
        "changed_columns": {},
        "affected_tables": [
            "model_quality"
        ]
    },
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 1,
        "added_columns": {
            "ts_id": "String",
            "property_id": "String",
            "calculated_at": "DateTime",
            "is_active": "Nullable(Bool)",
            "is_stationary": "Nullable(Bool)",
            "adf_pvalue": "Nullable(Float64)",
            "kpss_pvalue": "Nullable(Float64)",
            "lag1_autocorrelation": "Nullable(Float64)",
            "rolling_mean_cv": "Nullable(Float64)",
            "rolling_std_cv": "Nullable(Float64)",
            "outlier_ratio": "Nullable(Float64)",
            "main_period": "Nullable(UInt32)",
            "period_candidates": "Nullable(String)",
            "estimator_params": "Nullable(String)",
            "acf_values": "Nullable(String)",
            "decomposition_method": "Nullable(String)",
            "decomposition_params": "Nullable(String)",
            "contamination_isolation_forest": "Nullable(Float64)",
            "contamination_lof": "Nullable(Float64)",
            "n_neighbors_lof": "Nullable(UInt32)",
            "outlier_detection_method": "Nullable(String)",
            "scaler_type": "Nullable(String)",
            "scaler_params": "Nullable(String)",
            "scaler_serialized": "Nullable(String)",
            "original_stats": "Nullable(String)",
            "scaled_stats": "Nullable(String)",
            "max_age_analyzer": "Nullable(UInt32)",
            "max_age_periodicity": "Nullable(UInt32)",
            "max_age_outlier": "Nullable(UInt32)",
            "max_age_scaling": "Nullable(UInt32)",
            "max_age_decomposition": "Nullable(UInt32)",
            "force_recalc_analyzer": "Nullable(Bool)",
            "force_recalc_periodicity": "Nullable(Bool)",
            "force_recalc_outlier": "Nullable(Bool)",
            "force_recalc_scaling": "Nullable(Bool)",
            "force_recalc_decomposition": "Nullable(Bool)"
        },
        "removed_columns": {},
        "changed_columns": {},
        "affected_tables": [
            "ts_preprocessing_properties"
        ]
    }
}

def upgrade(client_session):
    """Apply migration for ORM tables"""

    # Import necessary modules
    from pipeline.db_schema_manager.models.orm_schema_extractor import SchemaExtractor
    import pipeline.db_schema_manager.models.orm_models as orm_models
    from datetime import datetime
    import json
    import logging

    def register_schema_in_registry(client_session, schema_key, schema_definition):
        """
        Registers a schema in the schema registry

        Args:
            client_session: ClickHouse connection session
            schema_key (str): Schema key (e.g., 'database.table_name')
            schema_definition (list): Schema definition as a list of tuples [(column_name, type), ...]

        Returns:
            bool: True if registration succeeds, False on error
        """
        try:
            # Extract database name from schema_key
            registry_db = "InfraKernel"  # Database for schema registry

            # Check existence of registry tables
            check_tables_query = f"SELECT 1 FROM system.tables WHERE database = '{registry_db}' AND name = 'schemas' LIMIT 1"
            check_tables_result = client_session.query(check_tables_query)
            if not check_tables_result.result_rows:
                print(f"WARNING: Schema registry tables do not exist in {registry_db}")
                return False

            # Check if schema already exists in the registry
            check_schema_query = f"SELECT 1 FROM {registry_db}.schemas WHERE schema_id = '{schema_key}' LIMIT 1"
            check_result = client_session.query(check_schema_query)

            # Prepare data for insertion
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Ensure correct JSON serialization of schema
            # Schema should be a list of tuples [(column_name, type), ...]
            if isinstance(schema_definition, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in schema_definition):
                # Convert tuples to lists for proper serialization
                schema_list = [[col[0], col[1]] for col in schema_definition]
                schema_json = json.dumps(schema_list, ensure_ascii=False)
            else:
                print(f"ERROR: Incorrect schema definition format: {schema_definition}")
                return False

            if not check_result.result_rows:
                # Register new schema

                # Insert record into schemas table
                insert_schema_query = f'''
                INSERT INTO {registry_db}.schemas 
                (schema_id, current_version, created_at, updated_at)
                VALUES 
                ('{schema_key}', 1, '{current_time}', '{current_time}')
                '''
                client_session.query(insert_schema_query)

                # Insert record into schema_versions table
                insert_version_query = f'''
                INSERT INTO {registry_db}.schema_versions 
                (schema_id, version, schema_definition, changelog, created_at)
                VALUES 
                ('{schema_key}', 1, '{schema_json}', 'Initial schema created by migration', '{current_time}')
                '''
                client_session.query(insert_version_query)

                print(f"Schema registered for {schema_key} in the schema registry")
            else:
                print(f"Schema {schema_key} already exists in the registry")

            return True
        except Exception as e:
            print(f"ERROR registering schema {schema_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Take the list of exported names
    __all__ = orm_models.__all__

    # Dynamically inject each name into its namespace
    for name in __all__:
        globals()[name] = getattr(orm_models, name)

    # Create new table ConfigKernel.ts_preprocessing_properties
    print(f"Creating new table ConfigKernel.ts_preprocessing_properties")

    # Get ORM model
    model_class = None
    for name, cls in globals().items():
        if hasattr(cls, '__tablename__') and cls.__tablename__ == 'ts_preprocessing_properties':
            model_class = cls
            break

    if model_class:
        # Generate SQL for table creation
        create_table_sql = SchemaExtractor.generate_create_table_sql(model_class)
        client_session.query(create_table_sql)
        print(f"Created table ConfigKernel.ts_preprocessing_properties")

        # Register schema in the schema registry
        schema_definition = SchemaExtractor.extract_simple_schema(model_class)
        register_schema_in_registry(client_session, 'ConfigKernel.ts_preprocessing_properties', schema_definition)
    else:
        print(f"WARNING: Could not find ORM model for ts_preprocessing_properties")

    # Remove columns from ConfigKernel.model_quality table
    removed_columns = {'maeNew': 'Nullable(Float64)'}

    print(f"Removing columns {'maeNew'} from ConfigKernel.model_quality")
    for column_name in removed_columns.keys():
        alter_query = f"""
        ALTER TABLE ConfigKernel.model_quality 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.model_quality")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Drop table ConfigKernel.ts_preprocessing_properties
    print(f"Dropping table ConfigKernel.ts_preprocessing_properties")
    drop_table_sql = "DROP TABLE IF EXISTS ConfigKernel.ts_preprocessing_properties"
    client_session.query(drop_table_sql)
    print(f"Dropped table ConfigKernel.ts_preprocessing_properties")

    # Restore removed columns in ConfigKernel.model_quality table
    removed_columns = {'maeNew': 'Nullable(Float64)'}

    print(f"Restoring columns {'maeNew'} to ConfigKernel.model_quality")
    for column_name, column_type in removed_columns.items():
        # Determine default value based on data type
        default_value = None
        if 'int' in column_type.lower() or 'uint' in column_type.lower() or 'float' in column_type.lower():
            default_value = '0'
        elif 'string' in column_type.lower():
            default_value = "''"
        elif 'datetime' in column_type.lower():
            default_value = "'1970-01-01 00:00:00'"
        elif 'bool' in column_type.lower() or column_type.lower() == 'uint8':
            default_value = 'false'

        # Add column with default value if defined
        if default_value is not None:
            alter_query = f"""
            ALTER TABLE ConfigKernel.model_quality 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type} DEFAULT {default_value}
            """
        else:
            alter_query = f"""
            ALTER TABLE ConfigKernel.model_quality 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """

        client_session.query(alter_query)
        print(f"Restored column {column_name} to table ConfigKernel.model_quality")
