"""
TimeSeriesDefinition new field reference_asset_id - ORM
Date: 2025-09-10
"""

import json
import sys
from datetime import datetime

description = "TimeSeriesDefinition new field reference_asset_id - ORM"
depends_on = "021"
auto_generated = True
schema_changes = {
    "ConfigKernel.time_series_definition": {
        "database": "ConfigKernel",
        "table_name": "time_series_definition",
        "is_new": 0,
        "added_columns": {
            "reference_asset_id": "Nullable(String)"
        },
        "removed_columns": {},
        "changed_columns": {},
        "affected_tables": [
            "time_series_definition"
        ]
    }
}

def upgrade(client_session):
    """Apply migration for ORM tables"""

    # Adding columns to ConfigKernel.time_series_definition table
    added_columns = {'reference_asset_id': 'Nullable(String)'}

    print(f"Adding columns {'reference_asset_id'} to ConfigKernel.time_series_definition")
    for column_name, column_type in added_columns.items():
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
            ALTER TABLE ConfigKernel.time_series_definition 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type} DEFAULT {default_value}
            """
        else:
            alter_query = f"""
            ALTER TABLE ConfigKernel.time_series_definition 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """

        client_session.query(alter_query)
        print(f"Added column {column_name} to table ConfigKernel.time_series_definition")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Remove added columns from ConfigKernel.time_series_definition table
    columns_to_remove = ['reference_asset_id']

    print(f"Removing columns {'reference_asset_id'} from ConfigKernel.time_series_definition")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.time_series_definition 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.time_series_definition")
