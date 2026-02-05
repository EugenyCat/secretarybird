"""
new fields ssa decomposition 2 - ORM
Date: 2025-08-20
"""

import json
import sys
from datetime import datetime

description = "new fields ssa decomposition 2 - ORM"
depends_on = "017"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "ssa_component_grouping": "Nullable(String)"
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

    # Adding columns to ConfigKernel.ts_preprocessing_properties table
    added_columns = {'ssa_component_grouping': 'Nullable(String)'}

    print(f"Adding columns {'ssa_component_grouping'} to ConfigKernel.ts_preprocessing_properties")
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
            ALTER TABLE ConfigKernel.ts_preprocessing_properties 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type} DEFAULT {default_value}
            """
        else:
            alter_query = f"""
            ALTER TABLE ConfigKernel.ts_preprocessing_properties 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """

        client_session.query(alter_query)
        print(f"Added column {column_name} to table ConfigKernel.ts_preprocessing_properties")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Removing added columns from ConfigKernel.ts_preprocessing_properties table
    columns_to_remove = ['ssa_component_grouping']

    print(f"Removing columns {'ssa_component_grouping'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")
