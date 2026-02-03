"""
remove field periodicity_estimator_params from table TSPreprocessingProperties - ORM
Date: 2025-05-16
"""

import json
import sys
from datetime import datetime

description = "remove field periodicity_estimator_params from table TSPreprocessingProperties - ORM"
depends_on = "006"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {},
        "removed_columns": {
            "periodicity_estimator_params": "Nullable(String)"
        },
        "changed_columns": {},
        "affected_tables": [
            "ts_preprocessing_properties"
        ]
    }
}

def upgrade(client_session):
    """Apply migration for ORM tables"""

    # Remove columns from ConfigKernel.ts_preprocessing_properties table
    removed_columns = {'periodicity_estimator_params': 'Nullable(String)'}

    print(f"Removing columns {'periodicity_estimator_params'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in removed_columns.keys():
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Restore removed columns in ConfigKernel.ts_preprocessing_properties table
    removed_columns = {'periodicity_estimator_params': 'Nullable(String)'}

    print(f"Restoring columns {'periodicity_estimator_params'} to ConfigKernel.ts_preprocessing_properties")
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
            ALTER TABLE ConfigKernel.ts_preprocessing_properties 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type} DEFAULT {default_value}
            """
        else:
            alter_query = f"""
            ALTER TABLE ConfigKernel.ts_preprocessing_properties 
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """

        client_session.query(alter_query)
        print(f"Restored column {column_name} to table ConfigKernel.ts_preprocessing_properties")
