"""
new fields in table TSPreprocessingProperties for periodicity - ORM
Date: 2025-05-15
"""

import json
import sys
from datetime import datetime

description = "new fields in table TSPreprocessingProperties for periodicity - ORM"
depends_on = "004"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "periods": "Nullable(String)",
            "period_confidence_scores": "Nullable(String)",
            "periodicity_detection_method": "Nullable(String)",
            "stl_seasonal_strength": "Nullable(Float64)",
            "periodicity_method_results": "Nullable(String)"
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

    # Adding columns to the ConfigKernel.ts_preprocessing_properties table
    added_columns = {'periods': 'Nullable(String)', 'period_confidence_scores': 'Nullable(String)', 'periodicity_detection_method': 'Nullable(String)', 'stl_seasonal_strength': 'Nullable(Float64)', 'periodicity_method_results': 'Nullable(String)'}

    print(f"Adding columns {'periods', 'period_confidence_scores', 'periodicity_detection_method', 'stl_seasonal_strength', 'periodicity_method_results'} to ConfigKernel.ts_preprocessing_properties")
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

    # Remove added columns from ConfigKernel.ts_preprocessing_properties table
    columns_to_remove = ['periods', 'period_confidence_scores', 'periodicity_detection_method', 'stl_seasonal_strength', 'periodicity_method_results']

    print(f"Removing columns {'periods', 'period_confidence_scores', 'periodicity_detection_method', 'stl_seasonal_strength', 'periodicity_method_results'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")
