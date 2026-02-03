"""
new fields tbats decomposition - ORM
Date: 2025-08-20
"""

import json
import sys
from datetime import datetime

description = "new fields tbats decomposition - ORM"
depends_on = "018"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "tbats_seasonal_periods": "Nullable(String)",
            "tbats_box_cox_lambda": "Nullable(Float64)",
            "tbats_aic": "Nullable(Float64)",
            "tbats_arma_order": "Nullable(String)",
            "tbats_damped_trend": "Nullable(Bool)"
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
    added_columns = {'tbats_seasonal_periods': 'Nullable(String)', 'tbats_box_cox_lambda': 'Nullable(Float64)', 'tbats_aic': 'Nullable(Float64)', 'tbats_arma_order': 'Nullable(String)', 'tbats_damped_trend': 'Nullable(Bool)'}

    print(f"Adding columns {'tbats_seasonal_periods', 'tbats_box_cox_lambda', 'tbats_aic', 'tbats_arma_order', 'tbats_damped_trend'} to ConfigKernel.ts_preprocessing_properties")
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
    columns_to_remove = ['tbats_seasonal_periods', 'tbats_box_cox_lambda', 'tbats_aic', 'tbats_arma_order', 'tbats_damped_trend']

    print(f"Removing columns {'tbats_seasonal_periods', 'tbats_box_cox_lambda', 'tbats_aic', 'tbats_arma_order', 'tbats_damped_trend'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")
