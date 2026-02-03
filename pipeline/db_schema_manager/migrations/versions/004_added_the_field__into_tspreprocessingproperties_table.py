"""
added the field  into TSPreprocessingProperties table
Date: 2025-05-09
"""

import json
import sys
from datetime import datetime

description = "added the field  into TSPreprocessingProperties table"
depends_on = "003"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "is_combined": "UInt32"
        },
        "removed_columns": {},
        "changed_columns": {
            "is_active": [
                "UInt8",
                "UInt32"
            ],
            "force_recalc_analyzer": [
                "UInt8",
                "UInt32"
            ],
            "is_stationary": [
                "UInt8",
                "UInt32"
            ],
            "force_recalc_outlier": [
                "UInt8",
                "UInt32"
            ],
            "force_recalc_scaling": [
                "UInt8",
                "UInt32"
            ],
            "force_recalc_periodicity": [
                "UInt8",
                "UInt32"
            ],
            "force_recalc_decomposition": [
                "UInt8",
                "UInt32"
            ]
        },
        "affected_tables": [
            "ts_preprocessing_properties"
        ]
    },
    "ConfigKernel.time_series_definition": {
        "database": "ConfigKernel",
        "table_name": "time_series_definition",
        "is_new": 0,
        "added_columns": {},
        "removed_columns": {},
        "changed_columns": {
            "is_active": [
                "UInt8",
                "UInt32"
            ]
        },
        "affected_tables": [
            "time_series_definition"
        ]
    }
}

def upgrade(client_session):
    """Apply migration for ORM tables"""

    # Adding columns to the ConfigKernel.ts_preprocessing_properties table
    added_columns = {'is_combined': 'UInt32'}

    print(f"Adding columns {'is_combined'} to ConfigKernel.ts_preprocessing_properties")
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

    # Change type of column is_active in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column is_active from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN is_active UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column is_active in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column force_recalc_analyzer in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column force_recalc_analyzer from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_analyzer UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column force_recalc_analyzer in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column is_stationary in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column is_stationary from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN is_stationary UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column is_stationary in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column force_recalc_outlier in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column force_recalc_outlier from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_outlier UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column force_recalc_outlier in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column force_recalc_scaling in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column force_recalc_scaling from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_scaling UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column force_recalc_scaling in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column force_recalc_periodicity in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column force_recalc_periodicity from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_periodicity UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column force_recalc_periodicity in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column force_recalc_decomposition in ConfigKernel.ts_preprocessing_properties table
    print(f"Changing type of column force_recalc_decomposition from UInt8 to UInt32 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_decomposition UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column force_recalc_decomposition in table ConfigKernel.ts_preprocessing_properties")

    # Change type of column is_active in ConfigKernel.time_series_definition table
    print(f"Changing type of column is_active from UInt8 to UInt32 in ConfigKernel.time_series_definition")
    alter_query = """
    ALTER TABLE ConfigKernel.time_series_definition 
    MODIFY COLUMN is_active UInt32
    """
    client_session.query(alter_query)
    print(f"Changed type of column is_active in table ConfigKernel.time_series_definition")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Remove added columns from ConfigKernel.ts_preprocessing_properties table
    columns_to_remove = ['is_combined']

    print(f"Removing columns {'is_combined'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column is_active in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column is_active from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN is_active UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column is_active in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column force_recalc_analyzer in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column force_recalc_analyzer from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_analyzer UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column force_recalc_analyzer in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column is_stationary in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column is_stationary from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN is_stationary UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column is_stationary in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column force_recalc_outlier in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column force_recalc_outlier from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_outlier UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column force_recalc_outlier in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column force_recalc_scaling in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column force_recalc_scaling from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_scaling UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column force_recalc_scaling in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column force_recalc_periodicity in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column force_recalc_periodicity from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_periodicity UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column force_recalc_periodicity in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column force_recalc_decomposition in ConfigKernel.ts_preprocessing_properties table
    print(f"Reverting type of column force_recalc_decomposition from UInt32 to UInt8 in ConfigKernel.ts_preprocessing_properties")
    alter_query = """
    ALTER TABLE ConfigKernel.ts_preprocessing_properties 
    MODIFY COLUMN force_recalc_decomposition UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column force_recalc_decomposition in table ConfigKernel.ts_preprocessing_properties")

    # Revert type of column is_active in ConfigKernel.time_series_definition table
    print(f"Reverting type of column is_active from UInt32 to UInt8 in ConfigKernel.time_series_definition")
    alter_query = """
    ALTER TABLE ConfigKernel.time_series_definition 
    MODIFY COLUMN is_active UInt8
    """
    client_session.query(alter_query)
    print(f"Reverted type of column is_active in table ConfigKernel.time_series_definition")
