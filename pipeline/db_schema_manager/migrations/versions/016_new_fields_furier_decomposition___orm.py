"""
new fields Fourier decomposition - ORM
Date: 2025-08-20
"""

import json
import sys
from datetime import datetime

description = "new fields Fourier decomposition - ORM"
depends_on = "015"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "fourier_n_harmonics": "Nullable(UInt32)",
            "fourier_spectral_entropy": "Nullable(Float64)",
            "fourier_harmonic_strength": "Nullable(Float64)"
        },
        "removed_columns": {
            "decomposition_confidence_threshold": "Nullable(Float64)"
        },
        "changed_columns": {},
        "affected_tables": [
            "ts_preprocessing_properties"
        ]
    }
}

def upgrade(client_session):
    """Apply migration for ORM tables"""

    # Adding columns to ConfigKernel.ts_preprocessing_properties table
    added_columns = {'fourier_n_harmonics': 'Nullable(UInt32)', 'fourier_spectral_entropy': 'Nullable(Float64)', 'fourier_harmonic_strength': 'Nullable(Float64)'}

    print(f"Adding columns {'fourier_n_harmonics', 'fourier_spectral_entropy', 'fourier_harmonic_strength'} to ConfigKernel.ts_preprocessing_properties")
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

    # Removing columns from ConfigKernel.ts_preprocessing_properties table
    removed_columns = {'decomposition_confidence_threshold': 'Nullable(Float64)'}

    print(f"Removing columns {'decomposition_confidence_threshold'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in removed_columns.keys():
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")


def downgrade(client_session):
    """Rollback migration for ORM tables"""

    # Removing added columns from ConfigKernel.ts_preprocessing_properties table
    columns_to_remove = ['fourier_n_harmonics', 'fourier_spectral_entropy', 'fourier_harmonic_strength']

    print(f"Removing columns {'fourier_n_harmonics', 'fourier_spectral_entropy', 'fourier_harmonic_strength'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")

    # Restoring removed columns to ConfigKernel.ts_preprocessing_properties table
    removed_columns = {'decomposition_confidence_threshold': 'Nullable(Float64)'}

    print(f"Restoring columns {'decomposition_confidence_threshold'} to ConfigKernel.ts_preprocessing_properties")
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
