"""
Added the new fields in decomposition section - ORM
Date: 2025-08-08
"""

import json
import sys
from datetime import datetime

description = "Added the new fields in decomposition section - ORM"
depends_on = "014"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "estimated_trend_strength": "Nullable(Float64)",
            "baseline_quality": "Nullable(Float64)",
            "model_type": "Nullable(String)",
            "decomposition_confidence_threshold": "Nullable(Float64)",
            "seasonal_strength": "Nullable(Float64)",
            "residual_strength": "Nullable(Float64)",
            "quality_score": "Nullable(Float64)",
            "reconstruction_error": "Nullable(Float64)",
            "mstl_stability_metrics_converged": "Nullable(UInt32)",
            "mstl_corrections_applied": "Nullable(String)"
        },
        "removed_columns": {
            "decomposition_params": "Nullable(String)"
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
    added_columns = {'estimated_trend_strength': 'Nullable(Float64)', 'baseline_quality': 'Nullable(Float64)', 'model_type': 'Nullable(String)', 'decomposition_confidence_threshold': 'Nullable(Float64)', 'seasonal_strength': 'Nullable(Float64)', 'residual_strength': 'Nullable(Float64)', 'quality_score': 'Nullable(Float64)', 'reconstruction_error': 'Nullable(Float64)', 'mstl_stability_metrics_converged': 'Nullable(UInt32)', 'mstl_corrections_applied': 'Nullable(String)'}

    print(f"Adding columns {'estimated_trend_strength', 'baseline_quality', 'model_type', 'decomposition_confidence_threshold', 'seasonal_strength', 'residual_strength', 'quality_score', 'reconstruction_error', 'mstl_stability_metrics_converged', 'mstl_corrections_applied'} to ConfigKernel.ts_preprocessing_properties")
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
    removed_columns = {'decomposition_params': 'Nullable(String)'}

    print(f"Removing columns {'decomposition_params'} from ConfigKernel.ts_preprocessing_properties")
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
    columns_to_remove = ['estimated_trend_strength', 'baseline_quality', 'model_type', 'decomposition_confidence_threshold', 'seasonal_strength', 'residual_strength', 'quality_score', 'reconstruction_error', 'mstl_stability_metrics_converged', 'mstl_corrections_applied']

    print(f"Removing columns {'estimated_trend_strength', 'baseline_quality', 'model_type', 'decomposition_confidence_threshold', 'seasonal_strength', 'residual_strength', 'quality_score', 'reconstruction_error', 'mstl_stability_metrics_converged', 'mstl_corrections_applied'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")

    # Restoring removed columns to ConfigKernel.ts_preprocessing_properties table
    removed_columns = {'decomposition_params': 'Nullable(String)'}

    print(f"Restoring columns {'decomposition_params'} to ConfigKernel.ts_preprocessing_properties")
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
