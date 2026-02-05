"""
Add catch22 features to TSPreprocessingProperties table
Date: 2025-07-14
"""

import json
import sys
from datetime import datetime

description = "Add catch22 features to TSPreprocessingProperties table"
depends_on = "013"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {
            "c22_mode_5": "Nullable(Float64)",
            "c22_mode_10": "Nullable(Float64)",
            "c22_outlier_timing_pos": "Nullable(Float64)",
            "c22_outlier_timing_neg": "Nullable(Float64)",
            "c22_acf_timescale": "Nullable(Float64)",
            "c22_acf_first_min": "Nullable(Float64)",
            "c22_low_freq_power": "Nullable(Float64)",
            "c22_centroid_freq": "Nullable(Float64)",
            "c22_ami_timescale": "Nullable(Float64)",
            "c22_periodicity": "Nullable(Float64)",
            "c22_ami2": "Nullable(Float64)",
            "c22_trev": "Nullable(Float64)",
            "c22_stretch_high": "Nullable(Float64)",
            "c22_stretch_decreasing": "Nullable(Float64)",
            "c22_entropy_pairs": "Nullable(Float64)",
            "c22_transition_variance": "Nullable(Float64)",
            "c22_whiten_timescale": "Nullable(Float64)",
            "c22_high_fluctuation": "Nullable(Float64)",
            "c22_forecast_error": "Nullable(Float64)",
            "c22_rs_range": "Nullable(Float64)",
            "c22_dfa": "Nullable(Float64)",
            "c22_embedding_dist": "Nullable(Float64)"
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
    added_columns = {'c22_mode_5': 'Nullable(Float64)', 'c22_mode_10': 'Nullable(Float64)', 'c22_outlier_timing_pos': 'Nullable(Float64)', 'c22_outlier_timing_neg': 'Nullable(Float64)', 'c22_acf_timescale': 'Nullable(Float64)', 'c22_acf_first_min': 'Nullable(Float64)', 'c22_low_freq_power': 'Nullable(Float64)', 'c22_centroid_freq': 'Nullable(Float64)', 'c22_ami_timescale': 'Nullable(Float64)', 'c22_periodicity': 'Nullable(Float64)', 'c22_ami2': 'Nullable(Float64)', 'c22_trev': 'Nullable(Float64)', 'c22_stretch_high': 'Nullable(Float64)', 'c22_stretch_decreasing': 'Nullable(Float64)', 'c22_entropy_pairs': 'Nullable(Float64)', 'c22_transition_variance': 'Nullable(Float64)', 'c22_whiten_timescale': 'Nullable(Float64)', 'c22_high_fluctuation': 'Nullable(Float64)', 'c22_forecast_error': 'Nullable(Float64)', 'c22_rs_range': 'Nullable(Float64)', 'c22_dfa': 'Nullable(Float64)', 'c22_embedding_dist': 'Nullable(Float64)'}

    print(f"Adding columns {'c22_mode_5', 'c22_mode_10', 'c22_outlier_timing_pos', 'c22_outlier_timing_neg', 'c22_acf_timescale', 'c22_acf_first_min', 'c22_low_freq_power', 'c22_centroid_freq', 'c22_ami_timescale', 'c22_periodicity', 'c22_ami2', 'c22_trev', 'c22_stretch_high', 'c22_stretch_decreasing', 'c22_entropy_pairs', 'c22_transition_variance', 'c22_whiten_timescale', 'c22_high_fluctuation', 'c22_forecast_error', 'c22_rs_range', 'c22_dfa', 'c22_embedding_dist'} to ConfigKernel.ts_preprocessing_properties")
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
    columns_to_remove = ['c22_mode_5', 'c22_mode_10', 'c22_outlier_timing_pos', 'c22_outlier_timing_neg', 'c22_acf_timescale', 'c22_acf_first_min', 'c22_low_freq_power', 'c22_centroid_freq', 'c22_ami_timescale', 'c22_periodicity', 'c22_ami2', 'c22_trev', 'c22_stretch_high', 'c22_stretch_decreasing', 'c22_entropy_pairs', 'c22_transition_variance', 'c22_whiten_timescale', 'c22_high_fluctuation', 'c22_forecast_error', 'c22_rs_range', 'c22_dfa', 'c22_embedding_dist']

    print(f"Removing columns {'c22_mode_5', 'c22_mode_10', 'c22_outlier_timing_pos', 'c22_outlier_timing_neg', 'c22_acf_timescale', 'c22_acf_first_min', 'c22_low_freq_power', 'c22_centroid_freq', 'c22_ami_timescale', 'c22_periodicity', 'c22_ami2', 'c22_trev', 'c22_stretch_high', 'c22_stretch_decreasing', 'c22_entropy_pairs', 'c22_transition_variance', 'c22_whiten_timescale', 'c22_high_fluctuation', 'c22_forecast_error', 'c22_rs_range', 'c22_dfa', 'c22_embedding_dist'} from ConfigKernel.ts_preprocessing_properties")
    for column_name in columns_to_remove:
        alter_query = f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties 
        DROP COLUMN IF EXISTS {column_name}
        """
        client_session.query(alter_query)
        print(f"Removed column {column_name} from table ConfigKernel.ts_preprocessing_properties")
