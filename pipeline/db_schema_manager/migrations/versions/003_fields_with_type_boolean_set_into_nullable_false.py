"""
fields with type Boolean set to Integer (UInt8) and not nullable
Date: 2025-05-09
"""

description = "Convert Boolean fields to Integer (UInt8) and set not nullable"
depends_on = "002"
auto_generated = True
schema_changes = {
    "ConfigKernel.ts_preprocessing_properties": {
        "database": "ConfigKernel",
        "table_name": "ts_preprocessing_properties",
        "is_new": 0,
        "added_columns": {},
        "removed_columns": {},
        "changed_columns": {
            "is_active": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 1"},
            "is_stationary": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"},
            "force_recalc_analyzer": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"},
            "force_recalc_periodicity": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"},
            "force_recalc_outlier": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"},
            "force_recalc_scaling": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"},
            "force_recalc_decomposition": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 0"}
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
            "is_active": {"from": "Nullable(Bool)", "to": "UInt8 DEFAULT 1"}
        },
        "affected_tables": [
            "time_series_definition"
        ]
    }
}

def upgrade(client_session):
    """Apply migration"""

    # Modify fields in the table ts_preprocessing_properties
    print("Modifying fields in table ConfigKernel.ts_preprocessing_properties")

    # Update NULL values before removing nullable
    client_session.query("""
    ALTER TABLE ConfigKernel.ts_preprocessing_properties
    UPDATE is_active = 1 WHERE is_active IS NULL
    """)

    client_session.query("""
    ALTER TABLE ConfigKernel.ts_preprocessing_properties
    UPDATE is_stationary = 0 WHERE is_stationary IS NULL
    """)

    for field in ['force_recalc_analyzer', 'force_recalc_periodicity', 'force_recalc_outlier',
                 'force_recalc_scaling', 'force_recalc_decomposition']:
        client_session.query(f"""
        ALTER TABLE ConfigKernel.ts_preprocessing_properties
        UPDATE {field} = 0 WHERE {field} IS NULL
        """)

    # Change field types in ts_preprocessing_properties
    client_session.query("""
    ALTER TABLE ConfigKernel.ts_preprocessing_properties
    MODIFY COLUMN is_active UInt8 DEFAULT 1,
    MODIFY COLUMN is_stationary UInt8 DEFAULT 0,
    MODIFY COLUMN force_recalc_analyzer UInt8 DEFAULT 0,
    MODIFY COLUMN force_recalc_periodicity UInt8 DEFAULT 0,
    MODIFY COLUMN force_recalc_outlier UInt8 DEFAULT 0,
    MODIFY COLUMN force_recalc_scaling UInt8 DEFAULT 0,
    MODIFY COLUMN force_recalc_decomposition UInt8 DEFAULT 0
    """)

    print("Successfully modified fields in table ConfigKernel.ts_preprocessing_properties")

    # Modify field is_active in table time_series_definition
    print("Modifying field is_active in table ConfigKernel.time_series_definition")

    # Update NULL values before removing nullable
    client_session.query("""
    ALTER TABLE ConfigKernel.time_series_definition
    UPDATE is_active = 1 WHERE is_active IS NULL
    """)

    # Change field type is_active
    client_session.query("""
    ALTER TABLE ConfigKernel.time_series_definition
    MODIFY COLUMN is_active UInt8 DEFAULT 1
    """)

    print("Successfully modified field is_active in table ConfigKernel.time_series_definition")


def downgrade(client_session):
    """Rollback migration"""

    # Revert field types in ts_preprocessing_properties
    print("Rolling back modifications in table ConfigKernel.ts_preprocessing_properties")

    client_session.query("""
    ALTER TABLE ConfigKernel.ts_preprocessing_properties
    MODIFY COLUMN is_active Nullable(Bool) DEFAULT true,
    MODIFY COLUMN is_stationary Nullable(Bool) DEFAULT false,
    MODIFY COLUMN force_recalc_analyzer Nullable(Bool) DEFAULT false,
    MODIFY COLUMN force_recalc_periodicity Nullable(Bool) DEFAULT false,
    MODIFY COLUMN force_recalc_outlier Nullable(Bool) DEFAULT false,
    MODIFY COLUMN force_recalc_scaling Nullable(Bool) DEFAULT false,
    MODIFY COLUMN force_recalc_decomposition Nullable(Bool) DEFAULT false
    """)

    print("Successfully rolled back changes in table ConfigKernel.ts_preprocessing_properties")

    # Revert field type is_active in table time_series_definition
    print("Rolling back modification of field is_active in table ConfigKernel.time_series_definition")

    client_session.query("""
    ALTER TABLE ConfigKernel.time_series_definition
    MODIFY COLUMN is_active Nullable(Bool) DEFAULT true
    """)

    print("Successfully rolled back field is_active in table ConfigKernel.time_series_definition")
