import os
import re
import json
import logging
from datetime import datetime


class MigrationGenerator:
    """
    Component for generating migration files based on schema differences.
    Generates code for upgrade and downgrade operations.
    """

    def create_initial_migration(self, description, next_id):
        """
        Creates an initial migration file

        Args:
            description (str): Migration description
            next_id (str): Migration ID

        Returns:
            str: Path to created file
        """
        migrations_dir = os.path.join(os.path.dirname(__file__), 'versions')
        file_path = os.path.join(migrations_dir, f'{next_id}_initial_schema_snapshot.py')

        # Create initial migration template
        template = self._get_initial_migration_template(description)

        # Save file
        with open(file_path, 'w') as f:
            f.write(template)

        logging.info(f"Created initialization migration file: {file_path}")
        return file_path

    def generate_migration_from_differences(self, description, differences, next_id):
        """
        Generates a migration file based on differences

        Args:
            description (str): Migration description
            differences (dict): Dictionary with schema differences
            next_id (str): Migration ID

        Returns:
            str: Path to created file
        """
        # Form filename
        migration_name = re.sub(r'[^a-z0-9_]', '_', description.lower())
        filename = f"{next_id}_{migration_name}.py"

        # Generate code for upgrade and downgrade
        upgrade_code = self._generate_upgrade_code(differences, next_id)
        downgrade_code = self._generate_downgrade_code(differences, next_id)

        # Form migration template
        migration_template = f"""\"\"\"
{description}
Date: {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import json
import sys
from datetime import datetime

description = "{description}"
depends_on = "{next_id if int(next_id) <= 1 else str(int(next_id) - 1).zfill(3)}"
auto_generated = True
schema_changes = {json.dumps(differences, indent=4)}

def upgrade(client_session):
    \"\"\"Apply migration\"\"\"
{"".join(upgrade_code) if upgrade_code else "    pass"}

def downgrade(client_session):
    \"\"\"Rollback migration\"\"\"
{"".join(downgrade_code) if downgrade_code else "    pass"}
"""

        # Save file
        migrations_dir = os.path.join(os.path.dirname(__file__), 'versions')
        file_path = os.path.join(migrations_dir, filename)

        with open(file_path, 'w') as f:
            f.write(migration_template)

        logging.info(f"Created migration file: {file_path}")
        return file_path

    def generate_orm_migration_from_differences(self, description, differences, next_id):
        """
        Generates a migration file based on ORM model schema differences

        Args:
            description (str): Migration description
            differences (dict): Dictionary with schema differences
            next_id (str): Migration ID

        Returns:
            str: Path to created file
        """
        # Form filename
        migration_name = re.sub(r'[^a-z0-9_]', '_', description.lower())
        filename = f"{next_id}_{migration_name}.py"

        # Generate code for upgrade and downgrade
        upgrade_code = self._generate_orm_upgrade_code(differences)
        downgrade_code = self._generate_orm_downgrade_code(differences)

        # Form migration template
        migration_template = f"""\"\"\"
{description}
Date: {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

import json
import sys
from datetime import datetime

description = "{description}"
depends_on = "{next_id if int(next_id) <= 1 else str(int(next_id) - 1).zfill(3)}"
auto_generated = True
schema_changes = {json.dumps(differences, indent=4)}

def upgrade(client_session):
    \"\"\"Apply migration for ORM tables\"\"\"
{"".join(upgrade_code) if upgrade_code else "    pass"}

def downgrade(client_session):
    \"\"\"Rollback migration for ORM tables\"\"\"
{"".join(downgrade_code) if downgrade_code else "    pass"}
"""

        # Save file
        migrations_dir = os.path.join(os.path.dirname(__file__), 'versions')
        file_path = os.path.join(migrations_dir, filename)

        with open(file_path, 'w') as f:
            f.write(migration_template)

        logging.info(f"Created ORM migration file: {file_path}")
        return file_path

    def _generate_upgrade_code(self, differences, next_id):
        """
        Generates upgrade code based on time series schema differences

        Args:
            differences (dict): Dictionary with schema differences
            next_id (str): Migration ID

        Returns:
            list: List of code strings for upgrade
        """
        upgrade_code = []

        for schema_key, diff in differences.items():
            database = diff["database"]
            processing_stage = diff["processing_stage"]
            interval = diff["interval"]
            affected_tables = diff["affected_tables"]

            # Convert table list to string
            tables_str = ", ".join([f"'{table}'" for table in affected_tables])

            # Adding columns
            if diff.get("added_columns"):
                columns_str = ", ".join([f"('{col}', '{dtype}')" for col, dtype in diff["added_columns"].items()])
                columns_names_str = ", ".join([f"'{col}'" for col in diff["added_columns"].keys()])

                upgrade_code.append(f"""
    # Adding columns to {schema_key} ({len(affected_tables)} tables)
    added_columns = [{columns_str}]
    affected_tables = [{tables_str}]

    print(f"Adding columns {{{columns_names_str}}} to schema {schema_key}")
    for table in affected_tables:
        for column_name, column_type in added_columns:
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
                alter_query = f\"\"\"
                ALTER TABLE {database}.{{table}} 
                ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}} DEFAULT {{default_value}}
                \"\"\"
            else:
                alter_query = f\"\"\"
                ALTER TABLE {database}.{{table}} 
                ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}}
                \"\"\"

            client_session.query(alter_query)
            print(f"Added column {{column_name}} to table {database}.{{table}}" + 
                  (f" with default value {{default_value}}" if default_value else ""))
                """)

            # Removing columns
            if diff.get("removed_columns"):
                # Create string with column info for upgrade
                removed_columns_names_str = ", ".join([f"'{col}'" for col in diff["removed_columns"].keys()])

                # Create string for downgrade with column type info
                removed_columns_str = ", ".join(
                    [f"('{col}', '{dtype}')" for col, dtype in diff["removed_columns"].items()])

                # Code for upgrade - removing columns
                upgrade_code.append(f"""
    # Removing columns from {schema_key} ({len(affected_tables)} tables)
    removed_columns = [{removed_columns_names_str}]
    affected_tables = [{tables_str}]

    # Save information about removed column types for possible restoration
    column_types = {{}}
    for table in affected_tables:
        table_column_types = {{}}
        for column_name in removed_columns:
            # Get column type information before removal
            describe_query = f\"\"\"
            DESCRIBE {database}.{{table}}
            \"\"\"
            result = client_session.query(describe_query)
            for row in result.result_rows:
                if row[0] == column_name:
                    table_column_types[column_name] = row[1]
                    break

        if table_column_types:
            column_types[table] = table_column_types

    print(f"Removing columns {{{removed_columns_names_str}}} from schema {schema_key}")
    print(f"Types of removed columns by table: {{column_types}}")

    # Write column type information to log file for possible restoration
    with open('column_types_backup_{next_id}.json', 'w') as f:
        import json
        json.dump(column_types, f, indent=2)

    for table in affected_tables:
        for column_name in removed_columns:
            alter_query = f\"\"\"
            ALTER TABLE {database}.{{table}} 
            DROP COLUMN IF EXISTS {{column_name}}
            \"\"\"
            client_session.query(alter_query)
            print(f"Removed column {{column_name}} from table {database}.{{table}}")
                """)

            # Changing column types
            if diff.get("changed_columns"):
                for col, (old_type, new_type) in diff["changed_columns"].items():
                    upgrade_code.append(f"""
    # Changing column type {col} in {schema_key} ({len(affected_tables)} tables)
    affected_tables = [{tables_str}]

    print(f"Changing column type {col} from {old_type} to {new_type}")
    for table in affected_tables:
        alter_query = f\"\"\"
        ALTER TABLE {database}.{{table}} 
        MODIFY COLUMN {col} {new_type}
        \"\"\"
        client_session.query(alter_query)
        print(f"Changed column type {col} to {new_type} in table {database}.{{table}}")
                    """)

        return upgrade_code

    def _generate_downgrade_code(self, differences, next_id):
        """
        Generates downgrade code based on time series schema differences

        Args:
            differences (dict): Dictionary with schema differences
            next_id (str): Migration ID

        Returns:
            list: List of code strings for downgrade
        """
        downgrade_code = []

        for schema_key, diff in differences.items():
            database = diff["database"]
            processing_stage = diff["processing_stage"]
            interval = diff["interval"]
            affected_tables = diff["affected_tables"]

            # Convert table list to string
            tables_str = ", ".join([f"'{table}'" for table in affected_tables])

            # Code for downgrade - restoring removed columns
            if diff.get("removed_columns"):
                removed_columns_names_str = ", ".join([f"'{col}'" for col in diff["removed_columns"].keys()])
                removed_columns_str = ", ".join(
                    [f"('{col}', '{dtype}')" for col, dtype in diff["removed_columns"].items()])

                downgrade_code.append(f"""
    # Restoring removed columns in {schema_key}
    # During removal we saved data types to file, so restoration should be accurate
    removed_columns_with_types = [{removed_columns_str}]  # [(column, type), ...]
    affected_tables = [{tables_str}]

    print(f"Restoring columns {{{removed_columns_names_str}}} in schema {schema_key}")

    # Try to load saved column types from log file
    column_types = {{}}
    try:
        with open('column_types_backup_{next_id}.json', 'r') as f:
            import json
            column_types = json.load(f)
        print(f"Loaded column types from backup: {{column_types}}")
    except Exception as e:
        print(f"Failed to load column types from file: {{e}}")
        print("Using column types specified in migration")

    for table in affected_tables:
        for column_name, default_column_type in removed_columns_with_types:
            # Use type from log file if available, otherwise from migration
            if table in column_types and column_name in column_types[table]:
                column_type = column_types[table][column_name]
            else:
                column_type = default_column_type

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

            try:
                # Add column with default value if defined
                if default_value is not None:
                    alter_query = f\"\"\"
                    ALTER TABLE {database}.{{table}} 
                    ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}} DEFAULT {{default_value}}
                    \"\"\"
                else:
                    alter_query = f\"\"\"
                    ALTER TABLE {database}.{{table}} 
                    ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}}
                    \"\"\"

                client_session.query(alter_query)
                print(f"Restored column {{column_name}} of type {{column_type}} to table {database}.{{table}}" + 
                      (f" with default value {{default_value}}" if default_value else ""))
            except Exception as e:
                print(f"Error restoring column {{column_name}}: {{e}}")
                # Try alternative approach if main one failed
                try:
                    alter_query = f\"\"\"
                    ALTER TABLE {database}.{{table}} 
                    ADD COLUMN IF NOT EXISTS {{column_name}} {{default_column_type}}
                    \"\"\"
                    client_session.query(alter_query)
                    print(f"Restored column {{column_name}} with alternative type {{default_column_type}}")
                except Exception as e2:
                    print(f"Also failed to restore with alternative type: {{e2}}")
                """)

            # Code for downgrade - removing added columns
            if diff.get("added_columns"):
                columns_names_str = ", ".join([f"'{col}'" for col in diff["added_columns"].keys()])

                downgrade_code.append(f"""
    # Removing added columns from {schema_key}
    affected_tables = [{tables_str}]
    added_columns = [{columns_names_str}]

    print(f"Removing columns {{{columns_names_str}}} from schema {schema_key}")
    for table in affected_tables:
        for column_name in added_columns:
            alter_query = f\"\"\"
            ALTER TABLE {database}.{{table}} 
            DROP COLUMN IF EXISTS {{column_name}}
            \"\"\"
            client_session.query(alter_query)
            print(f"Removed column {{column_name}} from table {database}.{{table}}")
                """)

            # Code for downgrade - reverting column types
            if diff.get("changed_columns"):
                for col, (old_type, new_type) in diff["changed_columns"].items():
                    downgrade_code.append(f"""
    # Reverting column type {col} in {schema_key} ({len(affected_tables)} tables)
    affected_tables = [{tables_str}]

    print(f"Reverting column type {col} from {new_type} to {old_type}")
    for table in affected_tables:
        alter_query = f\"\"\"
        ALTER TABLE {database}.{{table}} 
        MODIFY COLUMN {col} {old_type}
        \"\"\"
        client_session.query(alter_query)
        print(f"Changed column type {col} back to {old_type} in table {database}.{{table}}")
                    """)

        return downgrade_code

    def _generate_orm_upgrade_code(self, differences):
        """
        Generates upgrade code based on ORM model schema differences

        Args:
            differences (dict): Dictionary with ORM model schema differences

        Returns:
            list: List of code strings for upgrade
        """
        upgrade_code = []

        # Process new tables
        new_tables = {k: v for k, v in differences.items() if v["is_new"] == 1}
        if new_tables:
            # Import SchemaExtractor at the beginning of migration file
            upgrade_code.append("""
    # Import necessary modules
    from pipeline.db_schema_manager.models.orm_schema_extractor import SchemaExtractor
    import pipeline.db_schema_manager.models.orm_models as orm_models
    from datetime import datetime
    import json
    import logging

    def register_schema_in_registry(client_session, schema_key, schema_definition):
        \"\"\"
        Registers schema in the schema registry

        Args:
            client_session: ClickHouse connection session
            schema_key (str): Schema key (e.g., 'database.table_name')
            schema_definition (list): Schema definition as list of tuples [(column_name, type), ...]

        Returns:
            bool: True if registration successful, False on error
        \"\"\"
        try:
            # Extract database name from schema_key
            registry_db = "InfraKernel"  # Database for schema registry

            # Check existence of registry tables
            check_tables_query = f"SELECT 1 FROM system.tables WHERE database = '{registry_db}' AND name = 'schemas' LIMIT 1"
            check_tables_result = client_session.query(check_tables_query)
            if not check_tables_result.result_rows:
                print(f"WARNING: Schema registry tables do not exist in {registry_db}")
                return False

            # Check if schema already exists in registry
            check_schema_query = f"SELECT 1 FROM {registry_db}.schemas WHERE schema_id = '{schema_key}' LIMIT 1"
            check_result = client_session.query(check_schema_query)

            # Prepare data for insertion
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Ensure proper schema serialization to JSON
            # Schema should be a list of tuples [(column_name, type), ...]
            if isinstance(schema_definition, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in schema_definition):
                # Convert tuples to lists for correct serialization
                schema_list = [[col[0], col[1]] for col in schema_definition]
                schema_json = json.dumps(schema_list, ensure_ascii=False)
            else:
                print(f"ERROR: Invalid schema definition format: {schema_definition}")
                return False

            if not check_result.result_rows:
                # Register new schema

                # Insert record into schemas table
                insert_schema_query = f'''
                INSERT INTO {registry_db}.schemas 
                (schema_id, current_version, created_at, updated_at)
                VALUES 
                ('{schema_key}', 1, '{current_time}', '{current_time}')
                '''
                client_session.query(insert_schema_query)

                # Insert record into schema_versions table
                insert_version_query = f'''
                INSERT INTO {registry_db}.schema_versions 
                (schema_id, version, schema_definition, changelog, created_at)
                VALUES 
                ('{schema_key}', 1, '{schema_json}', 'Initial schema created by migration', '{current_time}')
                '''
                client_session.query(insert_version_query)

                print(f"Registered schema for {schema_key} in schema registry")
            else:
                print(f"Schema {schema_key} already exists in registry")

            return True
        except Exception as e:
            print(f"ERROR registering schema {schema_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Get list of exported names
    __all__ = orm_models.__all__

    # Dynamically inject each name into our namespace
    for name in __all__:
        globals()[name] = getattr(orm_models, name)
                """)

            for schema_key, diff in new_tables.items():
                table_name = diff["table_name"]
                database = diff["database"]

                # Add code for creating table
                upgrade_code.append(f"""
    # Creating new table {schema_key}
    print(f"Creating new table {schema_key}")

    # Get ORM model
    model_class = None
    for name, cls in globals().items():
        if hasattr(cls, '__tablename__') and cls.__tablename__ == '{table_name}':
            model_class = cls
            break

    if model_class:
        # Generate SQL for creating table
        create_table_sql = SchemaExtractor.generate_create_table_sql(model_class)
        client_session.query(create_table_sql)
        print(f"Created table {schema_key}")

        # Register schema in schema registry
        schema_definition = SchemaExtractor.extract_simple_schema(model_class)
        register_schema_in_registry(client_session, '{schema_key}', schema_definition)
    else:
        print(f"WARNING: Could not find ORM model for {table_name}")
                    """)

        # Process existing tables with changes
        existing_tables = {k: v for k, v in differences.items() if v["is_new"] == 0}
        for schema_key, diff in existing_tables.items():
            table_name = diff["table_name"]
            database = diff["database"]

            # Processing added columns
            if diff["added_columns"]:
                columns_str = ", ".join([f"'{col}': '{dtype}'" for col, dtype in diff["added_columns"].items()])
                columns_names_str = ", ".join([f"'{col}'" for col in diff["added_columns"].keys()])

                upgrade_code.append(f"""
    # Adding columns to table {schema_key}
    added_columns = {{{columns_str}}}

    print(f"Adding columns {{{columns_names_str}}} to {schema_key}")
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
            alter_query = f\"\"\"
            ALTER TABLE {schema_key} 
            ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}} DEFAULT {{default_value}}
            \"\"\"
        else:
            alter_query = f\"\"\"
            ALTER TABLE {schema_key} 
            ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}}
            \"\"\"

        client_session.query(alter_query)
        print(f"Added column {{column_name}} to table {schema_key}")
                    """)

            # Processing removed columns
            if diff["removed_columns"]:
                removed_columns_str = ", ".join(
                    [f"'{col}': '{dtype}'" for col, dtype in diff["removed_columns"].items()])
                removed_columns_names_str = ", ".join([f"'{col}'" for col in diff["removed_columns"].keys()])

                upgrade_code.append(f"""
    # Removing columns from table {schema_key}
    removed_columns = {{{removed_columns_str}}}

    print(f"Removing columns {{{removed_columns_names_str}}} from {schema_key}")
    for column_name in removed_columns.keys():
        alter_query = f\"\"\"
        ALTER TABLE {schema_key} 
        DROP COLUMN IF EXISTS {{column_name}}
        \"\"\"
        client_session.query(alter_query)
        print(f"Removed column {{column_name}} from table {schema_key}")
                    """)

            # Processing changed column types
            if diff["changed_columns"]:
                for col_name, (old_type, new_type) in diff["changed_columns"].items():
                    upgrade_code.append(f"""
    # Changing column type {col_name} in table {schema_key}
    print(f"Changing type of column {col_name} from {old_type} to {new_type} in {schema_key}")
    alter_query = \"\"\"
    ALTER TABLE {schema_key} 
    MODIFY COLUMN {col_name} {new_type}
    \"\"\"
    client_session.query(alter_query)
    print(f"Changed type of column {col_name} in table {schema_key}")
                        """)

        return upgrade_code


    def _generate_orm_downgrade_code(self, differences):
        """
        Generates downgrade code based on ORM model schema differences

        Args:
            differences (dict): Dictionary with ORM model schema differences

        Returns:
            list: List of code strings for downgrade
        """
        downgrade_code = []

        # Process new tables
        new_tables = {k: v for k, v in differences.items() if v["is_new"] == 1}
        if new_tables:
            for schema_key, diff in new_tables.items():
                table_name = diff["table_name"]
                database = diff["database"]

                # Add code for dropping table
                downgrade_code.append(f"""
    # Dropping table {schema_key}
    print(f"Dropping table {schema_key}")
    drop_table_sql = "DROP TABLE IF EXISTS {schema_key}"
    client_session.query(drop_table_sql)
    print(f"Dropped table {schema_key}")
                """)

        # Process existing tables with changes
        existing_tables = {k: v for k, v in differences.items() if v["is_new"] == 0}
        for schema_key, diff in existing_tables.items():
            table_name = diff["table_name"]
            database = diff["database"]

            # Processing added columns - remove them during downgrade
            if diff["added_columns"]:
                columns_names_str = ", ".join([f"'{col}'" for col in diff["added_columns"].keys()])

                downgrade_code.append(f"""
    # Removing added columns from table {schema_key}
    columns_to_remove = [{columns_names_str}]

    print(f"Removing columns {{{columns_names_str}}} from {schema_key}")
    for column_name in columns_to_remove:
        alter_query = f\"\"\"
        ALTER TABLE {schema_key} 
        DROP COLUMN IF EXISTS {{column_name}}
        \"\"\"
        client_session.query(alter_query)
        print(f"Removed column {{column_name}} from table {schema_key}")
                """)

            # Processing removed columns - restore them during downgrade
            if diff["removed_columns"]:
                removed_columns_str = ", ".join(
                    [f"'{col}': '{dtype}'" for col, dtype in diff["removed_columns"].items()])
                removed_columns_names_str = ", ".join([f"'{col}'" for col in diff["removed_columns"].keys()])

                downgrade_code.append(f"""
    # Restoring removed columns in table {schema_key}
    removed_columns = {{{removed_columns_str}}}

    print(f"Restoring columns {{{removed_columns_names_str}}} to {schema_key}")
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
            alter_query = f\"\"\"
            ALTER TABLE {schema_key} 
            ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}} DEFAULT {{default_value}}
            \"\"\"
        else:
            alter_query = f\"\"\"
            ALTER TABLE {schema_key} 
            ADD COLUMN IF NOT EXISTS {{column_name}} {{column_type}}
            \"\"\"

        client_session.query(alter_query)
        print(f"Restored column {{column_name}} to table {schema_key}")
                """)

            # Processing changed column types - restore old types during downgrade
            if diff["changed_columns"]:
                for col_name, (old_type, new_type) in diff["changed_columns"].items():
                    downgrade_code.append(f"""
    # Reverting column type {col_name} in table {schema_key}
    print(f"Reverting type of column {col_name} from {new_type} to {old_type} in {schema_key}")
    alter_query = \"\"\"
    ALTER TABLE {schema_key} 
    MODIFY COLUMN {col_name} {old_type}
    \"\"\"
    client_session.query(alter_query)
    print(f"Reverted type of column {col_name} in table {schema_key}")
                    """)

        return downgrade_code

    def _get_initial_migration_template(self, description):
        """
        Returns template for initialization migration

        Args:
            description (str): Migration description

        Returns:
            str: Template for initialization migration
        """
        current_date = datetime.now().strftime('%Y-%m-%d')

        return f"""\"\"\"
    {description}
    Date: {current_date}
    \"\"\"

from datetime import datetime
import json
import re
import sys
import importlib

description = "{description}"
depends_on = None

def get_all_database_tables(client_session):
    \"\"\"
    Gets all tables from database and groups them by type (raw/transformed)
    \"\"\"
    # Query all tables from supported databases
    supported_databases = ["'CRYPTO'"]
    query = f\"\"\"
    SELECT database, name 
    FROM system.tables 
    WHERE database IN ({{', '.join(supported_databases)}})
    \"\"\"

    result = client_session.query(query)

    # Dictionary for storing table information
    tables = {{
        'raw': [],
        'transformed': []
    }}

    # Regular expression for parsing table name
    pattern = r'^(.+)_(.+)_(raw|transformed)$'

    for row in result.result_rows:
        database, table_name = row

        # Extract table information using regular expression
        match = re.match(pattern, table_name)
        if match:
            currency = match.group(1)
            interval = match.group(2)
            processing_stage = match.group(3)

            # Add table to corresponding group
            table_info = {{
                'database': database,
                'table_name': table_name,
                'currency': currency,
                'interval': interval
            }}

            tables[processing_stage].append(table_info)

    print(f"Found {{len(tables['raw'])}} raw tables and {{len(tables['transformed'])}} transformed tables")
    return tables

def get_table_schema(client_session, database, table_name):
    \"\"\"
    Gets table schema from database
    \"\"\"
    describe_query = f\"\"\"
    DESCRIBE {{database}}.{{table_name}}
    \"\"\"
    result = client_session.query(describe_query)

    # Convert result to list of tuples
    schema = [(row[0], row[1]) for row in result.result_rows]

    return schema

def register_schema(client_session, schema_id, schema_definition, changelog="Initial schema"):
    \"\"\"
    Registers schema in registry
    \"\"\"
    # Convert schema definition to JSON string
    schema_json = json.dumps(schema_definition)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if schema with this ID already exists
    query = f\"\"\"
    SELECT current_version FROM InfraKernel.schemas 
    WHERE schema_id = '{{schema_id}}'
    \"\"\"

    result = client_session.query(query)

    if not result.result_rows:
        # Create new schema record and first version
        insert_schema = f\"\"\"
        INSERT INTO InfraKernel.schemas 
        (schema_id, current_version, created_at, updated_at)
        VALUES 
        ('{{schema_id}}', 1, '{{current_time}}', '{{current_time}}')
        \"\"\"

        insert_version = f\"\"\"
        INSERT INTO InfraKernel.schema_versions 
        (schema_id, version, schema_definition, changelog, created_at)
        VALUES 
        ('{{schema_id}}', 1, '{{schema_json}}', '{{changelog}}', '{{current_time}}')
        \"\"\"

        client_session.query(insert_schema)
        client_session.query(insert_version)

        print(f"Created new schema: {{schema_id}}")
    else:
        # Get current version
        current_version = result.result_rows[0][0]
        new_version = current_version + 1

        # Create new schema version
        insert_version = f\"\"\"
        INSERT INTO InfraKernel.schema_versions 
        (schema_id, version, schema_definition, changelog, created_at)
        VALUES 
        ('{{schema_id}}', {{new_version}}, '{{schema_json}}', '{{changelog}}', '{{current_time}}')
        \"\"\"

        # Update current version in schemas table
        update_schema = f\"\"\"
        UPDATE InfraKernel.schemas 
        SET current_version = {{new_version}}, updated_at = '{{current_time}}' 
        WHERE schema_id = '{{schema_id}}'
        \"\"\"

        client_session.query(insert_version)
        client_session.query(update_schema)

        print(f"Updated schema: {{schema_id}} (version {{new_version}})")

def register_all_current_schemas(client_session):
    \"\"\"
    Registers all current schemas in registry
    \"\"\"
    # Get all tables
    tables = get_all_database_tables(client_session)

    # Register raw table schemas
    # Group tables by database to create common schemas
    raw_schemas = {{}}
    for table_info in tables['raw']:
        database = table_info['database']
        schema_id = f"{{database}}_raw"

        if schema_id not in raw_schemas:
            raw_schemas[schema_id] = {{
                'database': database,
                'table': table_info['table_name']
            }}

    for schema_id, info in raw_schemas.items():
        # Get schema of first table for this database and type
        schema = get_table_schema(client_session, info['database'], info['table'])

        # Register schema
        register_schema(client_session, schema_id, schema, f"Initial raw schema for {{info['database']}}")

    # Register transformed table schemas
    # Group by database and interval
    transformed_schemas = {{}}
    for table_info in tables['transformed']:
        database = table_info['database']
        interval = table_info['interval']
        schema_id = f"{{database}}_{{interval}}_transformed"

        if schema_id not in transformed_schemas:
            transformed_schemas[schema_id] = {{
                'database': database,
                'interval': interval,
                'table': table_info['table_name']
            }}

    for schema_id, info in transformed_schemas.items():
        # Get schema of first table for this database, interval and type
        schema = get_table_schema(client_session, info['database'], info['table'])

        # Register schema
        register_schema(client_session, schema_id, schema, f"Initial transformed schema for {{info['database']}} {{info['interval']}}")

def create_orm_tables(client_session):
    \"\"\"
    Creates tables for ORM models in InfraKernel database
    \"\"\"
    print("Creating tables for ORM models...")

    # Create databases
    client_session.query("CREATE DATABASE IF NOT EXISTS InfraKernel")
    client_session.query("CREATE DATABASE IF NOT EXISTS ConfigKernel")

    # Proper import of ORM models
    from pipeline.db_schema_manager.models.orm_schema_extractor import SchemaExtractor
    import pipeline.db_schema_manager.models.orm_models as orm_module

    # Use model list from __all__
    __all__ = orm_module.__all__
    print(f"Importing ORM models: {{__all__}}")

    # Dynamically import each model into current namespace
    for name in __all__:
        globals()[name] = getattr(orm_module, name)

    # Get all ORM models
    orm_models_dict = {{}}
    for name in __all__:
        model_class = globals()[name]
        if hasattr(model_class, '__tablename__'):
            orm_models_dict[model_class.__tablename__] = model_class
            print(f"Added model: {{name}} -> {{model_class.__tablename__}}")

    # Create tables
    for table_name, model_class in orm_models_dict.items():
        database = getattr(model_class, "__clickhouse_database__", "InfraKernel")
        print(f"Creating table {{database}}.{{table_name}}...")

        try:
            # Generate SQL for creating table
            create_table_sql = SchemaExtractor.generate_create_table_sql(model_class)

            # Create table
            client_session.query(create_table_sql)
            print(f"Created table {{database}}.{{table_name}}")

            # Register table schema
            schema_id = f"{{database}}.{{table_name}}"
            schema_definition = SchemaExtractor.extract_simple_schema(model_class)
            register_schema(client_session, schema_id, schema_definition, f"Initial schema for {{table_name}}")

        except Exception as e:
            # On error continue creating other tables
            print(f"Error creating table {{table_name}}: {{e}}")
            import traceback
            traceback.print_exc()

    print("ORM table creation completed")

def upgrade(client_session):
    \"\"\"
    Apply migration: creates schema registry tables,
    ORM tables and registers all current schemas
    \"\"\"
    print("Initializing schema tables, ORM tables and registering existing schemas...")

    # First create ORM tables (including schemas and schema_versions tables)
    create_orm_tables(client_session)

    # Then register time series schemas
    register_all_current_schemas(client_session)

    print("Initialization of all schemas completed successfully.")

def downgrade(client_session):
    \"\"\"
    Rollback migration: drops all schema registry tables and ORM tables
    \"\"\"
    print("WARNING: This will delete all schema registry data and ORM tables!")

    # List of all ORM tables
    orm_tables = [
        "migrations",
        "schema_versions",
        "schemas",
        "time_series_definition",
        "models",
        "model_training",
        "model_quality"
    ]

    # Drop ORM tables
    for table in orm_tables:
        drop_query = f"DROP TABLE IF EXISTS InfraKernel.{{table}}"
        try:
            client_session.query(drop_query)
            print(f"Dropped table InfraKernel.{{table}}")
        except Exception as e:
            print(f"Error dropping table {{table}}: {{e}}")

    print("All schema registry tables and ORM tables have been dropped.")
"""