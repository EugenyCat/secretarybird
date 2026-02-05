import os
import importlib
import re
import json
import logging
from datetime import datetime
from pipeline.database.clickHouseConnection import ClickHouseConnection
from pipeline.db_schema_manager.migrations.comparator import SchemaComparator
from pipeline.db_schema_manager.migrations.generator import MigrationGenerator

# todo: translate entire block to English
# todo: create table based on orm schema from ./models/orm_models.py
# todo: how to avoid sql queries and use ORM and Query Builder
# todo: too much code with CREATE DATABASE IF NOT EXISTS - in this module and others


class MigrationManager:
    """Schema migration management for ClickHouse"""

    def __init__(self, client_session=None, sqlalchemy_session=None):
        """
        Initializes the migration manager

        Args:
            client_session: Existing ClickHouse session or None to create a new one
            sqlalchemy_session: Existing SQLAlchemy session or None
        """
        # Initialize connection if not provided
        if client_session is None:
            conn = ClickHouseConnection()
            self.client_session = conn.get_client_session()
        else:
            self.client_session = client_session

        self.sqlalchemy_session = sqlalchemy_session

        # Initialize helper components
        self.comparator = SchemaComparator()
        self.generator = MigrationGenerator()

        # Create necessary tables and initialize
        self._ensure_migration_table_exists()
        self._ensure_schema_registry_tables_exist()

        logging.info("Initializing schema registry with current schemas...")
        try:
            self._initialize_schema_registry()
            logging.info("Schema registry initialized successfully.")
        except Exception as e:
            logging.warning(f"Could not initialize schema registry: {e}")

    def _ensure_migration_table_exists(self):
        """Creates a table for tracking applied migrations"""

        create_db_query = "CREATE DATABASE IF NOT EXISTS InfraKernel"
        self.client_session.query(create_db_query)

        create_table_query = """
        CREATE TABLE IF NOT EXISTS InfraKernel.migrations (
            migration_id String,
            applied_at DateTime DEFAULT now(),
            description String,
            schema_changes String, -- JSON string with schema change information
            PRIMARY KEY (migration_id)
        ) ENGINE = MergeTree()
        """
        self.client_session.query(create_table_query)

    def _ensure_schema_registry_tables_exist(self):
        """Creates schema registry tables if they do not exist"""
        # schemas table
        query_schemas = """
        CREATE TABLE IF NOT EXISTS InfraKernel.schemas (
            schema_id String,
            current_version UInt32 DEFAULT 1,
            created_at DateTime DEFAULT now(),
            updated_at DateTime DEFAULT now(),
            PRIMARY KEY (schema_id)
        ) ENGINE = MergeTree()
        """
        self.client_session.query(query_schemas)

        # schema_versions table
        query_versions = """
        CREATE TABLE IF NOT EXISTS InfraKernel.schema_versions (
            schema_id String,
            version UInt32,
            schema_definition String, -- JSON string with schema
            changelog String,
            created_at DateTime DEFAULT now(),
            PRIMARY KEY (schema_id, version)
        ) ENGINE = MergeTree()
        ORDER BY (schema_id, version)
        """
        self.client_session.query(query_versions)

    def _initialize_schema_registry(self):
        """Initializes the schema registry if it is empty"""
        # Check if there are records in the schemas table
        check_query = """
            SELECT COUNT(*) FROM InfraKernel.schemas
        """
        result = self.client_session.query(check_query)
        count = result.result_rows[0][0]

        # If no records, initialize the registry
        if count == 0:
            # Create migration file
            self.auto_generate_migration("Initial schema snapshot", initialize_only=True)

            # Apply migration
            migration = self.get_pending_migrations()[0]
            module_name = f"pipeline.db_schema_manager.migrations.versions.{migration['filename'][:-3]}"
            module = importlib.import_module(module_name)

            # Call only the upgrade method
            module.upgrade(self.client_session)

            # Record information about applied migration
            query = """
                INSERT INTO InfraKernel.migrations 
                (migration_id, description) 
                VALUES ('001', 'Initial schema snapshot')
            """
            self.client_session.query(query)

    def get_available_migrations(self):
        """Finds all available migration files"""
        migrations_dir = os.path.join(os.path.dirname(__file__), 'versions')
        migration_files = [f for f in os.listdir(migrations_dir)
                           if re.match(r'^\d+_.*\.py$', f) and not f.startswith('__')]

        migrations = []
        for filename in sorted(migration_files):
            migration_id = filename.split('_')[0]
            module_name = f"pipeline.db_schema_manager.migrations.versions.{filename[:-3]}"
            module = importlib.import_module(module_name)

            migrations.append({
                'id': migration_id,
                'filename': filename,
                'description': getattr(module, 'description', ''),
                'depends_on': getattr(module, 'depends_on', None)
            })

        return migrations

    def get_applied_migrations(self):
        """Gets the list of already applied migrations"""
        query = """
            SELECT migration_id, applied_at, description, schema_changes 
            FROM InfraKernel.migrations 
            ORDER BY applied_at
        """
        result = self.client_session.query(query)

        applied = []
        for row in result.result_rows:
            applied.append({
                'id': row[0],
                'applied_at': row[1],
                'description': row[2],
                'schema_changes': row[3] if len(row) > 3 else None
            })

        return applied

    def get_pending_migrations(self):
        """Gets the list of migrations that have not yet been applied"""
        available = self.get_available_migrations()
        applied_ids = {m['id'] for m in self.get_applied_migrations()}

        return [m for m in available if m['id'] not in applied_ids]

    def update_schema_directly(self, schema_key, changes, is_upgrade=True):
        """
        Updates the schema registry directly via SQL queries

        Args:
            schema_key (str): Schema key (e.g., 'CRYPTO_raw')
            changes (dict): Description of schema changes
            is_upgrade (bool): True if applying migration, False if rolling back
        """
        logging.info(f"Direct update of schema {schema_key} in registry...")

        try:
            # Get current schema
            get_schema_query = f"""
                SELECT s.current_version, sv.schema_definition 
                FROM InfraKernel.schemas s
                JOIN InfraKernel.schema_versions sv 
                ON s.schema_id = sv.schema_id AND s.current_version = sv.version
                WHERE s.schema_id = '{schema_key}'
            """

            result = self.client_session.query(get_schema_query)

            if not result.result_rows:
                logging.warning(f"Schema {schema_key} not found in registry")
                return

            current_version = result.result_rows[0][0]
            current_schema_json = result.result_rows[0][1]

            try:
                current_schema = json.loads(current_schema_json)
                logging.debug(f"Current schema {schema_key} (version {current_version}): {current_schema}")
            except json.JSONDecodeError:
                logging.error(f"JSON decode error for current schema: {current_schema_json}")
                return

            # Apply changes to schema
            modified_schema = self.comparator.apply_schema_changes(current_schema, changes, is_upgrade)

            if is_upgrade:
                # Check if schema actually changed
                if not self.comparator.schemas_are_different(current_schema, modified_schema):
                    logging.info(f"Schema {schema_key} did not change after applying changes")
                    return

                logging.info(f"Schema {schema_key} changed")

                # Create new version
                new_version = current_version + 1

                # Check if such version already exists
                check_version_query = f"""
                SELECT COUNT(*) FROM InfraKernel.schema_versions 
                WHERE schema_id = '{schema_key}' AND version = {new_version}
                """
                result = self.client_session.query(check_version_query)

                if result.result_rows[0][0] > 0:
                    # Delete existing version to prevent duplication
                    delete_version_query = f"""
                    DELETE FROM InfraKernel.schema_versions 
                    WHERE schema_id = '{schema_key}' AND version = {new_version}
                    """
                    self.client_session.query(delete_version_query)

                # Save new schema version
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                schema_definition = json.dumps(modified_schema)

                # Insert new version
                insert_version_query = f"""
                INSERT INTO InfraKernel.schema_versions 
                (schema_id, version, schema_definition, changelog, created_at)
                VALUES 
                ('{schema_key}', {new_version}, '{schema_definition}', 'Migration-generated schema version', '{timestamp}')
                """
                self.client_session.query(insert_version_query)

                # Update current version
                update_schema_query = f"""
                ALTER TABLE InfraKernel.schemas 
                UPDATE current_version = {new_version}, updated_at = '{timestamp}' 
                WHERE schema_id = '{schema_key}'
                """
                self.client_session.query(update_schema_query)

                # Check if version was successfully updated
                check_update_query = f"""
                SELECT current_version FROM InfraKernel.schemas 
                WHERE schema_id = '{schema_key}'
                """
                check_update_result = self.client_session.query(check_update_query)

                if check_update_result.result_rows[0][0] != new_version:
                    logging.error(
                        f"CRITICAL ERROR: current_version was not updated to {new_version} in schemas table!")
                    # Retry update
                    force_update_query = f"""
                    ALTER TABLE InfraKernel.schemas 
                    UPDATE current_version = {new_version}, updated_at = '{timestamp}'
                    WHERE schema_id = '{schema_key}'
                    """
                    try:
                        self.client_session.query(force_update_query)
                        logging.info(f"Forced version update completed")
                    except Exception as e:
                        logging.error(f"Failed to forcefully update version: {e}")
                else:
                    logging.info(f"Updated schema {schema_key}: version {current_version} -> {new_version}")

            else:  # Rollback
                if current_version <= 1:
                    logging.warning(f"Schema {schema_key} is already at minimum version (1), rollback not possible")
                    return

                # Get previous schema version
                prev_version = current_version - 1

                # Update current version
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                update_schema_query = f"""
                ALTER TABLE InfraKernel.schemas 
                UPDATE current_version = {prev_version}, updated_at = '{timestamp}' 
                WHERE schema_id = '{schema_key}'
                """
                self.client_session.query(update_schema_query)

                # Delete current version
                delete_version_query = f"""
                DELETE FROM InfraKernel.schema_versions 
                WHERE schema_id = '{schema_key}' AND version = {current_version}
                """
                self.client_session.query(delete_version_query)

                logging.info(f"Rollback schema {schema_key}: version {current_version} -> {prev_version}")

        except Exception as e:
            logging.error(f"Error during direct update of schema {schema_key}: {e}")
            logging.debug("Error details:", exc_info=True)

    def _update_schema_registry(self, schema_changes, is_upgrade=True):
        """
        Updates the schema registry according to migration changes.
        Uses direct update via SQL for more reliable operation.

        Args:
            schema_changes (dict): Description of schema changes
            is_upgrade (bool): True if applying migration, False if rolling back
        """
        if not schema_changes:
            logging.info("No schema change information to update registry")
            return

        try:
            schema_changes_data = json.loads(schema_changes) if isinstance(schema_changes, str) else schema_changes
            logging.debug(f"Processing schema changes: {json.dumps(schema_changes_data, indent=2)}")

            # Process each schema
            for schema_key, changes in schema_changes_data.items():
                # Update schema directly via SQL
                self.update_schema_directly(schema_key, changes, is_upgrade)

        except Exception as e:
            logging.error(f"Error updating schema registry: {e}")
            logging.debug("Error details:", exc_info=True)

    def migrate(self, target_migration=None, dry_run=False):
        """
        Applies all pending migrations or up to the specified one

        Args:
            target_migration: Migration ID up to which to execute migration (inclusive)
            dry_run: If True, only shows what will be done without actual execution

        Returns:
            List of applied migrations
        """
        pending = self.get_pending_migrations()
        if not pending:
            logging.info("No pending migrations")
            return []

        applied = []
        for migration in pending:
            if target_migration and migration['id'] > target_migration:
                break

            module_name = f"pipeline.db_schema_manager.migrations.versions.{migration['filename'][:-3]}"
            module = importlib.import_module(module_name)

            logging.info(f"Applying migration {migration['id']}: {migration['description']}")

            # Determine schema changes before applying migration
            schema_changes = getattr(module, 'schema_changes', {}) if hasattr(module, 'auto_generated') else {}

            if not dry_run:
                try:
                    # Apply migration
                    module.upgrade(self.client_session)

                    # Update schema registry if there are changes
                    if schema_changes:
                        self._update_schema_registry(schema_changes, is_upgrade=True)

                    # Record information about applied migration
                    schema_changes_json = json.dumps(schema_changes) if schema_changes else ""
                    query = f"""
                    INSERT INTO InfraKernel.migrations 
                    (migration_id, description, schema_changes) 
                    VALUES (
                        '{migration['id']}', 
                        '{migration['description']}', 
                        '{schema_changes_json}'
                    )
                    """
                    self.client_session.query(query)

                    applied.append(migration)
                    logging.info(f"Migration {migration['id']} successfully applied")

                    # Output information about changed schemas
                    if schema_changes:
                        for schema_key in schema_changes:
                            logging.info(f"Updated schema: {schema_key}")

                except Exception as e:
                    logging.error(f"Error applying migration {migration['id']}: {str(e)}")
                    logging.debug("Error details:", exc_info=True)
                    raise
            else:
                applied.append(migration)
                logging.info(f"[DRY RUN] Migration {migration['id']} will be applied")

                # Output information about schema changes
                if schema_changes:
                    self._log_schema_changes(schema_changes, dry_run=True)

        return applied

    def _log_schema_changes(self, schema_changes, dry_run=False):
        """Logs schema changes in readable format"""
        prefix = "[DRY RUN] " if dry_run else ""

        for schema_key, changes in schema_changes.items():
            logging.info(f"{prefix}Schema: {schema_key}")

            if "added_columns" in changes and changes["added_columns"]:
                logging.info(f"{prefix}  Adding columns: {', '.join(changes['added_columns'].keys())}")

            if "removed_columns" in changes and changes["removed_columns"]:
                logging.info(f"{prefix}  Removing columns: {', '.join(changes['removed_columns'].keys())}")

            if "changed_columns" in changes and changes["changed_columns"]:
                changes_desc = [f"{col} ({old} -> {new})" for col, (old, new) in changes["changed_columns"].items()]
                logging.info(f"{prefix}  Changing columns: {', '.join(changes_desc)}")

    def rollback(self, migration_id, dry_run=False):
        """
        Rolls back a migration and all dependent ones

        Args:
            migration_id: Migration ID to rollback
            dry_run: If True, only shows what will be done without actual execution

        Returns:
            List of rolled back migrations
        """
        applied = self.get_applied_migrations()

        # Find migration to rollback
        migration_to_rollback = next((m for m in applied if m['id'] == migration_id), None)
        if not migration_to_rollback:
            logging.warning(f"Migration {migration_id} not found or was not applied")
            return []

        # Find dependent migrations (those applied after)
        to_rollback = [m for m in reversed(applied) if m['id'] >= migration_id]

        rolled_back = []
        for migration in to_rollback:
            # Find migration file
            available = self.get_available_migrations()
            migration_file = next((m['filename'] for m in available if m['id'] == migration['id']), None)

            if not migration_file:
                logging.error(f"Migration file for {migration['id']} not found")
                continue

            module_name = f"pipeline.db_schema_manager.migrations.versions.{migration_file[:-3]}"
            module = importlib.import_module(module_name)

            if not hasattr(module, 'downgrade'):
                logging.warning(f"Migration {migration['id']} does not have a rollback function")
                continue

            logging.info(f"Rolling back migration {migration['id']}: {migration['description']}")

            # Get schema changes from migration record
            schema_changes = {}
            if migration.get('schema_changes'):
                try:
                    schema_changes = json.loads(migration['schema_changes'])
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse JSON with schema changes for migration {migration['id']}")

            if not dry_run:
                try:
                    # Execute migration rollback
                    module.downgrade(self.client_session)

                    # Update schema registry if there are changes
                    if schema_changes:
                        self._update_schema_registry(schema_changes, is_upgrade=False)

                    # Delete migration record
                    query = f"""
                    DELETE FROM InfraKernel.migrations 
                    WHERE migration_id = '{migration['id']}'
                    """
                    self.client_session.query(query)

                    rolled_back.append(migration)
                    logging.info(f"Migration {migration['id']} successfully rolled back")

                except Exception as e:
                    logging.error(f"Error rolling back migration {migration['id']}: {str(e)}")
                    logging.debug("Error details:", exc_info=True)
                    raise
            else:
                rolled_back.append(migration)
                logging.info(f"[DRY RUN] Migration {migration['id']} will be rolled back")

                # Output information about schema changes
                if schema_changes:
                    self._log_schema_changes(schema_changes, dry_run=True)

        return rolled_back

    def auto_generate_migration(self, description="Auto-generated migration", initialize_only=False):
        """
        Automatically generates a migration by comparing schemas in code and DB

        Args:
            description: Migration description
            initialize_only: If True, creates initialization migration without comparing schemas

        Returns:
            Path to created file or None if no changes
        """
        # Get last migration ID and increment by 1
        migrations = self.get_available_migrations()
        next_id = '001'
        if migrations:
            last_id = max(int(m['id']) for m in migrations)
            next_id = f"{last_id + 1:03d}"

        # If initialize_only=True, create initialization migration
        if initialize_only:
            return self.generator.create_initial_migration(description, next_id)

        # Compare schemas and collect differences
        differences = self.comparator.compare_code_and_db_schemas(self.client_session)

        # If no differences, return None
        if not differences:
            logging.info("No schema changes found. Migration file not created.")
            return None

        # Generate migration file
        return self.generator.generate_migration_from_differences(description, differences, next_id)

    def auto_generate_orm_migration(self, description="Auto-generated ORM migration"):
        """
        Automatically generates a migration for ORM model schema changes

        Args:
            description (str): Migration description

        Returns:
            str: Path to created migration file or None if no changes
        """
        # Get all ORM models
        orm_models = self.comparator.get_all_orm_models()

        if not orm_models:
            logging.warning("ORM models not found")
            return None

        # Compare ORM models with current schemas in DB
        differences = self.comparator.compare_orm_models_with_db(orm_models, self.client_session)

        # If no differences, return None
        if not differences:
            logging.info("No ORM schema differences found")
            return None

        # Get next migration ID
        migrations = self.get_available_migrations()
        next_id = '001'
        if migrations:
            last_id = max(int(m['id']) for m in migrations)
            next_id = f"{last_id + 1:03d}"

        # Generate migration file
        return self.generator.generate_orm_migration_from_differences(description, differences, next_id)

    def auto_generate_all_migrations(self, description="Auto-generated schema migration"):
        """
        Automatically generates migrations for both time series and ORM tables

        Args:
            description (str): Migration description

        Returns:
            dict: Information about created migration files
        """
        result = {
            'time_series_migration': None,
            'orm_migration': None
        }

        # Generate migration for time series
        ts_file = self.auto_generate_migration(f"{description} - Time Series")
        if ts_file:
            result['time_series_migration'] = ts_file

        # Generate migration for ORM tables
        orm_file = self.auto_generate_orm_migration(f"{description} - ORM")
        if orm_file:
            result['orm_migration'] = orm_file

        return result