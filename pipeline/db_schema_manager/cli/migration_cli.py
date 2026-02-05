#!/usr/bin/env python3
"""
CLI for managing ClickHouse schema migrations.

This script provides a command-line interface for:
- Viewing migration status
- Applying new migrations
- Rolling back migrations
- Generating new migration files (manual and automatic)

Usage examples:
    python -m pipeline.db_schema_manager.cli.migration_cli status
    python -m pipeline.db_schema_manager.cli.migration_cli migrate
    python -m pipeline.db_schema_manager.cli.migration_cli rollback 002
    python -m pipeline.db_schema_manager.cli.migration_cli generate "Add new fields"
    python -m pipeline.db_schema_manager.cli.migration_cli auto-generate --name "Time series schema changes"
    python -m pipeline.db_schema_manager.cli.migration_cli orm-auto-generate --name "ORM model changes"
    python -m pipeline.db_schema_manager.cli.migration_cli auto-generate-all --name "Complex changes"
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

try:
    from pipeline.database.clickHouseConnection import ClickHouseConnection
    from pipeline.db_schema_manager.migrations.manager import MigrationManager
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Make sure you are in the project root directory.")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='ClickHouse schema migration management')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # status command
    subparsers.add_parser('status', help='Show migration status')

    # migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Apply pending migrations')
    migrate_parser.add_argument('--to', help='Migration ID up to which to apply migrations (inclusive)')
    migrate_parser.add_argument('--dry-run', action='store_true', help='Dry run without actual changes')

    # rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback migrations')
    rollback_parser.add_argument('migration_id', help='Migration ID up to which to rollback (inclusive)')
    rollback_parser.add_argument('--dry-run', action='store_true', help='Dry run without actual changes')

    # generate command
    generate_parser = subparsers.add_parser('generate', help='Create an empty migration')
    generate_parser.add_argument('description', help='Migration description')

    # auto-generate command
    auto_generate_parser = subparsers.add_parser('auto-generate',
                                                 help='Automatically create a migration for time series based on schema differences')
    auto_generate_parser.add_argument('--name', required=True, help='Migration name')

    # orm-auto-generate command
    orm_auto_generate_parser = subparsers.add_parser('orm-auto-generate',
                                                     help='Automatically create a migration for ORM models')
    orm_auto_generate_parser.add_argument('--name', required=True, help='Migration name')

    # auto-generate-all command
    auto_generate_all_parser = subparsers.add_parser('auto-generate-all',
                                                     help='Automatically create migrations for all table types')
    auto_generate_all_parser.add_argument('--name', required=True, help='Migration name')

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args()


def display_migration_status(manager: MigrationManager):
    """Displays the current migration status"""
    applied = manager.get_applied_migrations()
    pending = manager.get_pending_migrations()

    print("\n=== Migration Status ===")

    print("\nApplied migrations:")
    if applied:
        for m in applied:
            print(f"  {m['id']}: {m['description']} (applied {m['applied_at']})")
    else:
        print("  No applied migrations")

    print("\nPending migrations:")
    if pending:
        for m in pending:
            print(f"  {m['id']}: {m['description']}")
    else:
        print("  No pending migrations")

    print("\n======================\n")


def execute_migrate(manager: MigrationManager, target_migration: Optional[str] = None, dry_run: bool = False):
    """Applies pending migrations"""
    print(
        f"\n{'[DRY RUN] ' if dry_run else ''}Applying migrations{' up to ' + target_migration if target_migration else ''}...")

    try:
        applied = manager.migrate(target_migration=target_migration, dry_run=dry_run)

        if applied:
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrations successfully applied:")
            for m in applied:
                print(f"  {m['id']}: {m['description']}")
        else:
            print("\nNo migrations to apply.")

    except Exception as e:
        logging.error(f"Error while applying migrations: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"\nError: {str(e)}")
        return False

    return True


def execute_rollback(manager: MigrationManager, migration_id: str, dry_run: bool = False):
    """Rolls back migrations up to the specified one (inclusive)"""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Rolling back migrations up to {migration_id}...")

    try:
        rolled_back = manager.rollback(migration_id=migration_id, dry_run=dry_run)

        if rolled_back:
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrations successfully rolled back:")
            for m in rolled_back:
                print(f"  {m['id']}: {m['description']}")
        else:
            print(f"\nNo migrations to rollback up to {migration_id}.")

    except Exception as e:
        logging.error(f"Error while rolling back migrations: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"\nError: {str(e)}")
        return False

    return True


def generate_empty_migration(manager: MigrationManager, description: str):
    """Generates an empty migration file with a basic template"""
    print(f"\nCreating empty migration: '{description}'...")

    # Get the last migration ID and increment by 1
    migrations = manager.get_available_migrations()
    next_id = '001'
    if migrations:
        last_id = max(int(m['id']) for m in migrations)
        next_id = f"{last_id + 1:03d}"

    # Build file name
    migration_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in description.lower())
    filename = f"{next_id}_{migration_name}.py"
    migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'migrations', 'versions')
    file_path = os.path.join(migrations_dir, filename)

    # Build migration template
    template = f"""\"\"\"
{description}
Date: {datetime.now().strftime('%Y-%m-%d')}
\"\"\"

description = "{description}"
depends_on = "{next_id if int(next_id) <= 1 else str(int(next_id) - 1).zfill(3)}"

def upgrade(client_session):
    \"\"\"Apply migration\"\"\"
    # Your code for applying the migration goes here
    pass

def downgrade(client_session):
    \"\"\"Rollback migration\"\"\"
    # Your code for rolling back the migration goes here
    pass
"""

    try:
        # Create directory if it does not exist
        os.makedirs(migrations_dir, exist_ok=True)

        # Save file
        with open(file_path, 'w') as f:
            f.write(template)

        print(f"\nMigration created: {file_path}")
        return True

    except Exception as e:
        logging.error(f"Error while creating migration file: {e}")
        print(f"\nError: {str(e)}")
        return False


def auto_generate_migration(manager: MigrationManager, name: str):
    """Automatically generates a migration based on time series schema differences"""
    print(f"\nAutomatic migration generation for time series: '{name}'...")

    try:
        file_path = manager.auto_generate_migration(description=name)

        if file_path:
            print(f"\nMigration successfully created: {file_path}")
        else:
            print("\nNo time series schema changes found to create a migration.")

        return bool(file_path)

    except Exception as e:
        logging.error(f"Error while generating migration: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"\nError: {str(e)}")
        return False


def orm_auto_generate_migration(manager: MigrationManager, name: str):
    """Automatically generates a migration based on ORM model differences"""
    print(f"\nAutomatic migration generation for ORM models: '{name}'...")

    try:
        file_path = manager.auto_generate_orm_migration(description=name)

        if file_path:
            print(f"\nMigration successfully created: {file_path}")
        else:
            print("\nNo ORM model changes found to create a migration.")

        return bool(file_path)

    except Exception as e:
        logging.error(f"Error while generating ORM migration: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"\nError: {str(e)}")
        return False


def auto_generate_all_migrations(manager: MigrationManager, name: str):
    """Automatically generates migrations for all table types"""
    print(f"\nAutomatic migration generation for all table types: '{name}'...")

    try:
        result = manager.auto_generate_all_migrations(description=name)

        if result['time_series_migration'] or result['orm_migration']:
            print("\nMigrations successfully created:")
            if result['time_series_migration']:
                print(f"  Time series: {result['time_series_migration']}")
            if result['orm_migration']:
                print(f"  ORM models: {result['orm_migration']}")
        else:
            print("\nNo changes found to create migrations.")

        return bool(result['time_series_migration'] or result['orm_migration'])

    except Exception as e:
        logging.error(f"Error while generating migrations: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"\nError: {str(e)}")
        return False


def main():
    """Main CLI function"""
    args = parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose output mode enabled")

    # Initialize migration manager
    try:
        logging.info("Initializing ClickHouse connection...")
        connection = ClickHouseConnection()
        client_session = connection.get_client_session()

        logging.info("Initializing migration manager...")
        manager = MigrationManager(client_session=client_session)

    except Exception as e:
        logging.error(f"Initialization error: {e}")
        logging.debug("Error details:", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Execute command
    success = True

    if args.command == 'status':
        display_migration_status(manager)

    elif args.command == 'migrate':
        success = execute_migrate(manager, args.to, args.dry_run)

    elif args.command == 'rollback':
        success = execute_rollback(manager, args.migration_id, args.dry_run)

    elif args.command == 'generate':
        success = generate_empty_migration(manager, args.description)

    elif args.command == 'auto-generate':
        success = auto_generate_migration(manager, args.name)

    elif args.command == 'orm-auto-generate':
        success = orm_auto_generate_migration(manager, args.name)

    elif args.command == 'auto-generate-all':
        success = auto_generate_all_migrations(manager, args.name)

    else:
        print("Unknown command. Use --help for usage information.")
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()