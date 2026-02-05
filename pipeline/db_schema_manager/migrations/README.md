# 🛠️ ClickHouse Schema Migration System

The migration system allows versioning and management of time series table schemas and ORM models in ClickHouse.

## 🔧 Basic Commands
```bash
# Status: which migrations are applied, which are pending
python -m pipeline.db_schema_manager.cli.migration_cli status

# Apply all pending migrations
python -m pipeline.db_schema_manager.cli.migration_cli migrate

# Rollback to specified migration (inclusive)
python -m pipeline.db_schema_manager.cli.migration_cli rollback 002

# Create empty migration with template
python -m pipeline.db_schema_manager.cli.migration_cli generate "Migration description"

# Auto-generate based on time series schema differences
python -m pipeline.db_schema_manager.cli.migration_cli auto-generate --name "Add Volume field"

# Auto-generate migrations only for ORM tables
python -m pipeline.db_schema_manager.cli.migration_cli orm-auto-generate --name "Add field to model"

# Generate migrations for all table types (time series + ORM)
python -m pipeline.db_schema_manager.cli.migration_cli auto-generate-all --name "Complex changes"
```

## ⚙️ Additional Options
```bash
# Apply migrations up to specific version
python -m pipeline.db_schema_manager.cli.migration_cli migrate --to 003

# Dry run (no changes)
python -m pipeline.db_schema_manager.cli.migration_cli migrate --dry-run
python -m pipeline.db_schema_manager.cli.migration_cli rollback 002 --dry-run

# Verbose output (for debugging)
python -m pipeline.db_schema_manager.cli.migration_cli status --verbose
```

## 🚀 Usage Scenarios

### Changing Time Series Schema
1. Modify schema in schema factory (add/remove field)
2. Generate migration:
```bash
   python -m pipeline.db_schema_manager.cli.migration_cli auto-generate --name "Description"
```
3. Review migration file and apply:
```bash
   python -m pipeline.db_schema_manager.cli.migration_cli migrate
```

### Changing ORM Models
1. Modify ORM model in `orm_models.py` (add/remove field)
2. Generate migration for ORM tables:
```bash
   python -m pipeline.db_schema_manager.cli.migration_cli orm-auto-generate --name "Model change"
```
3. Review migration file and apply:
```bash
   python -m pipeline.db_schema_manager.cli.migration_cli migrate
```

### Complex Changes
For changes affecting both time series and ORM models:
```bash
python -m pipeline.db_schema_manager.cli.migration_cli auto-generate-all --name "Full schema update"
```

## 📋 Migration Table Structure

The system uses the following tables in databases:

- `InfraKernel.migrations`: Tracks applied migrations
- `InfraKernel.schemas`: Stores information about current schema versions
- `InfraKernel.schema_versions`: Stores historical versions of all schemas

## 🏗️ System Architecture

The migration system consists of the following key components:

- **MigrationManager**: Main class for migration management (`manager.py`)
- **SchemaComparator**: Compares schemas between code and DB (`comparator.py`)
- **MigrationGenerator**: Creates migration files (`generator.py`)
- **SchemaExtractor**: Extracts schemas from ORM models (`orm_schema_extractor.py`)
- **TimeSeriesSchemaFactory**: Defines time series schemas (`ts_schema_factories.py`)

Migrations are stored in the `db_schema_manager/migrations/versions/` directory as Python files with `upgrade()` and `downgrade()` functions.

## 💡 Useful Tips

- Always implement `downgrade()` for safe migration rollback
- Test migrations with `--dry-run` before applying to production
- Do not edit already applied migrations — create new ones
- When changing ORM models ensure types correspond to ClickHouse
- Use `ClickHouseModelMixin` for all ORM models
- Explicitly specify `__clickhouse_order_by__` for all ORM models
- For each new ORM table in `ConfigKernel` add it to the first migration
- Keep migrations in version control system (Git)

## 🔍 Troubleshooting

If the migration system detects false differences in data types:
1. Check the equivalent types dictionary in `comparator.py`
2. Ensure `TYPE_MAPPING` in `orm_schema_extractor.py` is properly configured
3. Manually edit generated migration file, removing unnecessary type changes

If errors occur when running migrations:
1. Use `--verbose` flag to get detailed information
2. Check logs in execution directory (`migration_*.log` files)
3. Ensure ClickHouse connection is properly configured

## 🧪 Testing Migrations

To test migrations before applying to production:
1. Use `--dry-run` flag to simulate execution
2. Apply migration to test DB
3. Verify that `downgrade()` correctly rolls back changes

## 🔄 Continuous Integration Process

Recommended workflow for working with migrations:
1. Create branch for schema changes
2. Make changes to models/schema factories
3. Generate migration using `auto-generate-all`
4. Review and test migration with `--dry-run`
5. Apply migration to test environment
6. After verification create PR to main branch