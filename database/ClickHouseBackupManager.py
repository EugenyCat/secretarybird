import logging

from database.ClickHouseConnection import ClickHouseConnection
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()

CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST')
BACKUPS_PATH = os.getenv('BACKUPS_PATH')
OLD_BACKUPS_SUBDIRECTORY = os.getenv('OLD_BACKUPS_SUBDIRECTORY')

class ClickHouseBackupManager:
    def __init__(self):
        self.db_session = ClickHouseConnection().get_session()
        self.ensure_subdirectory_exists()


    def ensure_subdirectory_exists(self):
        """
            Ensure that the `OLD_BACKUPS_SUBDIRECTORY` exists in the ClickHouse container.
        """
        try:
            result = subprocess.run(
                ['docker', 'exec', 'clickhouse-server', 'mkdir', '-p', f"{BACKUPS_PATH}{OLD_BACKUPS_SUBDIRECTORY}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error creating subdirectory {BACKUPS_PATH}{OLD_BACKUPS_SUBDIRECTORY}: {result.stderr}")

        except Exception as e:
            print(f"Error in ensure_subdirectory_exists: {e}")


    def get_databases(self):
        """
            Get all the name of existed databases from Clickhouse
        """
        try:
            all_databases = self.db_session.query('SHOW DATABASES').result_rows
            tech_db = {'INFORMATION_SCHEMA', 'default', 'information_schema', 'system'}
            logging.info(f'Database names are successfully gotten.')
        except Exception as e:
            print(f"Error: {e}")
            return []

        return [db[0] for db in all_databases if
                db[0] not in tech_db]


    def rename_existed_backups(self, db_name: str, old_backup_suffix='', new_backup_suffix='_old', sub_backup_directory=''):
        """
            Check if backup `old_backup_name` exists, if yes, rename it to new_backup_name and move it to subdirectory.
        """
        backups = self.get_backups_from_container(sub_backup_directory)
        old_backup_name = f'{db_name}{old_backup_suffix}-backup.zip'
        new_backup_name = f'{db_name}{new_backup_suffix}-backup.zip'

        if old_backup_name in backups:
            try:
                result = subprocess.run(
                    ['docker', 'exec', 'clickhouse-server', 'mv',
                     f"{BACKUPS_PATH}{sub_backup_directory}{old_backup_name}",
                     f"{BACKUPS_PATH}{OLD_BACKUPS_SUBDIRECTORY}{new_backup_name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    message = f"Error renaming backup {old_backup_name} to {new_backup_name}: {result.stderr}"
                    return None, {'status': 'error', 'message': message}

                return {'status': 'success', 'message': f'Successfully renamed {old_backup_name} to {new_backup_name} '}, None
            except Exception as e:
                message = f"Error in rename_existed_backups: {e}"
                return None, {'status': 'error', 'message': message}

        return {'status': 'success', 'message': f'rename_existed_backups: The backup doesn\'t exist {OLD_BACKUPS_SUBDIRECTORY}/{old_backup_name} yet'}, None


    def get_backups_from_container(self, subdirectory=''):
        """
            Get all the names of existed backups from Clickhouse container
        """
        try:
            # command ls (all the filenames in the BACKUPS_PATH) in container
            result = subprocess.run(
                ['docker', 'exec', 'clickhouse-server', 'sh', '-c',
                 f"find {BACKUPS_PATH}{subdirectory} -maxdepth 1 -type f -name '*.zip' -exec basename {{}} \\;"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return []

            files = result.stdout.strip().split('\n')
            return files

        except Exception as e:
            print(f"The process error get_backups_from_container: {e}")
            return []


    def create_backup_database(self, db_name):
        """
            Create backup of database db_name
        """
        try:
            backup_name = f'{db_name}-backup.zip'
            backup_request = f"BACKUP DATABASE {db_name} TO Disk('backups', '{backup_name}')"
            self.db_session.command(backup_request)
            return {'status': 'success', 'message': f"The backup {backup_name} created succesfully"}, None
        except Exception as e:
            message = f"The process error create_backup_database for {backup_name}: {e}"
            logging.info(message)
            return None, {'status': 'error', 'message': message}


    def delete_backup(self, backup_name: str):
        """
            Delete the backup {backup_name}.
        """
        backups = self.get_backups_from_container(OLD_BACKUPS_SUBDIRECTORY)
        subfolder_path = f"{BACKUPS_PATH}{OLD_BACKUPS_SUBDIRECTORY}"

        if backup_name in backups:
            try:
                result = subprocess.run(
                    ['docker', 'exec', 'clickhouse-server', 'rm',
                     f"{subfolder_path}/{backup_name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                   message = "Error deleting backup {backup_name}: {result.stderr}"
                   return None, {'status': 'error', 'message': message}

                return {'status': 'success',
                        'message': f'Successfully deleted {backup_name}'}, None
            except Exception as e:
                message = f"Error in delete_backup: {e}"
                return None, {'status': 'error', 'message': message}

        return {'status': 'success', 'message': f'delete_backup: The backup doesn\'t exist {OLD_BACKUPS_SUBDIRECTORY}{backup_name} yet'}, None


    def update_and_backup_database(self, db_name):
        """
            Rename existed backups, create a new and remove old backups
        """
        title = f'🚨🚨 Error backup creating for <b>{db_name}</b>\n'
        line_split_messages = '\n✦•·················•✦•·················•✦\n'

        # Rename and remove reserve old backup
        result, error = self.rename_existed_backups(db_name, old_backup_suffix='_old', new_backup_suffix='_old_remove', sub_backup_directory=OLD_BACKUPS_SUBDIRECTORY)
        if error:
            error['message'] += title + error['message'] + line_split_messages
            return None, error

        # Rename and remove current relevant backup
        result, error = self.rename_existed_backups(db_name)
        if error:
            error['message'] = title + error['message'] + line_split_messages
            return None, error

        # Create a new relevant backup
        result_create, error = self.create_backup_database(db_name)
        if error:
            error['message'] = title + error['message'] + line_split_messages
            return None, error

        # Remove backup_old_remove
        result, error = self.delete_backup(f'{db_name}_old_remove-backup.zip')
        if error:
            error['message'] = title + error['message'] + line_split_messages
            return None, error

        return {'status': 'success', 'message': result_create['message']}, None


    def restore_backup_database(self, backup_name):
        """
            Restore database from backup backup_name in case the database doesn't exist
        """
        # Init database_name and backup_path
        database_name = backup_name.split('-')[0]   # CRYPTO_1h-backup.zip -> CRYPTO_1h
        backup_path = f"Disk('backups', '{backup_name}')"

        # Check if db exists
        check_db_query = f"SHOW DATABASES LIKE '{database_name}'"
        db_exists = self.db_session.query(check_db_query).result_rows

        # If db doesn't exist, then RESTORE
        try:
            if not db_exists:
                restore_query = f"RESTORE DATABASE {database_name} FROM {backup_path}"
                self.db_session.command(restore_query)
                logging.info(f"DB {database_name} successfully recovered.")
            else:
                logging.info(f"DB {database_name} already exists.")
        except Exception as e:
            message = f'The process restore_backup_database finished with error for {backup_name}: {e}.'
            logging.info(message)
            return None, {'status': 'error', 'message': message}

        return {'status': 'success', 'message': f'The process restore_backup_database finished without errors for {backup_name}.'}, None