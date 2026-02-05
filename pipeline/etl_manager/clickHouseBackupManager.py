from pipeline.etl_manager.baseManager import BaseManager
import subprocess
import logging
from system_files.constants.constants import CH_BACKUPS_PATH, CH_OLD_BACKUPS_SUBDIRECTORY


class ClickHouseBackupManager(BaseManager):
    """
        ClickHouseBackupManager is responsible for managing backups of a ClickHouse database.

        This class provides functionality to ensure backup directories exist, rename existing backups,
        delete old backups, and retrieve lists of backups stored within the ClickHouse Docker container.
        It extends from `BaseManager`, leveraging common configuration and database session handling.

        Key Responsibilities:
        - Ensure the backup subdirectory exists in the ClickHouse container.
        - Rename and move existing backups.
        - Delete old backups from the specified directories.
        - Retrieve lists of backups stored within the container.
    """

    def __init__(self, source_name, use_extended_timeout=False):
        """
            Initializes the ClickHouseBackupManager instance, setting up the backup management
            environment and ensuring required directories are in place.

            During initialization, this method performs the following actions:
            - Calls the parent class initializer to set up the base manager with configurations
              and database connection settings.
            - Ensures that the backup subdirectory exists within the ClickHouse Docker container.

            Args:
                source_name (str): The name of the source configuration to be used for the backup management.
                use_extended_timeout (bool): Optional flag to set an extended connection timeout
                                             if longer database operations are expected.
        """

        super().__init__(source_name=source_name, use_extended_timeout=use_extended_timeout)  # Call the parent class initializer

        # Ensure the backup subdirectory exists in the ClickHouse container
        self.ensure_subdirectory_exists()


    def ensure_subdirectory_exists(self):
        """
            Ensures that the specified backup subdirectory exists within the ClickHouse container.

            This method attempts to create the subdirectory defined by the constants
            `CH_BACKUPS_PATH` and `CH_OLD_BACKUPS_SUBDIRECTORY` in the ClickHouse Docker container.
            If the subdirectory creation fails, a warning will be logged with the error message.

            Raises:
                Exception: Logs a warning if an error occurs while executing the Docker command or
                during the subprocess execution.
        """
        try:
            # Execute the Docker command to create the backup subdirectory
            result = subprocess.run(
                ['docker', 'exec', 'clickhouse-server', 'mkdir', '-p', f"{CH_BACKUPS_PATH}{CH_OLD_BACKUPS_SUBDIRECTORY}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Check if the command was successful
            if result.returncode != 0:
                # Log a warning if the subdirectory could not be created
                logging.warning(f"Error creating subdirectory {CH_BACKUPS_PATH}{CH_OLD_BACKUPS_SUBDIRECTORY}: {result.stderr}")

        except Exception as e:
            # Log any exception that occurs during execution
            logging.warning(f"Error in ensure_subdirectory_exists: {e}")


    def rename_existed_backups(self, db_name: str, existing_backup_suffix='', renamed_backup_suffix='_old',
                               current_backup_directory=''):
        """
            Renames an existing backup if it exists and moves it to a specified subdirectory.

            This method checks if a backup file with the given `existing_backup_suffix` exists in the specified
            `current_backup_directory`. If it does, the method renames the backup to include `renamed_backup_suffix`
            and moves it to the `CH_OLD_BACKUPS_SUBDIRECTORY`.

            Args:
                db_name (str): The name of the database for which the backup is being managed.
                existing_backup_suffix (str, optional): The suffix for the existing backup name. Defaults to ''.
                renamed_backup_suffix (str, optional): The suffix to append for the renamed backup. Defaults to '_old'.
                current_backup_directory (str, optional): The subdirectory where the backup is currently located. Defaults to ''.

            Returns:
                tuple: A tuple containing a status dictionary and an optional error dictionary.
                       - On success: ({'status': 'success', 'message': '...'}, None)
                       - On failure: (None, {'status': 'error', 'message': '...'})
        """

        # Get the list of backups from the specified subdirectory
        backups = self.get_backups_from_container(current_backup_directory)

        # Construct the old and new backup names
        old_backup_name = f'{db_name}{existing_backup_suffix}-backup.zip'
        new_backup_name = f'{db_name}{renamed_backup_suffix}-backup.zip'

        # Check if the old backup name exists in the list of backups
        if old_backup_name in backups:
            try:
                # Execute the Docker command to rename and move the backup file
                result = subprocess.run(
                    ['docker', 'exec', 'clickhouse-server', 'mv',
                     f"{CH_BACKUPS_PATH}{current_backup_directory}{old_backup_name}",
                     f"{CH_BACKUPS_PATH}{CH_OLD_BACKUPS_SUBDIRECTORY}{new_backup_name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Check if the command was successful
                if result.returncode != 0:
                    message = f"Error renaming backup {old_backup_name} to {new_backup_name}: {result.stderr}"
                    return None, {'status': 'error', 'message': message}

                # Return success message if renaming was successful
                return {'status': 'success',
                        'message': f'Successfully renamed {old_backup_name} to {new_backup_name}'}, None

            except Exception as e:
                # Log and return an error message if an exception occurs
                message = f"Error in rename_existed_backups: {e}"
                return None, {'status': 'error', 'message': message}

        # Return a message indicating that the backup does not exist
        return {'status': 'success',
                'message': f'rename_existed_backups: The backup doesn\'t exist {CH_OLD_BACKUPS_SUBDIRECTORY}/{old_backup_name} yet'}, None


    def get_databases(self):
        """
            Retrieves the names of all existing databases from ClickHouse.

            Returns:
                list: A list of database names. If an error occurs, returns an empty list.
        """
        try:
            # Fetch all database names using the db_funcs object
            all_databases = self.db_funcs.get_databases()

            # Log success message if database names are fetched successfully
            logging.info(f'Database names are successfully retrieved.')

        except Exception as e:
            # Log the error message and return an empty list in case of an exception
            logging.error(f"Error while retrieving databases: {e}")
            return []

        # Return the list of all databases
        return all_databases


    def get_backups_from_container(self, subdirectory=''):
        """
            Retrieves the names of existing backup files from the ClickHouse container.

            This method executes a Docker command to list all `.zip` files in the specified
            backup directory within the ClickHouse container. It uses the `find` command to
            locate these files and returns their names without paths.

            Args:
                subdirectory (str, optional): The subdirectory within the backup path to search for backup files. Defaults to ''.

            Returns:
                list: A list of backup file names (without paths). Returns an empty list if no files are found
                      or if an error occurs during command execution.

            Raises:
                Exception: Logs an error message if an exception occurs during the execution of the subprocess.
        """

        try:
            # Execute a Docker command to find all .zip files in the specified backup directory
            result = subprocess.run(
                ['docker', 'exec', 'clickhouse-server', 'sh', '-c',
                 f"find {CH_BACKUPS_PATH}{subdirectory} -maxdepth 1 -type f -name '*.zip' -exec basename {{}} \\;"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Check if the command was successful
            if result.returncode != 0:
                logging.error(f"Error: {result.stderr}")
                return []  # Return an empty list if the command fails

            # Split the output into a list of file names
            files = result.stdout.strip().split('\n')
            return files  # Return the list of backup file names

        except Exception as e:
            # Log any exception that occurs during execution
            logging.warning(f"The process error get_backups_from_container: {e}")
            return []  # Return an empty list on exception


    def delete_backup(self, backup_name: str):
        """
            Deletes the specified backup file from the ClickHouse container.

            This method checks if the specified backup file exists in the
            CH_OLD_BACKUPS_SUBDIRECTORY. If it does, it attempts to delete the
            backup using a Docker command. The method returns a status message
            indicating the success or failure of the operation.

            Args:
                backup_name (str): The name of the backup file to delete.

            Returns:
                tuple: A tuple containing a status dictionary and an optional
                       error dictionary.
                       - On success: ({'status': 'success', 'message': '...'}, None)
                       - On failure: (None, {'status': 'error', 'message': '...'})
        """

        # Retrieve the list of backups from the CH_OLD_BACKUPS_SUBDIRECTORY
        backups = self.get_backups_from_container(CH_OLD_BACKUPS_SUBDIRECTORY)
        # Construct the full path to the sub_folder containing the backups
        sub_folder_path = f"{CH_BACKUPS_PATH}{CH_OLD_BACKUPS_SUBDIRECTORY}"

        try:
            # Check if the specified backup exists in the list of backups
            if backup_name in backups:
                # Execute the Docker command to remove the specified backup
                result = subprocess.run(
                    ['docker', 'exec', 'clickhouse-server', 'rm',
                     f"{sub_folder_path}/{backup_name}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Check if the command was successful
                if result.returncode != 0:
                    message = f"Error deleting backup {backup_name}: {result.stderr}"
                    return None, {'status': 'error', 'message': message}

                # Return success message if deletion was successful
                return {'status': 'success', 'message': f'Successfully deleted {backup_name}'}, None

            # Return success message if backup wasn't found
            return {'status': 'success', 'message': f'{backup_name} isn\'t found.'}, None

        except Exception as e:
            # Log and return an error message if an exception occurs
            message = f"Error in delete_backup: {e}"
            return None, {'status': 'error', 'message': message}


class ClickHouseBackupManagerFacade(ClickHouseBackupManager):
    """
        ClickHouseBackupManagerFacade provides a higher-level interface for managing backups of
        a ClickHouse database. It extends the functionality of ClickHouseBackupManager, adding
        additional features such as error notification and backup restoration.

        Key Responsibilities:
        - Handle the process of renaming existing backups.
        - Create new backups and manage backup deletion.
        - Restore databases from backups if they do not already exist.
        - Send notifications and log errors during backup operations.
    """

    def __init__(self, source_name, use_extended_timeout=False):
        """
            Initializes the ClickHouseBackupManagerFacade instance, setting up extended functionalities
            for comprehensive backup management and notification handling.

            This method calls the parent class initializer to set up inherited functionalities and
            prepares the facade for managing backups of the specified ClickHouse database.

            Args:
                source_name (str): The name of the source configuration to be used for backup management.
                use_extended_timeout (bool): Optional flag to apply an extended connection timeout
                                             for database operations.
        """
        # Call the parent class initializer to set up inherited functionalities
        super().__init__(source_name=source_name, use_extended_timeout=use_extended_timeout)


    def __construct_error_notification(self, title, text_message):
        """
            Constructs and sends a notification message, combining the title, text message,
            and line split delimiter. Also logs a warning with the text message.

            Args:
                title (str): The title or header of the notification.
                text_message (str): The body or content of the notification message.

            Returns:
                None
        """
        # Combine the title, text message, and line split into a single notification message
        notification_message = title + text_message + self.line_split_messages

        # Send the notification using the notifier instance
        self.notifier.notify(notification_message)

        # Log the text message as a warning in the system logs
        logging.error(f"{self.__str__()}: {text_message}")


    def update_and_backup_database(self):
        """
            Handles the process of renaming existing backups, creating a new backup,
            and removing old backups from the database.

            Steps:
                1. Rename existing backups with specific suffixes.
                2. Create a new database backup.
                3. Remove the backup with the '_old_remove' suffix.

            Returns:
                tuple: (result, error) where result contains the success message, and error contains
                       error details if any step fails.
        """
        title = f'🚨🚨 Error backup creating for <b>{self.database}</b>\n'

        # Define the number of backup versions to maintain
        backup_amount = 5

        # Loop through the backup versions in reverse order to rename them
        for i in range(backup_amount, 0, -1):
            # Define the existing backup suffix: no suffix for the first version
            existing_backup_suffix = '' if i == 1 else f'_{i - 1}'

            # Define the renamed backup suffix: increment the version number
            renamed_backup_suffix = f'_{i}'

            # Define the current backup directory: use the specified directory for all except the first backup
            current_backup_directory = CH_OLD_BACKUPS_SUBDIRECTORY if i != 1 else ''

            # Attempt to rename the existing backups with the specified parameters
            result, error = self.rename_existed_backups(
                self.database,
                existing_backup_suffix=existing_backup_suffix,
                renamed_backup_suffix=renamed_backup_suffix,
                current_backup_directory=current_backup_directory
            )

            # If an error occurs, notify and log the issue, then stop execution
            if error:
                self.__construct_error_notification(title, error['message'])
                return None, error

        # Try to create a new backup for the database
        try:
            backup_name = self.db_funcs.create_backup_database()
        except Exception as e:
            # Log and notify in case of an error while creating the new backup
            error_message = f"The process error create_backup_database: {e}"
            self.__construct_error_notification(title, error_message)
            return None, {'status': 'error', 'message': error_message}

        # Attempt to delete the last backup file with '_<backup_amount>' suffix
        result, error = self.delete_backup(f'{self.database}_{backup_amount}-backup.zip')
        if error:
            # Log and notify if an error occurs while deleting the old backup
            self.__construct_error_notification(title, error['message'])
            return None, error

        # Notify and log the successful backup creation
        result_message = f'{self.__str__()}: The backup {backup_name} created successfully'
        self.notifier.notify(result_message)
        logging.info(f"{result_message}")

        # Return success result and no errors
        return {'status': 'success', 'message': result_message}, None


    def restore_backup_database(self, database_name):
        """
            Restores the database from a backup if the current database doesn't exist.

            Args:
                database_name (str): The name of the restoring database.

            Returns:
                tuple: A dictionary containing the status ('success' or 'error') and a message,
                       along with a possible error dictionary.
        """
        title = f'🚨🚨 Error restoring from the backup for <b>{database_name}</b>\n'

        # Check if the database exists by filtering for its name
        db_exists = self.db_funcs.get_databases(filter_name=self.database)



        # If the database doesn't exist, proceed with the restoration process
        try:
            # Check if the backup exists
            backup_exists = f'{database_name}-backup.zip' in self.get_backups_from_container()
            if not db_exists and backup_exists:
                # Restore the database from the specified backup
                self.db_funcs.restore_database()
                logging.info(f"{self.__str__()}: DB successfully recovered.")
            elif not backup_exists:
                error_message = f'{self.__str__()}: the backup for restoring {database_name} isn\'t found.'
                self.__construct_error_notification(title, error_message)
            else:
                # Log a warning if the database already exists
                logging.warning(f"{self.__str__()}: DB already exists.")
        except Exception as e:
            # Handle exceptions during the restoration process and notify of the error
            error_message = f'{self.__str__()}: The process restore_backup_database finished with error: {e}.'
            self.__construct_error_notification(title, error_message)
            return None, {'status': 'error', 'message': error_message}

        # Log and notify of successful restoration
        result_message = f'{self.__str__()}: The process restore_backup_database finished without errors for {database_name}.'
        logging.info(result_message)

        return {'status': 'success', 'message': result_message}, None


    def __str__(self):
        return f'[etl_manager/clickHouseBackupManager.py] {self.database}'