import os
import requests
import json
import base64
from datetime import datetime
from system_files.constants.constants import GRAFANA_BACKUP_DIR
from pipeline.helpers.telegram_notifier import TelegramNotifier
import logging


class GrafanaBackupManager:
    """
        Handles API requests to Grafana.
    """

    __API_KEY = os.getenv('API_KEY')
    __GRAFANA_URL = os.getenv('GRAFANA_URL')
    _BACKUP_DIR = GRAFANA_BACKUP_DIR

    def __init__(self):
        """
            Initializes the GrafanaAPI with necessary credentials.
        """
        if not self.__API_KEY or not self.__GRAFANA_URL:
            raise ValueError("API_KEY and GRAFANA_URL must be set in the environment variables.")


    def __get_headers(self):
        """
            Return the headers required for API requests.
        """
        return {
            "Authorization": f"Basic {self.__get_basic_auth()}",
            "Content-Type": "application/json"
        }


    def __get_basic_auth(self):
        """
            Generate Basic Auth header value.
        """
        __USERNAME = os.getenv('GF_SECURITY_ADMIN_USER')
        __PASSWORD = os.getenv('GF_SECURITY_ADMIN_PASSWORD')
        credentials = f"{__USERNAME}:{__PASSWORD}"
        return base64.b64encode(credentials.encode()).decode()


    def get_all_dashboards(self):
        """
            Retrieve a list of all dashboards from Grafana.
        """
        try:
            response = requests.get(f"{self.__GRAFANA_URL}/api/search?query=&", headers=self.__get_headers())
            response.raise_for_status()
            logging.info(f"{self.__str__()}: Successfully retrieved all dashboards.")
            return response.json()
        except requests.HTTPError as http_err:
            logging.error(f"{self.__str__()}: HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            logging.error(f"{self.__str__()}: An error occurred: {err}")
            return None


    def export_dashboard(self, uid):
        """
            Export a dashboard by its UID and return the JSON data.
        """
        try:
            response = requests.get(f"{self.__GRAFANA_URL}/api/dashboards/uid/{uid}", headers=self.__get_headers())
            response.raise_for_status()
            return response.json()
        except Exception as err:
            logging.error(f"{self.__str__()}: Failed to export dashboard with UID '{uid}': {err}")
            return None


    def restore_dashboard(self, dashboard_json):
        """
            Restore a dashboard from a JSON object.
        """
        try:
            # Remove meta information and set id to null
            if 'meta' in dashboard_json:
                del dashboard_json['meta']
            dashboard_json['dashboard']['id'] = None  # Ensure the ID is null

            response = requests.post(f"{self.__GRAFANA_URL}/api/dashboards/db", headers=self.__get_headers(),
                                     json=dashboard_json)

            # Check the response status
            if response.status_code == 412:  # Dashboard already exists
                warning_message = f"{self.__str__()}: Dashboard '{dashboard_json['dashboard']['title']}' already exists in Grafana."
                logging.warning(warning_message)
                return {"status": "warning",
                        "message": warning_message}

            response.raise_for_status()  # Raise for other errors
            info_message = f"{self.__str__()}: Dashboard '{dashboard_json['dashboard']['title']}' has been successfully restored!"
            logging.info(info_message)
            return {"status": "success",
                    "message": info_message}

        except requests.HTTPError as http_err:
            error_message = f"{self.__str__()}: HTTP error occurred: {http_err}"
            logging.error(error_message)
            return {"status": "error", "message": error_message}
        except Exception as err:
            error_message = f"{self.__str__()}: Failed to restore dashboard: {err}"
            logging.error(error_message)
            return {"status": "error", "message": error_message}


    def __str__(self):
        return '[etl_manager/GrafanaBackupManager.py]'


class GrafanaBackupManagerFacade(GrafanaBackupManager):
    """
        Facade class to manage backup and restoration of Grafana dashboards.
    """

    # Variables for generating awesome alerts
    title = "⚠️️ REPORT about problems for grafana backup"
    line_str = '⤵\n'
    line_split_messages = '\n✦•·················•✦•·················•✦\n'


    def __init__(self):
        """
            Initializes the facade and the GrafanaBackupManager.
        """
        super().__init__()

        # Initialize the TelegramNotifier for sending notifications about backup status
        self.notifier = TelegramNotifier()


    def _notify_error(self, error_message: str, dashboard_name: str):
        """
            Generates a formatted error report for Grafana backup issues and
            sends a notification about an error to Telegram.
        """
        report = (
            f"{self.title} <b>{dashboard_name}</b>:\n"
            f" ➜ 🚨🚨🚨 {error_message}\n\n"
            f"{self.line_split_messages}"
        )
        self.notifier.notify(report)  # Assuming that TelegramNotifier has a method notify


    def backup_all_dashboards(self):
        """
            Perform a complete backup of all dashboards.
        """
        dashboards = self.get_all_dashboards()
        if dashboards is None:
            error_message = f"{self.__str__()}: Failed to retrieve dashboards."
            self._notify_error(error_message, "all dashboards")
            return {"status": "error", "message": error_message}

        for dashboard in dashboards:
            uid = dashboard['uid']
            export_response = self.export_dashboard(uid)
            if export_response:
                # Construct a safe filename and save the JSON to a file
                safe_title = dashboard['title'].replace(" ", "_")
                filename = f"{safe_title}_{uid}_{datetime.now().strftime('%Y%m%d')}.json"
                filepath = os.path.join(self._BACKUP_DIR, filename)

                with open(filepath, 'w') as f:
                    json.dump(export_response, f, indent=4)

                logging.info(f"{self.__str__()}: Dashboard '{dashboard['title']}' exported successfully to {filepath}.")
            else:
                error_message = f"{self.__str__()}: Failed to export dashboard with UID '{uid}'."
                logging.error(error_message)
                self._notify_error(error_message, dashboard['title'])

        info_message = f"{self.__str__()}: All dashboards have been successfully backed up!"
        logging.info(info_message)
        return {"status": "success", "message": info_message}


    def restore_all_dashboards(self):
        """
            Restore all dashboards from the backup directory.
        """
        for filename in os.listdir(self._BACKUP_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(self._BACKUP_DIR, filename)
                with open(filepath, 'r') as f:
                    dashboard_json = json.load(f)
                restore_response = self.restore_dashboard(dashboard_json)
                if restore_response['status'] == "error":
                    error_message = restore_response['message']
                    logging.error(error_message)
                    self._notify_error(error_message, dashboard_json['dashboard']['title'])

        info_message = f"{self.__str__()}: All dashboards have been restored successfully."
        logging.info(info_message)
        return {"status": "success", "message": info_message}

    def backup_and_restore(self):
        """
            Backup all dashboards and then restore them.
        """
        backup_response = self.backup_all_dashboards()
        if backup_response['status'] == 'error':
            return backup_response  # Return on first error

        restore_response = self.restore_all_dashboards()
        return restore_response


# Example usage
#manager = GrafanaBackupManagerFacade()
# manager.backup_all_dashboards()
#manager.backup_and_restore()
