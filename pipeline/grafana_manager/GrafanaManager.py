import os
import requests
import json

#TODO: refactor this code fully
class GrafanaManager:
    # Задайте ваш API ключ и URL Grafana
    API_KEY = "glsa_fILYA3dm78LtAgs8dcOpeZMtKH3yiBY9_e201595e"
    GRAFANA_URL = "http://localhost:3000"
    BACKUP_DIR = "../../system_files/provisioning/dashboards/"

    # Создайте директорию для хранения бэкапов
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Функция для получения списка всех дашбордов
    def get_all_dashboards(self):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{GRAFANA_URL}/api/search?query=&", headers=headers)
        response.raise_for_status()
        return response.json()

    # Функция для экспорта дашборда по UID
    def export_dashboard(self, uid, title):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{GRAFANA_URL}/api/dashboards/uid/{uid}", headers=headers)
        response.raise_for_status()
        dashboard_json = response.json()

        # Замените пробелы в названии на подчеркивания для корректного имени файла
        safe_title = title.replace(" ", "_")
        filename = os.path.join(BACKUP_DIR, f"{safe_title}_{uid}.json")

        # Сохраните JSON в файл
        with open(filename, 'w') as f:
            json.dump(dashboard_json, f, indent=4)

    # Основной блок
    def to_backup(self):

        dashboards = self.get_all_dashboards()

        for dashboard in dashboards:
            uid = dashboard['uid']
            title = dashboard['title']
            self.export_dashboard(uid, title)
