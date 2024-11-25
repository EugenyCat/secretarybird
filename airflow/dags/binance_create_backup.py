from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# Set up system path to access required modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from etl_pipeline.etl_manager.clickHouseBackupManager import ClickHouseBackupManagerFacade
from etl_pipeline.etl_manager.grafanaBackupManager import GrafanaBackupManagerFacade

API_NAME_LIST = ['binance_api']

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

with DAG(
        'ETL_Part_4_create_backup_monthly',
        default_args=default_args,
        description='Unified DAG for creating backups for Binance and Grafana dashboards monthly',
        schedule_interval='30 12 1 * *',  # Run at 12:30 on the 1st of every month
        catchup=False,
    ) as dag_unified_backup:

    # --- Step 1 --- Setup for creating database backup tasks for ClickHouse
    create_backup_tasks = []

    for API_NAME in API_NAME_LIST:
        # Initialize the ClickHouseBackupManagerFacade for each API
        ch_backup_manager = ClickHouseBackupManagerFacade(API_NAME)
        database = ch_backup_manager.get_api_configurations()['database']

        # Create a backup task for each ClickHouse database
        create_ch_db_backup_task = PythonOperator(
            task_id=f'backup_{database}',
            python_callable=ch_backup_manager.update_and_backup_database
        )
        create_backup_tasks.append(create_ch_db_backup_task)

    # --- Step 2 --- Setup for creating Grafana dashboard backup and restore task

    # Initialize GrafanaBackupManagerFacade instance for handling Grafana dashboard backup and restore
    grafana_backup_manager = GrafanaBackupManagerFacade()

    # Define the task to backup and restore all Grafana dashboards
    grafana_backup_restore_task = PythonOperator(
        task_id='backup_and_restore_grafana_dashboards',
        python_callable=grafana_backup_manager.backup_and_restore
    )

    # Add Grafana task to the list of backup tasks for sequencing
    create_backup_tasks.append(grafana_backup_restore_task)

    # --- Step 3 --- Define a control point to manage task dependencies
    control_point = EmptyOperator(
        task_id='control_point',
        dag=dag_unified_backup,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    # Set up dependencies so each backup task flows into the control point
    for task in create_backup_tasks:
        task >> control_point
