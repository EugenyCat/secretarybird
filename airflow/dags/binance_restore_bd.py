from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.clickHouseBackupManager import ClickHouseBackupManagerFacade

API_NAME_LIST = ['binance_api']

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


with DAG(
    'BinanceAPI_ETL_part_1_restore_db',
    default_args=default_args,
    description='DAG for ETL process from Binance. STEP 1 - restore db from backup if db isn\'t found',
    schedule='2 * * * *',   # each hour , but start only when hh:02
    catchup=False,
) as dag_binance_restore_bd:

    # ---STEP 1--- Restoring database from backups if database doesn't exist

    # Init lists for storing PythonOperator for restoring databases
    restore_tasks = []

    # Iterate through 'API_NAME' and init the ClickHouseBackupManagerFacade instance with the certain API_NAME
    for API_NAME in API_NAME_LIST:
        # Retrieve the name of the database to restore from the configuration file (etl_assets/currencies.json)
        ch_backup_manager = ClickHouseBackupManagerFacade(source_name=API_NAME, use_extended_timeout=True)
        database = ch_backup_manager.get_api_configurations()['database']

        # Create the restoring db task
        restore_db_from_backup = PythonOperator(
            task_id=f'restore_if_not_exists_{database}',
            python_callable=ch_backup_manager.restore_backup_database,
            op_kwargs={'database_name': database},
            dag=dag_binance_restore_bd
        )
        restore_tasks.append(restore_db_from_backup)

    # Empty operator as a control_point
    control_point = EmptyOperator(
        task_id=f'control_point',
        dag=dag_binance_restore_bd,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    trigger_dag_binance_etl = TriggerDagRunOperator(
        task_id='trigger_dag_BinanceAPI_ETL_part_2_etl',
        trigger_dag_id='BinanceAPI_ETL_part_2_etl',  # the DAG ID for next running
        trigger_rule=TriggerRule.NONE_SKIPPED
    )


    for i in range(len(restore_tasks)):
        restore_tasks[i] >> control_point

    control_point >> trigger_dag_binance_etl