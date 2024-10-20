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


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

def restore_db_if_not_exists(manager, backup_name):
    """
        Call the method restore_backup_database(), that restores bd from files "backup_name" if db doesn't exist.
    """
    return manager.restore_backup_database(backup_name)


with DAG(
    'BinanceAPI_ETL_part_1_restore_db',
    default_args=default_args,
    description='DAG for ETL process from Binance. STEP 1 - restore db from backup if db isn\'t found',
    schedule='2 * * * *',   # each hour , but start only when hh:02
    catchup=False,
) as dag_binance_restore_bd:

    # ---STEP 1--- Restoring bd from backups if db doesn't exist

    # Get the list of existed backup names from container Clickhouse
    ch_backup_manager = ClickHouseBackupManagerFacade()
    backups = ch_backup_manager.get_backups_from_container()

    # Init lists for storing PythonOperator for restoring bd
    restore_tasks = []

    # Iterate through 'backups' and init the tasks for restoring db
    for backup_name in backups:
            # restoring db
            restore_db_from_backup = PythonOperator(
                task_id=f'restore_if_not_exists_{backup_name}',
                python_callable=ch_backup_manager.restore_backup_database,
                op_kwargs={'backup_name': backup_name},
                dag=dag_binance_restore_bd
            )
            restore_tasks.append(restore_db_from_backup)

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