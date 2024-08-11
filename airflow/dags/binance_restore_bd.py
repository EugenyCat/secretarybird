from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from database.ClickHouseBackupManager import ClickHouseBackupManager


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


def is_there_a_problem_with_restoring_db(**kwargs):
    """
        Branch function to NOTIFY/NO NOTIFY when there are problems with `restoring db from backup`
    """
    task_instance = kwargs['ti']
    message, error = task_instance.xcom_pull(task_ids=f'restore_if_not_exists_{kwargs["backup_name"]}')
    if error:
        if error['message'] is None:
            text = f'Something is wrong with dag_restore_{kwargs["backup_name"]}'
        else:
            text = error['message']
        task_instance.xcom_push(key='error__restore_db_message', value=text)
        return f'failed_restoring_{kwargs["backup_name"]}__notify'
    else:
        return f'success_restoring_{kwargs["backup_name"]}__no_notify'


with DAG(
    'BinanceAPI_ETL_part_1_restore_db',
    default_args=default_args,
    description='DAG for ETL process from Binance. STEP 1 - restore db from backup if db isn\'t found',
    schedule='2 * * * *',   # each hour , but start only when hh:02
    catchup=False,
) as dag_binance_restore_bd:

    # ---STEP 1--- Restoring bd from backups if db doesn't exist

    # get the list of existed backup names from container Clickhouse
    ch_backup_manager = ClickHouseBackupManager()
    backups = ch_backup_manager.get_backups_from_container()

    # Init lists for storing tasks for:
    # 1. PythonOperator for restoring tasks
    restore_tasks = []
    # 2. BranchPythonOperator for defining if the restoring process finished with problems or successfully
    branch_is_there_a_problem_with_restoring_db_tasks = []
    # 3. TelegramOperator for sending notifications if there are problems
    notify_about_restoring_problem_tasks = []
    # 4. EmptyOperator if the restoring process finished successfully
    no_notify_restoring_tasks = []

    # Iterate through 'backups' and init the tasks for
    #  - 1. restoring db
    #  - 2. branch check if the restoring process finished with problems or successfully
    #  - 3. telegram operator for sending notifications if there are problems
    #  - 4. empty operator if the restoring process finished successfully
    for backup_name in backups:
            # - 1. restoring db
            restore_db_from_backup = PythonOperator(
                task_id=f'restore_if_not_exists_{backup_name}',
                python_callable=restore_db_if_not_exists,
                op_kwargs={'manager': ch_backup_manager, 'backup_name': backup_name},
                dag=dag_binance_restore_bd
            )
            restore_tasks.append(restore_db_from_backup)

            #  - 2. branch check if the restoring process finished with problems or successfully
            define_restore_status = BranchPythonOperator(
                task_id=f'define_restore_status_{backup_name}',
                python_callable=is_there_a_problem_with_restoring_db,
                op_kwargs={'backup_name': backup_name},
                provide_context=True,
                dag=dag_binance_restore_bd
            )
            branch_is_there_a_problem_with_restoring_db_tasks.append(define_restore_status)

            #  - 3. telegram operator for sending notifications if there are problems
            notify_about_restoring_problem = TelegramOperator(
                task_id=f'failed_restoring_{backup_name}__notify',
                token=os.getenv('TOKEN_TELEGRAM_ALERT_BOT'),
                chat_id=os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT'),
                text= "{{ task_instance.xcom_pull(task_ids='define_restore_status_' + params.backup_name, key='error__restore_db_message') }}",
                params={'backup_name': backup_name},
                dag=dag_binance_restore_bd
            )
            notify_about_restoring_problem_tasks.append(notify_about_restoring_problem)

            #  - 4. empty operator if the restoring process finished successfully
            no_notification_restoring = EmptyOperator(
                task_id=f'success_restoring_{backup_name}__no_notify',
                dag=dag_binance_restore_bd
            )
            no_notify_restoring_tasks.append(no_notification_restoring)

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
        restore_tasks[i] >> branch_is_there_a_problem_with_restoring_db_tasks[i] >> [notify_about_restoring_problem_tasks[i], no_notify_restoring_tasks[i]]
        notify_about_restoring_problem_tasks[i] >> control_point
        no_notify_restoring_tasks[i] >> control_point
        time.sleep(0.05)

    control_point >> trigger_dag_binance_etl