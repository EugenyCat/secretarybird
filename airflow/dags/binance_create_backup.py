from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.clickHouseBackupManager import ClickHouseBackupManagerFacade


binance_api = 'binance_api'


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


def create_backup(manager, db_name):
    """
        Call the method create_backup_database(), that create a new bd backup.
    """
    return manager.update_and_backup_database(db_name)


def is_there_a_problem_with_backup_creating(**kwargs):
    """
        Branch function to NOTIFY/NO NOTIFY when there are problems with `backup creating`
    """
    task_instance = kwargs['ti']
    message, error = task_instance.xcom_pull(task_ids=f'backup_{kwargs["db_name"]}')
    if error:
        if error['message'] is None:
            text = f'Something is wrong with dag etl_create_backup_{kwargs["db_name"]}'
        else:
            text = error['message']
        task_instance.xcom_push(key='error__creating_backup_process_message', value=text)
        return f'failed_create_backup_{kwargs["db_name"]}__notify'
    else:
        task_instance.xcom_push(key='success__creating_backup_process_message', value=message['message'])
        return f'success_create_backup_{kwargs["db_name"]}__no_notify'


with DAG(
        'BinanceAPI_ETL_part_4_create_backup',
        default_args=default_args,
        description='DAG for ETL process from Binance. STEP 4 - create backup',
        schedule_interval='30 12 1 * *',  #  at 12:30 on the first day of each month
        catchup=False,
    ) as dag_create_backup:

    # --- Step 4 --- Set up creating backups db tasks

    ch_backup_manager = ClickHouseBackupManagerFacade()

    # Init lists for storing PythonOperator for create backup tasks
    create_backup_tasks = []

    # Get databases names from clickhouse container
    databases = ch_backup_manager.get_databases()

    # Iterate through 'currency'+'interval' and init the `create backup process` tasks (by calling create_backup)
    for db_name in databases:

        # Create backup data task
        create_ch_db_backup_task = PythonOperator(
            task_id=f'backup_{db_name}',
            python_callable=create_backup,
            op_kwargs={'manager': ch_backup_manager, 'db_name': db_name},
        )
        create_backup_tasks.append(create_ch_db_backup_task)

    control_point = EmptyOperator(
        task_id=f'control_point',
        dag=dag_create_backup,
        trigger_rule=TriggerRule.NONE_FAILED
    )


    for i in range(len(create_backup_tasks)):
        create_backup_tasks[i] >> control_point