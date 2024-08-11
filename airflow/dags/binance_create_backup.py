from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from datetime import datetime, timedelta
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from database.ClickHouseBackupManager import ClickHouseBackupManager


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

    ch_backup_manager = ClickHouseBackupManager()

    """
    # Sensor to wait until the first day of the month
    monthly_backup_sensor = TimeDeltaSensor(
        task_id='monthly_backup_sensor',
        delta=timedelta(days=(datetime.now().day - 31) % 30),  # Запуск в 12:30 первого числа каждого месяца,
        mode='reschedule',
        dag=dag_create_backup,
    )
    """

    # Init lists for storing tasks for:
    # 1. PythonOperator for create backup tasks
    create_backup_tasks = []
    # 2. BranchPythonOperator for defining if the `create backup process` finished with problems or successfully
    branch_is_there_a_problem_with_create_backup_tasks = []
    # 3. TelegramOperator for sending notifications if there are problems
    notify_about_create_backup_problem_tasks = []
    # 4. EmptyOperator if the `create backup process` finished successfully
    no_notify_create_backup_tasks = []

    # Get databases names from clickhouse container
    databases = ch_backup_manager.get_databases()

    # Iterate through 'currency'+'interval' and init the tasks for
    #  - 1. `create backup process` (by calling create_backup)
    #  - 2. branch check if the `create backup process` finished with problems or successfully
    #  - 3. telegram operator for sending notifications if there are problems
    #  - 4. empty operator if the `create backup process` finished successfully
    for db_name in databases:

        # - 1. create backup data task
        create_ch_db_backup_task = PythonOperator(
            task_id=f'backup_{db_name}',
            python_callable=create_backup,
            op_kwargs={'manager': ch_backup_manager, 'db_name': db_name},
        )
        create_backup_tasks.append(create_ch_db_backup_task)

        #  - 2. branch check if the `create backup process` finished with problems or successfully
        branch_define_create_backup_status_task = BranchPythonOperator(
            task_id=f'define_create_backup_status_{db_name}',
            python_callable=is_there_a_problem_with_backup_creating,
            op_kwargs={'db_name': db_name},
            provide_context=True,
            dag=dag_create_backup
        )
        branch_is_there_a_problem_with_create_backup_tasks.append(branch_define_create_backup_status_task)

        #  - 3. telegram operator for sending notifications if there are problems in `create backup process`
        notify_about_create_backup_problem_task = TelegramOperator(
            task_id=f'failed_create_backup_{db_name}__notify',
            token=os.getenv('TOKEN_TELEGRAM_ALERT_BOT'),
            chat_id=os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT'),
            text="{{ task_instance.xcom_pull(task_ids='define_create_backup_status_' + params.db_name, key='error__creating_backup_process_message') }}",
            params={
                'db_name': db_name
            },
            dag=dag_create_backup
        )
        notify_about_create_backup_problem_tasks.append(notify_about_create_backup_problem_task)

        #  - 4. empty operator if the `create backup process` finished successfully
        no_notification_create_backup_task = TelegramOperator(
            task_id=f'success_create_backup_{db_name}__no_notify',
            token=os.getenv('TOKEN_TELEGRAM_ALERT_BOT'),
            chat_id=os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT'),
            text="{{ task_instance.xcom_pull(task_ids='define_create_backup_status_' + params.db_name, key='success__creating_backup_process_message') }}",
            params={
                'db_name': db_name
            },
            dag=dag_create_backup
        )
        no_notify_create_backup_tasks.append(no_notification_create_backup_task)

        for i in range(len(create_backup_tasks)):
            create_backup_tasks[i] >> branch_is_there_a_problem_with_create_backup_tasks[i] >> [notify_about_create_backup_problem_tasks[i], no_notify_create_backup_tasks[i]]
            time.sleep(0.1)