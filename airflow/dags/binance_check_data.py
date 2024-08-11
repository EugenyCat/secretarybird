from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.ETLManager import ETLManager


binance_api = 'binance_api'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


def check_data(manager, currency, interval):
    """
        Check CH db data on:
        - if data has lacks like (1 row: 2024-01-01 09:00:00, 2 row 2024-01-01 12:00:00 - miss 10:00, 11:00)
        - if data is continuous (e.g. all months, days etc. - similar prev. case)
        - if  last data from dt is really today
    """
    return manager.check_db(currency, interval)


def is_there_a_problem_from_checker(**kwargs):
    """
        Branch function to NOTIFY/NO NOTIFY when there are problems with `data from Checker`
    """
    task_instance = kwargs['ti']
    message, error = task_instance.xcom_pull(task_ids=f'check_data_{kwargs["currency"]}_{kwargs["interval"]}')
    if error:
        task_instance.xcom_push(key='error__checking_data_process_message', value=error['message'])
        return f'failed_checking_{kwargs["currency"]}_{kwargs["interval"]}__notify'
    else:
        return f'success_checking_{kwargs["currency"]}_{kwargs["interval"]}__no_notify'


with DAG(
        'BinanceAPI_ETL_part_3_check_data',
        default_args=default_args,
        description='DAG for ETL process from Binance. STEP 3 - checking data',
        schedule_interval=None,  # This DAG is triggered, not scheduled
        catchup=False,
    ) as dag_binance_check_data:

    # ---STEP 3--- Checking data

    # Get the configurations for ETL (start, end, currency, interval)
    etl_process_manager = ETLManager(binance_api, os.getenv('PREFIX_CRYPTO_CURRENCIES'))
    elt_config = etl_process_manager.get_api_configurations()

    # Init lists for storing tasks for:
    # 1. PythonOperator for check data tasks
    check_data_tasks = []
    # 2. BranchPythonOperator for defining if the `check data process` finished with problems or successfully
    branch_is_there_a_problem_with_check_data_tasks = []
    # 3. TelegramOperator for sending notifications if there are problems
    notify_about_check_data_problem_tasks = []
    # 4. EmptyOperator if the `check data process` finished successfully
    no_notify_check_data_tasks = []

    # Iterate through 'currency'+'interval' and init the tasks for
    #  - 1. `check data process` (by calling check_data)
    #  - 2. branch check if the `check data process` finished with problems or successfully
    #  - 3. telegram operator for sending notifications if there are problems
    #  - 4. empty operator if the `check data process` finished successfully
    for currency in elt_config['currency']:
        for interval in elt_config['interval']:

            # - 1. check data task
            check_data_task = PythonOperator(
                task_id=f'check_data_{currency}_{interval}',
                python_callable=check_data,
                op_kwargs={
                    'manager': etl_process_manager,
                    'currency': currency,
                    'interval': interval
                },
            )
            check_data_tasks.append(check_data_task)

            #  - 2. branch check if the `check data process` finished with problems or successfully
            branch_define_etl_status_task = BranchPythonOperator(
                task_id=f'define_checking_status_{currency}_{interval}',
                python_callable=is_there_a_problem_from_checker,
                op_kwargs={
                    'currency': currency,
                    'interval': interval
                },
                provide_context=True,
                dag=dag_binance_check_data
            )
            branch_is_there_a_problem_with_check_data_tasks.append(branch_define_etl_status_task)

            #  - 3. telegram operator for sending notifications if there are problems in `check data process`
            notify_about_check_data_problem_task = TelegramOperator(
                task_id=f'failed_checking_{currency}_{interval}__notify',
                token=os.getenv('TOKEN_TELEGRAM_ALERT_BOT'),
                chat_id=os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT'),
                text="{{ task_instance.xcom_pull(task_ids='define_checking_status_' + params.currency + '_' + params.interval, key='error__checking_data_process_message') }}",
                params={
                    'currency': currency,
                    'interval': interval
                },
                dag=dag_binance_check_data
            )
            notify_about_check_data_problem_tasks.append(notify_about_check_data_problem_task)

            #  - 4. empty operator if the `check data process` finished successfully
            no_notification_check_data_task = EmptyOperator(
                task_id=f'success_checking_{currency}_{interval}__no_notify',
                dag=dag_binance_check_data
            )
            no_notify_check_data_tasks.append(no_notification_check_data_task)

    control_point = EmptyOperator(
        task_id=f'control_point',
        dag=dag_binance_check_data,
        trigger_rule=TriggerRule.NONE_FAILED
    )
    """
    trigger_dag_create_backup = TriggerDagRunOperator(
        task_id='trigger_dag_BinanceAPI_ETL_part_4_create_backup',
        trigger_dag_id='BinanceAPI_ETL_part_4_create_backup',
        trigger_rule=TriggerRule.NONE_SKIPPED
    )
    """

    for i in range(len(check_data_tasks)):
        check_data_tasks[i] >> branch_is_there_a_problem_with_check_data_tasks[i] >> [notify_about_check_data_problem_tasks[i], no_notify_check_data_tasks[i]]
        if i % 3 == 0:
            time.sleep(0.05)
        notify_about_check_data_problem_tasks[i] >> control_point
        no_notify_check_data_tasks[i] >> control_point