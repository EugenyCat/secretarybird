from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.telegram.operators.telegram import TelegramOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.ETLManager import ETLManager
from pipeline.etl.ETLBinance import ETLBinance


binance_api = 'binance_api'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


def run_etl(etl_instance, start, end, currency, interval):
    """
        Run ETL process for data from BinanceAPI
    """

    return etl_instance.run_etl({
        'currency': currency,
        'interval': interval
    })


def is_there_a_problem_with_etl(**kwargs):
    """
        Branch function to NOTIFY/NO NOTIFY when there are problems with `ETL process`
    """
    task_instance = kwargs['ti']
    message, error = task_instance.xcom_pull(task_ids=f'run_etl_{kwargs["currency"]}_{kwargs["interval"]}')
    if error:
        if not error['message']:
            text = f'Something is wrong with dag_etl_{kwargs["currency"]}_{kwargs["interval"]}'
        else:
            text = error['message']
        task_instance.xcom_push(key='error__etl_process_message', value=text)
        return f'failed_etl_{kwargs["currency"]}_{kwargs["interval"]}__notify'
    else:
        return f'success_etl_{kwargs["currency"]}_{kwargs["interval"]}__no_notify'



with DAG(
        'BinanceAPI_ETL_part_2_etl',
        default_args=default_args,
        description='DAG for ETL process from Binance. STEP 2 - ETL process',
        schedule_interval=None,  # This DAG is triggered, not scheduled
        catchup=False,
    ) as dag_binance_etl:

    # ---STEP 2--- Set up ETL Binance tasks

    # Get the configurations for ETL (start, end, currency, interval)
    etl_process_manager = ETLManager(binance_api, os.getenv('PREFIX_CRYPTO_CURRENCIES'))
    elt_config = etl_process_manager.get_api_configurations()

    # Init lists for storing tasks for:
    # 1. PythonOperator for `etl` tasks
    etl_binance_tasks = []
    # 2. BranchPythonOperator for defining if the `etl process` finished with problems or successfully
    branch_is_there_a_problem_with_etl_tasks = []
    # 3. TelegramOperator for sending notifications if there are problems
    notify_about_etl_problem_tasks = []
    # 4. EmptyOperator if the `etl process` finished successfully
    no_notify_etl_tasks = []

    # Iterate through 'currency'+'interval' and init the tasks for
    #  - 1. etl process (by calling run_etl)
    #  - 2. branch check if the `etl process` finished with problems or successfully
    #  - 3. telegram operator for sending notifications if there are problems
    #  - 4. empty operator if the `etl process` finished successfully
    for currency in elt_config['currency']:
        for interval in elt_config['interval']:

            etl_instance = ETLBinance()

            # - 1. `etl` task
            etl_task = PythonOperator(
                task_id=f'run_etl_{currency}_{interval}',
                python_callable=etl_instance.run_etl,
                op_kwargs={
                    'input_params': {
                        'currency': currency,
                        'interval': interval
                    }
                },
                dag=dag_binance_etl
            )

            etl_binance_tasks.append(etl_task)

            #  - 2. branch check if the `etl process` finished with problems or successfully
            branch_define_etl_status_task = BranchPythonOperator(
                task_id=f'define_etl_status_{currency}_{interval}',
                python_callable=is_there_a_problem_with_etl,
                op_kwargs={
                    'currency': currency,
                    'interval': interval
                },
                provide_context=True,
                dag=dag_binance_etl
            )
            branch_is_there_a_problem_with_etl_tasks.append(branch_define_etl_status_task)

            #  - 3. telegram operator for sending notifications if there are problems in `ETL process`
            notify_about_etl_problem_task = TelegramOperator(
                task_id=f'failed_etl_{currency}_{interval}__notify',
                token=os.getenv('TOKEN_TELEGRAM_ALERT_BOT'),
                chat_id=os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT'),
                text="{{ task_instance.xcom_pull(task_ids='define_etl_status_' + params.currency + '_' + params.interval, key='error__etl_process_message') }}",
                params={
                    'currency': currency,
                    'interval': interval
                },
                dag=dag_binance_etl
            )
            notify_about_etl_problem_tasks.append(notify_about_etl_problem_task)

            #  - 4. empty operator if the `etl process` finished successfully
            no_notification_etl_task = EmptyOperator(
                task_id=f'success_etl_{currency}_{interval}__no_notify',
                dag=dag_binance_etl
            )
            no_notify_etl_tasks.append(no_notification_etl_task)



    control_point = EmptyOperator(
        task_id=f'control_point',
        dag=dag_binance_etl,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    trigger_dag_check_data = TriggerDagRunOperator(
        task_id='trigger_dag_BinanceAPI_ETL_part_3_check_data',
        trigger_dag_id='BinanceAPI_ETL_part_3_check_data',  # the DAG ID for next running
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    for i in range(len(etl_binance_tasks)):
        etl_binance_tasks[i] >> branch_is_there_a_problem_with_etl_tasks[i] >> [notify_about_etl_problem_tasks[i], no_notify_etl_tasks[i]]
        time.sleep(0.05)
        notify_about_etl_problem_tasks[i] >> control_point
        no_notify_etl_tasks[i] >> control_point

    control_point >> trigger_dag_check_data