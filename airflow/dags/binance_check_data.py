from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.dataQualityManager import DataQualityManagerFacade


API_NAME = 'binance_api'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


with DAG(
        'BinanceAPI_ETL_part_3_check_data',
        default_args=default_args,
        description='DAG for ETL process from Binance. STEP 3 - checking data',
        schedule_interval=None,  # This DAG is triggered, not scheduled
        catchup=False,
    ) as dag_binance_check_data:

    # ---STEP 3--- Checking data

    # Get the configurations for ETL (start, end, currency, interval)
    etl_process_manager = DataQualityManagerFacade(API_NAME)
    elt_config = etl_process_manager.get_api_configurations()

    # Init the list for storing PythonOperator for check data tasks
    check_data_tasks = []

    # Iterate through 'currency'+'interval' and init the tasks for `check data process` (by calling check_data)
    for currency in elt_config['currency']:
        for interval in elt_config['interval']:

            # - 1. check data task
            check_data_task = PythonOperator(
                task_id=f'check_data_{currency}_{interval}',
                python_callable=etl_process_manager.check_db,
                op_kwargs={
                    'currency': currency,
                    'interval': interval
                },
            )
            check_data_tasks.append(check_data_task)

    control_point = EmptyOperator(
        task_id=f'control_point',
        dag=dag_binance_check_data,
        trigger_rule=TriggerRule.NONE_FAILED
    )

    for i in range(len(check_data_tasks)):
        check_data_tasks[i] >> control_point