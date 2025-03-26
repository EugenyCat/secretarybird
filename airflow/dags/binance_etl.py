from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, parent_dir)

from pipeline.etl_manager.dataQualityManager import DataQualityManager
from pipeline.etl.etlBinance import ETLBinance

API_NAME = 'binance_api'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}


with DAG(
        'BinanceAPI_ETL_part_2_etl',
        default_args=default_args,
        description='DAG for ETL process from Binance. STEP 2 - ETL process',
        schedule_interval=None,  # This DAG is triggered, not scheduled
        catchup=False,
    ) as dag_binance_etl:

    # ---STEP 2--- Set up ETL Binance tasks

    # Create an instance of ETLBinance to manage the ETL process
    etl_instance = ETLBinance()

    # Get the configurations for ETL (start, end, currency, interval)
    etl_process_manager = DataQualityManager(API_NAME)
    elt_config = etl_process_manager.get_api_configurations()

    # Init the list for storing PythonOperator for `etl` tasks
    etl_binance_tasks = []

    # Iterate through 'currency'+'interval' and init the tasks for etl process (by calling run_etl)
    for currency in elt_config['currency']:
        for interval in elt_config['interval']:

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

    # Empty operator as a control_point
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
        etl_binance_tasks[i] >> control_point

    control_point >> trigger_dag_check_data