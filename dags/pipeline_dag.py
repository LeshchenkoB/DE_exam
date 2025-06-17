from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.exceptions import AirflowException
from datetime import datetime, timedelta
import logging
from etl.load_data import load_and_save_data
from etl.preprocess_data import preprocess_data
from etl.train_model import train_model
from etl.evaluate_model import evaluate_model
from config import LOGS_DIR

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('airflow_dag')

def failure_callback(context):
    """Обратный вызов при сбое задачи"""
    task_instance = context['task_instance']
    exception = context.get('exception') or "Unknown error"
    logger.error(f"Задача {task_instance.task_id} завершилась с ошибкой: {exception}")

def success_callback(context):
    """Обратный вызов при успешном выполнении задачи"""
    task_instance = context['task_instance']
    logger.info(f"Задача {task_instance.task_id} успешно завершена")

# Общие параметры для задач
default_task_args = {
    'on_failure_callback': failure_callback,
    'on_success_callback': success_callback,
    'retries': 3,  # Количество попыток повтора
    'retry_delay': timedelta(minutes=1),  # Задержка между попытками
    'retry_exponential_backoff': True,  # Экспоненциальное увеличение задержки
    'max_retry_delay': timedelta(minutes=5),  # Максимальная задержка
    'execution_timeout': timedelta(minutes=5),  # Таймаут выполнения
}

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
}

with DAG(
    dag_id='breast_cancer_ml_pipeline',
    default_args=default_args,
    schedule_interval='0 8 * * *',
    catchup=False,
    tags=['ml', 'breast_cancer', 'data_engineering'],
) as dag:
    
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_and_save_data,
        **default_task_args
    )
    
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        **default_task_args
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        **default_task_args
    )
    
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        **default_task_args
    )
    
    # Определение зависимостей
    load_data >> preprocess >> train >> evaluate