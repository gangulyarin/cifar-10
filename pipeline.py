import airflow
from airflow.models import DAG 
from airflow.operators.python_operator import PythonOperator 


import sys
sys.path.insert(0,'/home/arindam/ai-ml/ImageClassification')

from inital_model_functions import load_preprocess,fit_model

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(1),      # this in combination with catchup=False ensures the DAG being triggered from the current date onwards along the set interval
    'provide_context': True,                            # this is set to True as we want to pass variables on from one task to another
}

dag = DAG(
    dag_id='inital_model_dag', 
    default_args=args, 
    schedule_interval='@once',
    catchup=False
)

task1 = PythonOperator(
    task_id='load_preprocess',
    python_callable=load_preprocess,
    dag=dag
)

task2 = PythonOperator(
    task_id='fit_model',
    python_callable=fit_model,
    dag=dag
)

task1 >> task2
