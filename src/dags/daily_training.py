from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.dates import days_ago
import os
import random
from datetime import timedelta
from src.utils import read_csv
from src.model_generation import train_random_forest
from metrics import ACCURACY_GAUGE, PUSHGATEWAY_URL, registry
from prometheus_client import push_to_gateway

# Create default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,  # Corresponds to RetryPolicy
    'retry_delay': timedelta(seconds=10),  # Retry delay of 10 seconds
}

with DAG(
    dag_id=os.getenv("DAILY_TRAINING_JOB_NAME"),
    default_args=default_args,
    description='Daily job to ingest data and train a model',
    schedule_interval="*/2 * * * *",
    start_date=days_ago(1),
    catchup=False,
) as dag:

    def ingest_data(**context):
        data_path = os.getenv("DATA_PATH")
        data = read_csv(data_path)

        print("Data Ingested")
        return data

    def train_model(**context):
        ingest_data = context['ti'].xcom_pull(task_ids='ingest_data')
        print(f"ingested_data:{ingest_data}")
        hyperparameters = {
            'n_estimators': random.randint(1, 2),
            'max_depth': random.randint(1, 5),
            'min_samples_split': random.randint(2, 3),
            'min_samples_leaf': random.randint(1, 2),
            'random_state': 42,
        }

        model, accuracy, mlflow_run_name = train_random_forest(ingest_data, os.getenv("MODEL_TARGET_VAR"), hyperparameters)

        dag_id = context['dag'].dag_id
        run = context['ti'].dag_run
        print(f"The accuracy of the trained model {mlflow_run_name} is: {accuracy}")
        ACCURACY_GAUGE.labels(job_name=dag_id).set(accuracy)
        push_to_gateway(PUSHGATEWAY_URL, job='accuracy', registry=registry)
        accuracy_threshold = os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD")
        if accuracy < float(os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD")):
            print(
                f"The accuracy of trained model {mlflow_run_name} is below the minimum accuracy threshold of {accuracy_threshold}.")
            raise Exception(f"Accuracy of {mlflow_run_name} too low.")


    trigger_next_dag = TriggerDagRunOperator(
        task_id='trigger_deploy_dag',
        trigger_dag_id=os.getenv("DEPLOY_MODEL_JOB_NAME"),
        wait_for_completion=True
    )

    ingest_data_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
        sla=timedelta(seconds=20)
    )



    ingest_data_task >> train_model_task >> trigger_next_dag
