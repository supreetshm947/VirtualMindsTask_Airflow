from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import os
import random
from datetime import timedelta
from src.utils import read_csv, download_model_artifact, build_image, containerize_docker_image
from src.model_generation import train_random_forest

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False
}

with DAG(
    dag_id=os.getenv("TEST_TRAINING_PIPELINE_JOB"),
    default_args=default_args,
    description='Daily job to ingest data and train a model',
    schedule_interval=None,
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
        accuracy_threshold = os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD")
        if accuracy < float(os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD")):
            print(
                f"The accuracy of trained model {mlflow_run_name} is below the minimum accuracy threshold of {accuracy_threshold}.")
            raise Exception(f"Accuracy of {mlflow_run_name} too low.")


    def build_docker_image(**context):
        latest_run_id = context['ti'].xcom_pull(task_ids='get_latest_run')
        artifacts_path = download_model_artifact(latest_run_id)
        image_created = build_image(artifacts_path, os.getenv("DOCKER_IMAGE_NAME"))
        return image_created

    def run_image_as_container(**context):
        containerize_docker_image(os.getenv("DOCKER_IMAGE_NAME"), "deployment_api")

    ingest_data_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )

    build_docker_image_task = PythonOperator(
        task_id='build_docker_image',
        python_callable=build_docker_image,
        provide_context=True
    )

    run_image_as_container_task = PythonOperator(
        task_id='run_image_as_container',
        python_callable=run_image_as_container,
        provide_context=True
    )



    ingest_data_task >> train_model_task >> build_docker_image_task >> run_image_as_container_task
