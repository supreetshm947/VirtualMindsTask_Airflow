from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from src.utils import get_latest_mlflow_run, download_model_artifact, build_image, push_docker_image_to_docker_hub, \
    restart_kubernetes_deployment, register_model_to_production
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(seconds=10),
}

with DAG(
        dag_id=os.getenv("DEPLOY_MODEL_JOB_NAME"),
        default_args=default_args,
        description='This DAG is triggered by the daily_training DAG',
        schedule_interval=None,
        start_date=days_ago(1),
        catchup=False,
) as dag:
    def get_latest_run(**context):
        metric = os.getenv("MLFLOW_METRIC_ACCURACY")
        latest_run = get_latest_mlflow_run()
        print(
            f"Selecting Model {latest_run.info.run_name} with run ID {latest_run.info.run_id} and accuracy:{metric}")

        return latest_run.info.run_id


    def build_docker_image(**context):
        latest_run_id = context['ti'].xcom_pull(task_ids='get_latest_run')
        artifacts_path = download_model_artifact(latest_run_id)
        image_created = build_image(artifacts_path, os.getenv("DOCKER_IMAGE_NAME"))
        return image_created


    def push_image_to_hub(**context):
        container_pushed = push_docker_image_to_docker_hub(os.getenv("DOCKER_IMAGE_NAME"),
                                                           os.getenv("DOCKER_TAG_NAME"))
        return container_pushed


    def redeploy_pods(**context):
        redeployed = restart_kubernetes_deployment(os.getenv("KUBERNETES_DEPLOYMENT_NAME"))
        return redeployed


    def register_run_to_production(**context):
        latest_run_id = context['ti'].xcom_pull(task_ids='get_latest_run')
        registered = register_model_to_production(latest_run_id)
        return registered


    get_latest_run_task = PythonOperator(
        task_id='get_latest_run',
        python_callable=get_latest_run,
        provide_context=True
    )

    build_docker_image_task = PythonOperator(
        task_id='build_docker_image',
        python_callable=build_docker_image,
        provide_context=True
    )

    push_image_to_hub_task = PythonOperator(
        task_id='push_image_to_hub',
        python_callable=push_image_to_hub,
        provide_context=True
    )

    redeploy_pods_task = PythonOperator(
        task_id='redeploy_pods',
        python_callable=redeploy_pods,
        provide_context=True
    )

    register_run_to_production_task = PythonOperator(
        task_id='register_run_to_production',
        python_callable=register_run_to_production,
        provide_context=True
    )

    get_latest_run_task >> build_docker_image_task >> push_image_to_hub_task >> redeploy_pods_task >> register_run_to_production_task
