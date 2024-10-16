from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta, timezone
import os
from prometheus_client import push_to_gateway
from metrics import TIMEOUT_ALERT_GAUGE, PUSHGATEWAY_URL, registry
from airflow.models import DagRun

with DAG(
        dag_id=os.getenv("TRAINING_TIMEOUT_DAG"),
        default_args={
            'owner': 'airflow',
            'retries': 3,
            'retry_delay': timedelta(seconds=10),
        },
        description='Sensor to check the timeout between training runs',
        schedule_interval="*/1 * * * *",
        start_date=days_ago(1),
        catchup=False,
) as dag:
    def check_training_timeout(**context):
        job_name = os.getenv("DAILY_TRAINING_JOB_NAME")
        timeout = eval(os.getenv("TRAINING_FRESHNESS_TIMEOUT"))

        dag_runs = DagRun.find(dag_id=job_name)
        dag_runs.sort(key=lambda x: x.execution_date, reverse=True)

        last_successful_run = None
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        print(f"Current time in UTC: {current_time}")

        print(f"{len(dag_runs)} recent runs found.")

        for run in dag_runs:
            if run.state == "success":
                last_successful_run = run
                break

        print(f"Last Execution time: {last_successful_run.execution_date}")

        if last_successful_run is None:
            context['ti'].log.warn(f"No successful run found for job {job_name}.")
        else:
            last_run_time = last_successful_run.execution_date
            time_since_last_run = (current_time - last_run_time).total_seconds()

            if time_since_last_run > timeout:
                context['ti'].log.warn(
                    f"ALERT: {job_name} has not completed successfully in the last {timeout} seconds!"
                )

                TIMEOUT_ALERT_GAUGE.labels(job_name=job_name).set(time_since_last_run)
                push_to_gateway(PUSHGATEWAY_URL, job='timeout', registry=registry)
            context['ti'].log.info(f"It's been {time_since_last_run} seconds since the last run.")


    check_timeout_task = PythonOperator(
        task_id='check_timeout',
        python_callable=check_training_timeout,
        provide_context=True,
    )
