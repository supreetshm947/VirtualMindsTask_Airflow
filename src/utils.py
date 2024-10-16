import pandas as pd
from joblib import Memory
from mlflow import MlflowClient
import mlflow
import os
import subprocess
import shutil
from influxdb import InfluxDBClient
from typing import Dict
import docker
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from docker.errors import BuildError, APIError, DockerException
from datetime import datetime

cache_path = "cache"
memory = Memory(cache_path)


@memory.cache
def read_csv(path):
    return pd.read_csv(path)


def get_latest_mlflow_run():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    experiment_id = experiment.experiment_id

    best_run = None
    metric = os.getenv("MLFLOW_METRIC_ACCURACY")
    threshold = float(os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD"))

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["end_time DESC"],
    )


    print(f"{len(runs)} found at MLFlow")

    for run in runs:
        if run.data.metrics.get(metric) >= threshold:
            best_run = run
            break

    return best_run


def download_model_artifact(run_id, local_dir="tmp_artifact_dir"):
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    os.makedirs(local_dir)

    if not run_id:
        print("No suitable run found.")
        return None

    artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=os.getenv("MLFLOW_MODEL_NAME"),
                                                         dst_path=local_dir)

    print(f"Model downloaded to {artifacts_path}")

    return artifacts_path


def build_image(artifacts_path, image_name, code_path="src/model_deployment", dockerfile_path="."):
    image_volume_dir = "image_volume"
    if os.path.exists(image_volume_dir):
        shutil.rmtree(image_volume_dir)

    os.makedirs(image_volume_dir)

    # copy artifacts and python code to image_volume_dir
    for item in os.listdir(artifacts_path):
        shutil.copy(os.path.join(artifacts_path, item), image_volume_dir)
    for item in os.listdir(code_path):
        shutil.copy(os.path.join(code_path, item), image_volume_dir)

    client = docker.from_env()

    print(f"Building Docker image: {image_name}...")
    try:
        image, logs = client.images.build(path=dockerfile_path, tag=image_name)

        for log in logs:
            if 'stream' in log:
                print(log['stream'].strip())

        print(f"Docker image: {image_name}, built successfully.")
        return True

    except BuildError as build_err:
        print("BuildError occurred: %s" % build_err)
        return False

    except APIError as api_err:
        print("APIError occurred: %s" % api_err)
        return False
    finally:
        client.close()

def containerize_docker_image(context, image_name, container_name):
    local_port = 8000
    container_port = 8000

    # Docker run command
    command = [
        "docker", "run",
        "--name", container_name,
        "-p", f"{local_port}:{container_port}",
        "-d",
        image_name
    ]

    try:
        subprocess.run(command, check=True)
        context.log.info(f"Successfully deployed {container_name} from image {image_name} on port {local_port}.")
    except subprocess.CalledProcessError as e:
        context.log.info(f"Failed to deploy the container: {e}")


def push_docker_image_to_docker_hub(image_name, tag_name):
    client = docker.from_env()

    try:
        print(f"Tagging image {image_name} as {tag_name}")
        image = client.images.get(image_name)
        image.tag(tag_name)
        print(f"Image {image_name} tagged as {tag_name}")

        print(f"Pushing image {tag_name} to Docker Hub...")
        for line in client.images.push(tag_name, stream=True, decode=True):
            print(line)

        print("Docker image pushed to Docker Hub successfully!")
        return True

    except DockerException as e:
        print(f"Error while tagging/pushing Docker image: {str(e)}")
        return False
    finally:
        client.close()

def restart_kubernetes_deployment(deployment_name, namespace="default"):
    config.load_kube_config()

    api_instance = client.AppsV1Api()

    try:
        print(f"Restarting Kubernetes deployment '{deployment_name}' in namespace '{namespace}'...")

        deployment = api_instance.read_namespaced_deployment(name=deployment_name, namespace=namespace)

        deployment.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] = str(datetime.utcnow())
        api_instance.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=deployment)

        print("Kubernetes deployment restarted successfully.")
        return True

    except ApiException as e:
        print(f"Failed to restart Kubernetes deployment: {e.reason}")
        return False

def register_model_to_production(latest_run_id):
    client = MlflowClient()
    model_uri = ""

    if latest_run_id:
        model_uri = f"runs:/{latest_run_id}/model"
        model_name = os.getenv("MLFLOW_MODEL_NAME")

        # Register model
        mlflow.register_model(model_uri, model_name)

        current_version_info = client.get_latest_versions(model_name, stages=["None"])

        latest_version_info = client.get_latest_versions(model_name, stages=["Production"])

        new_version = current_version_info[-1].version
        if latest_version_info:
            latest_version = latest_version_info[0].version  # Assuming the first one is the latest

            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage="Archived"
            )
            print(f"Previous Production model version {latest_version} archived")

        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
        print(f"Model version {new_version} transitioned to 'Production'")

    return model_uri


def log_to_influxdb(run_id: str, run_name: str, accuracy: float, hyperparams: Dict[str, int]):
    client = InfluxDBClient(host=os.getenv("INFLUX_DB_HOST"), port=os.getenv("INFLUX_DB_PORT"),
                            username=os.getenv("INFLUXDB_ADMIN_USER"), password=os.getenv("INFLUXDB_ADMIN_PASSWORD"),
                            database=os.getenv("INFLUXDB_DB"))

    json_body = [
        {
            "measurement": "model_performance",
            "tags": {
                "run_id": run_id,
                "run_name": run_name
            },
            "fields": {
                "accuracy": accuracy,
                **hyperparams
            }
        }
    ]

    client.write_points(json_body)


if __name__ == "__main__":
    os.environ['MLFLOW_EXPERIMENT_NAME'] = os.getenv('MLFLOW_EXPERIMENT_NAME', 'MyExperiment')
    os.environ["MLFLOW_METRIC_ACCURACY"] = os.getenv("MLFLOW_METRIC_ACCURACY", 'accuracy')
    os.environ["MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD"] = os.getenv("MLFLOW_DEPLOYMENT_ACCURACY_THRESHOLD", '0.5')
    os.environ["MLFLOW_MODEL_NAME"] = os.getenv("MLFLOW_MODEL_NAME", 'sk_models')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'user')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'password')
    os.environ['MLFLOW_BUCKET'] = os.getenv('MLFLOW_BUCKET', 'mlflow')
