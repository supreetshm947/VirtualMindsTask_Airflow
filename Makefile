# Load environment variables from .env file
include .env

# Export environment variables
export $(shell sed 's/=.*//' .env)


MC_ALIAS=myminio
CONTAINER_NAME=minio_virtual_minds

# Create Experiment in MLFlow and set artifact location to s3
create-experiment:
	@echo "Checking if experiment '$(MLFLOW_EXPERIMENT_NAME)' exists..."
	@if mlflow experiments search | grep -q "$(MLFLOW_EXPERIMENT_NAME)"; then \
		echo "Experiment '$(MLFLOW_EXPERIMENT_NAME)' already exists."; \
	else \
		echo "Creating experiment '$(MLFLOW_EXPERIMENT_NAME)'..."; \
		mlflow experiments create -n $(MLFLOW_EXPERIMENT_NAME) --artifact-location s3://$(MLFLOW_BUCKET)/models; \
		echo "Experiment '$(MLFLOW_EXPERIMENT_NAME)' created."; \
	fi

create-bucket:
	@echo "Configuring MinIO client in the Docker container..."
	@docker exec $(CONTAINER_NAME) mc alias set $(MC_ALIAS) http://$(MINIO_HOST):$(MINIO_API_PORT) $(MINIO_USER) $(MINIO_PASSWORD)
	@echo "Checking if bucket '$(MLFLOW_BUCKET)' exists..."
	@if docker exec $(CONTAINER_NAME) mc ls $(MC_ALIAS) | grep -q "$(MLFLOW_BUCKET)"; then \
		echo "Bucket '$(MLFLOW_BUCKET)' already exists."; \
	else \
		echo "Creating bucket '$(MLFLOW_BUCKET)'..."; \
		docker exec $(CONTAINER_NAME) mc mb $(MC_ALIAS)/$(MLFLOW_BUCKET); \
		echo "Bucket '$(MLFLOW_BUCKET)' created."; \
	fi

# set permissions to docker volumes
set_volume_permissions:
	mkdir -p ./grafana_data
	sudo chmod -R 777 ./grafana_data

# Run docker & create minio bucket
run: create-experiment set_volume_permissions up create-bucket start_minikube

run_git: create-experiment up create-bucket

# Start docker containers
up:
	@echo "Starting Docker containers..."
	docker compose up -d
	@echo "Waiting for containers to start..."
	@sleep 20

# Stop docker containers
down:
	@echo "Stopping Docker containers..."
	docker compose down

run_mlflow_tracking:
	@echo "Starting MLFlow Tracking server..."
	@AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) mlflow server --default-artifact-root s3://$(MLFLOW_BUCKET)/models --host localhost --port 4144


init_db:
	airflow db upgrade
	airflow db reset --yes
	airflow db init

create_user:
	airflow users create \
		--username $(AIRFLOW_USER_NAME) \
		--firstname $(AIRFLOW_USER_FIRSTNAME) \
		--lastname $(AIRFLOW_USER_LASTNAME) \
		--role Admin \
		--email $(AIRFLOW_USER_EMAIL) \
		--password $(AIRFLOW_USER_PASSWORD)

setup_airflow: init_db create_user
	@echo "Airflow setup complete."

# Targets for starting Airflow
start_airflow_webserver:
	airflow webserver --port 8080 &

start_airflow_scheduler:
	airflow scheduler &

start_airflow: start_airflow_webserver start_airflow_scheduler
	@echo "Airflow is running."

unpause_dag:
	airflow dags unpause daily_training
	airflow dags unpause deploy_model

kill_airflow:
	pkill -f dagster

start_test_pipeline_dag:
	dotenv -f .env run airflow dags test $(TEST_TRAINING_PIPELINE_JOB) $(date -u +"%Y-%m-%dT%H:%M:%S")

install_minikube:
	echo "Installing Minikube"
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube
	sudo rm -f minikube-linux-amd64
install_kubectl:
	echo "Installing Kubectl"
	sudo apt-get install -y apt-transport-https ca-certificates
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
	sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
	echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
	sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list
	sudo apt update
	sudo apt-get install -y kubectl
setup_minikube:
	install_minikube install_kubectl
start_minikube:
	minikube start --driver=docker
	kubectl apply -f deployment.yaml
	kubectl apply -f service.yaml

start_minikube_service:
	kubectl port-forward deployment/mlflow-model-deployment 8000:8000

get_pods:
	kubectl get pods

# eval $$(minikube -p minikube docker-env)
# kubectl get pods
# kubectl describe pod
# minikube ip
# curl -X 'GET' "http://192.168.49.2:30000/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2" -H 'accept: application/json'


# dotenv -f .env run airflow dags test training_timeout 2024-10-16T00:00:00