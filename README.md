# 🚀 **Model Deployment Pipeline**

This repository contains a Machine Learning pipeline orchestrated using Apache Airflow that automates model training, experiment tracking using MLflow, model artifact storage in MinIO, and deployment of the trained model as a FastAPI service. Additionally, it provides CI/CD automation for deploying Docker images to Kubernetes, and monitoring with Prometheus, InfluxDB, and Grafana.

## 🔥 **Project Overview**

The primary objectives of this project are:
- 🏋️‍♂️ Train a **Random Forest** model on the **Iris dataset**.
- 📊 Track model experiments using **MLflow**.
- 🗄️ Store model artifacts in **MinIO** (local S3-compatible storage).
- 🌐 Deploy the trained model as a **FastAPI** service.
- 🛠️ Automate deployment using **Docker** and **Kubernetes**.
- 📉 Set up monitoring and alerting with **Prometheus**, **InfluxDB**, and **Grafana**.

## ✨ **Key Features**
- 🔄 **Automated Model Training:** The training pipeline retrains the model daily, and the results are logged to MLflow.
- 📝 **Experiment Tracking:** MLflow keeps track of experiment runs, including hyperparameters and accuracy metrics.
- 🚀 **Model Deployment:** The model is deployed via a FastAPI service that provides predictions via an API.
- 🤖 **CI/CD Pipeline:** Automatically builds and runs a Airflow DAG to test model training and building and deploying Docker image with Fast API server.
- 📊 **Monitoring:** Model accuracy metrics and hyperparameters are logged to InfluxDB, enabling visualization in Grafana dashboards.

## 🏗️ **Workflow Architechture**

![workflow](https://github.com/supreetshm947/VirtualMindsTask_Airflow/blob/main/workflow.png)

## 🛠️ **Setup Instructions**

1. **Prerequisites**
- 🐳 Docker and Docker Compose installed.
- ☸️ Minikube for Kubernetes orchestration. You can install it with below commands
  
   ```bash
   make setup_minikube
   ```
   
- 🚀 Conda/Python Environment

2. **One time Setup**
-  Install Python Requirements.
  
```bash
   make setup_minikube
```
 
- Config Airflow
  
```bash
   make setup_airflow
```

3. **Starting Project**
- Create MLFlow experiment, start Docker containers, build minio bucket and start Minikube
  ```bash
   make run
  ```
  
- Start Airflow and unpause all DAGs
  ```bash
   make start_airflow
   make unpause_dag
  ```

4. **Access the FastAPI Model**
   After deployment, the model API can be accessed at:
   ```bash
   curl -X 'GET' "http://<MINIKUBE_IP>:30000/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2" -H 'accept: application/json'
   ```

   Sample Response:
   ![Sample API call](https://github.com/supreetshm947/VirtualMindsTask_Airflow/blob/main/api_minkube.PNG)

## 🛠️ **CI/CD Pipeline**

The pipeline is triggered automatically whenever:

  There are changes to the repository code.

Pipeline Components

  🛠️ GitHub Actions: The code is built and the Model is trained and delpoyed as a FastAPI server.

## 📈 **Monitoring and Alerts**

  - 📡 Prometheus: Collects model performance metrics.
  - 📊 InfluxDB: Stores metrics related to model accuracy and hyperparameters.
  - 📉 Grafana: Visualizes metrics for real-time monitoring. Can be accessed at http://localhost:3000

![Sample Grafana Dashboard](https://github.com/supreetshm947/VirtualMindsTask_Airflow/blob/main/grafana_dashboard.jpg)

Alerts are configured in Prometheus to notify you of performance degradation or other issues via Slack integration.

![Sample Slack Alert](https://github.com/supreetshm947/VirtualMindsTask_Airflow/blob/main/slack_alert.jpg)

