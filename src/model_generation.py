import random
from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from src.utils import log_to_influxdb

import mlflow


def train_random_forest(df: pd.DataFrame,
                        target_column: str,
                        hyperparameters: Dict[str, int]):
    """
    Train a Random Forest Classifier on the given dataset.

    Parameters:
        df (pd.DataFrame): The input dataframe containing features and target.
        target_column (str): The name of the target column in the dataframe.
        hyperparameters (dict): A dictionary of hyperparameters for the RandomForestClassifier.

    Returns:
        model: The trained Random Forest model.
        accuracy: The accuracy score of the model on the test set.
    """

    # Separate features and target from the DataFrame
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target

    X = X.to_numpy()

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (optional but recommended for some models, not strictly necessary for Random Forest)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))

    with mlflow.start_run() as run:
        mlflow.log_params(hyperparameters)

        # Create and train the Random Forest Classifier model with hyperparameters
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric(os.getenv("MLFLOW_METRIC_ACCURACY"), accuracy)
        print(f"Successfully Logged metrics to MLFlow.")
        mlflow.sklearn.log_model(model, os.getenv("MLFLOW_MODEL_NAME"))
        print(f"Successfully Logged model artifacts to S3.")
        run_id = run.info.run_id
        run_name = run.info.run_name

        log_to_influxdb(run_id, run_name, accuracy, hyperparameters)
        print(f"Successfully Logged run to InfluxDB.")
        return model, accuracy, run_name


def set_mlflow_environment():
    """
    Set environment variables for MLflow S3 configuration.
    """
    # Load environment variables for MLflow
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'user')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'password')
    os.environ['MLFLOW_BUCKET'] = os.getenv('MLFLOW_BUCKET', 'mlflow')
    os.environ['MLFLOW_METRIC_ACCURACY'] = os.getenv('MLFLOW_METRIC_ACCURACY', 'accuracy')
    os.environ['MLFLOW_MODEL_NAME'] = os.getenv('MLFLOW_MODEL_NAME', 'sk_models')
    os.environ['MLFLOW_EXPERIMENT_NAME'] = os.getenv('MLFLOW_EXPERIMENT_NAME', 'MyExperiment')
    os.environ['INFLUX_DB_HOST'] = os.getenv('INFLUX_DB_HOST', 'localhost')
    os.environ['INFLUX_DB_PORT'] = os.getenv('INFLUX_DB_PORT', '8086')
    os.environ['INFLUXDB_DB'] = os.getenv('INFLUXDB_DB', 'mlflow_db')
    os.environ['INFLUXDB_ADMIN_USER'] = os.getenv('INFLUXDB_ADMIN_USER', 'admin')
    os.environ['INFLUXDB_ADMIN_PASSWORD'] = os.getenv('INFLUXDB_ADMIN_PASSWORD', 'password')


if __name__ == '__main__':
    set_mlflow_environment()
    df = pd.read_csv('data/iris.csv')
    hyperparameters = {
        'n_estimators': random.randint(1, 2),  # Number of trees in the forest
        'max_depth': random.randint(1, 5),
        # Maximum depth of the tree (None means nodes are expanded until all leaves are pure)
        'min_samples_split': random.randint(2, 3),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': random.randint(1, 2),  # Minimum number of samples required to be at a leaf node
        'random_state': 42,  # Controls the randomness of the estimator for reproducibility
    }
    model, accuracy, run_name = train_random_forest(df, 'variety', hyperparameters)
    print("the accuracy of the model is:", accuracy)
