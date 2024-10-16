from prometheus_client import CollectorRegistry, Gauge
import os

registry = CollectorRegistry()

PUSHGATEWAY_URL = f"http://{os.getenv('PUSHGATEWAY_HOST')}:{os.getenv('PUSHGATEWAY_PORT')}/"

DURATION_GAUGE = Gauge('run_duration', 'Duration of successful Dagster runs in seconds', ['job_name'], registry=registry)
TIMEOUT_ALERT_GAUGE = Gauge('no_run_since_timeout', 'Number of timeout alerts', ['job_name'], registry=registry)
ACCURACY_GAUGE = Gauge('model_accuracy', 'Model accuracy', ['job_name'], registry=registry)