from fastapi import FastAPI
from prometheus_client import make_asgi_app
import uvicorn
import threading
import os
import socket
# from metrics import SUCCESS_COUNTER, FAILURE_COUNTER, DURATION_GAUGE, TIMEOUT_ALERT_COUNTER


_fastapi_server_started = False

app = FastAPI()

prometheus_app = make_asgi_app()

app.mount("/metrics", prometheus_app)

@app.get("/")
async def root():
    return {"message": "Welcome to the metrics server!"}

def start_metrics_server():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("METRICS_PORT")))

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0

def run_in_background():
    # To ensure it doesn't restart the web server as for some reason dagster re-executes the code over and over again.
    if not is_port_in_use(int(os.getenv("METRICS_PORT"))):
        server_thread = threading.Thread(target=start_metrics_server)
        server_thread.daemon = True
        server_thread.start()
