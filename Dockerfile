FROM python:3.11-slim

WORKDIR /app

COPY image_volume/ /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-server.txt

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]