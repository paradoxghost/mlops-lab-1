FROM python:3.13.14-slim-bookworm
RUN pip install --no-cache-dir mlflow==3.16.1
ENV MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
WORKDIR /mlflow-data
EXPOSE 5000
# Proxy artifact downloads over HTTP: other containers cannot read this volume.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", "--default-artifact-root", "mlflow-artifacts:/", \
     "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow:*,localhost:*,127.0.0.1:*"]
