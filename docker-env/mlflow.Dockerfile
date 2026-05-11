FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.16.2 && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY docker-env/entrypoint-mlflow.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
