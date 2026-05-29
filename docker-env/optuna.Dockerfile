FROM python:3.12-slim

RUN pip install --no-cache-dir optuna optuna-dashboard && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY docker-env/entrypoint-optuna.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
