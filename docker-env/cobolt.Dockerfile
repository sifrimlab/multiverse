FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/requirements-cobolt.txt ./requirements.txt
RUN apt-get update && apt-get install -y git
RUN pip install --no-cache-dir -r requirements.txt
COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.cobolt", "--config_path", "./config_alldatasets.json"]
