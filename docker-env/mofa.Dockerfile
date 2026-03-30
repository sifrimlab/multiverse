FROM cupy/cupy

WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/requirements-mofa.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY config_alldatasets.json .

ENTRYPOINT ["python3", "-m", "multiverse.models.mofa",  "--config_path", "./config_alldatasets.json"]