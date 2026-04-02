FROM mambaorg/micromamba:2.3.0

USER root
WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/environment-multivi.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_multivi/bin:$PATH

COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.multivi", "--config_path", "./config_alldatasets.json"]
