FROM mambaorg/micromamba:2.3.0

USER root
WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/environment-totalvi.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_totalvi/bin:$PATH

COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.totalvi", "--config_path", "./config_alldatasets.json"]
