FROM mambaorg/micromamba:2.3.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/environment-cobolt.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_cobolt/bin:$PATH

COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.models.cobolt", "--config_path", "./config_alldatasets.json"]
