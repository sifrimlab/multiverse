FROM mambaorg/micromamba:2.3.0

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org')" \
    && R -e "remotes::install_github('theislab/kBET')"

WORKDIR /app

COPY multiverse ./multiverse
COPY docker-env/environment-evaluation.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_evaluation/bin:$PATH

COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.evaluate", "--config_path", "./config_alldatasets.json"]
