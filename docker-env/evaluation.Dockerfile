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

COPY docker-env/environment-evaluation.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_evaluation/bin:$PATH

# Install the multiverse package so multiverse.evaluate and multiverse.worker
# (plus the [eval] scientific deps) are importable. The heavy stack is pinned
# by environment-evaluation.yml above; the [eval] extra fills in the rest.
COPY pyproject.toml README.md /tmp/multiverse/
COPY multiverse/ /tmp/multiverse/multiverse/
RUN pip install "/tmp/multiverse[eval]"

# The host's containerized evaluation runner mounts a launch cohort and passes
# its path as --config_path. CMD is a harmless default for bare `docker run`.
ENTRYPOINT ["python", "-m", "multiverse.evaluate"]
CMD ["--help"]
