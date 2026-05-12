FROM mambaorg/micromamba:2.3.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY store/models/cobolt/container/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_cobolt/bin:$PATH

COPY sdk/mvr-worker/ /tmp/mvr-worker/
RUN pip install /tmp/mvr-worker/

COPY store/models/cobolt/container/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
