FROM mambaorg/micromamba:2.3.0

USER root
WORKDIR /app

COPY store/models/pca/container/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_pca/bin:$PATH

COPY sdk/mvr-worker/ /tmp/mvr-worker/
RUN pip install /tmp/mvr-worker/

COPY store/models/pca/container/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
