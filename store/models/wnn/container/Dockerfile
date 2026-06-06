FROM mambaorg/micromamba:2.3.0

USER root
WORKDIR /app

COPY store/models/wnn/container/environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml && micromamba clean -afy

ENV PATH=/opt/conda/envs/multiverse_wnn/bin:$PATH

# pyWNN is not on PyPI; install directly from the upstream repository.
RUN git clone https://github.com/dylkot/pyWNN.git

COPY sdk/mvr-worker/ /tmp/mvr-worker/
RUN pip install /tmp/mvr-worker/

COPY store/models/wnn/container/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
