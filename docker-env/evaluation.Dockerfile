FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Install R and basic system dependencies
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
COPY docker-env/requirements-evaluation.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config_alldatasets.json .

ENTRYPOINT ["python", "-m", "multiverse.evaluate", "--config_path", "./config_alldatasets.json"]