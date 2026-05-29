FROM python:3.12-slim

RUN pip install --no-cache-dir uv && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the GUI dependencies (ml-legacy group)
COPY pyproject.toml uv.lock* ./
RUN uv sync --group ml-legacy --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "multiverse/gui.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
