FROM python:3.12-slim

RUN pip install --no-cache-dir uv && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install GUI dependencies from the project base deps (includes streamlit).
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "multiverse/gui.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
