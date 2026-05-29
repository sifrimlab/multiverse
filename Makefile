# Makefile for the Multi-verse project

# Keep state in the project directory for local development.
# MVEXP_STATE_DIR can be overridden from the environment to point elsewhere.
export MVEXP_STATE_DIR ?= $(CURDIR)

# --- Dependency Management ---

.PHONY: install
install:
	@echo "Installing dependencies using uv..."
	uv sync --group dev

.PHONY: init
init:
	@echo "Initializing registry state..."
	uv run multiverse init-db

.PHONY: setup
setup:
	@echo "Installing GUI/local-runner dependencies using uv (dev + ml-legacy)..."
	uv sync --group dev --group ml-legacy

.PHONY: gui
gui:
	@echo "Starting Multiverse GUI (Streamlit)..."
	uv run python -m streamlit run multiverse/gui.py

# bootstrap: one-shot first-run initialisation after git clone.
# Installs deps, creates the SQLite registry, and registers all built-in models.
# Assumes Docker images for built-in models are already present (pull or build separately).
.PHONY: bootstrap
bootstrap: install init register-models
	@echo ""
	@echo "Bootstrap complete."
	@echo "  Next steps:"
	@echo "    make register-all-datasets   # if datasets already exist in store/datasets/"
	@echo "    make services-up             # start MLflow + Optuna Dashboard"
	@echo "    make setup                   # install GUI/local-runner extras"
	@echo "    make gui                     # launch the Streamlit GUI"

# register-all-datasets: batch-register every dataset.yaml found under store/datasets/.
# Safe to re-run; uses --update to refresh existing registry rows.
.PHONY: register-all-datasets
register-all-datasets:
	@echo "Scanning store/datasets/ for dataset.yaml manifests..."
	@found=0; \
	for yaml in store/datasets/*/dataset.yaml; do \
		[ -f "$$yaml" ] || continue; \
		slug=$$(basename $$(dirname "$$yaml")); \
		echo "  → registering '$$slug'"; \
		uv run multiverse register-dataset --slug "$$slug" --update || true; \
		found=$$((found + 1)); \
	done; \
	echo "Done — $$found dataset(s) processed."

# --- Docker Image Builds ---
# Dockerfiles live under docker-env/; build context is the repository root
# (COPY multiverse, docker-env/environment-*.yml, config_alldatasets.json).
# Images use micromamba + conda env YAML. GPU envs target linux/amd64 (pytorch::pytorch-cuda=12.1).

DOCKER_ENV ?= docker-env
DOCKER_BUILD_FLAGS ?=

.PHONY: build-all
build-all: build-pca build-multivi build-mowgli build-mofa build-cobolt build-totalvi build-evaluate
	@echo "All model and evaluation images built."

.PHONY: build-pca
build-pca:
	@echo "Building PCA image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/pca/container/Dockerfile -t multiverse-pca:1.0.0 .

.PHONY: build-multivi
build-multivi:
	@echo "Building MultiVI image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/multivi/container/Dockerfile -t multiverse-multivi:1.0.0 .

.PHONY: build-mowgli
build-mowgli:
	@echo "Building Mowgli image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/mowgli/container/Dockerfile -t multiverse-mowgli:1.0.0 .

.PHONY: build-mofa
build-mofa:
	@echo "Building MOFA image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/mofa/container/Dockerfile -t multiverse-mofa:1.0.0 .

.PHONY: build-cobolt
build-cobolt:
	@echo "Building Cobolt image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/cobolt/container/Dockerfile -t multiverse-cobolt:1.0.0 .

.PHONY: build-totalvi
build-totalvi:
	@echo "Building TotalVI image..."
	docker build $(DOCKER_BUILD_FLAGS) -f store/models/totalvi/container/Dockerfile -t multiverse-totalvi:1.0.0 .

.PHONY: build-evaluate
build-evaluate:
	@echo "Building evaluation image..."
	docker build $(DOCKER_BUILD_FLAGS) -f $(DOCKER_ENV)/evaluation.Dockerfile -t multiverse-evaluate .


# --- Observability Services ---

.PHONY: services-up
services-up:
	@echo "Starting MLflow and Optuna dashboard services..."
	docker compose up -d mlflow optuna-ui
	@echo "MLflow  → http://localhost:$${MLFLOW_PORT:-5000}"
	@echo "Optuna  → http://localhost:$${OPTUNA_PORT:-8080}"

.PHONY: services-down
services-down:
	@echo "Stopping observability services..."
	docker compose down

.PHONY: status
status:
	docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"


# --- Orchestrator Runner ---

# Define some default paths for the runner
# Users should override these as needed
INPUT_DIR ?= ./sample_data/input
OUTPUT_DIR ?= ./results
MODELS_TO_RUN ?= pca multivi mowgli
CONFIG_FILE ?= config_alldatasets.json
MANIFEST ?= run_manifest.yaml

.PHONY: run
run:
	@echo "Running multiverse benchmark from manifest..."
	uv run multiverse run --output $(OUTPUT_DIR) --manifest $(MANIFEST)

.PHONY: benchmark
benchmark:
	@echo "Running Multi-verse benchmark from manifest..."
	uv run multiverse run --output $(OUTPUT_DIR) --manifest $(MANIFEST)

.PHONY: test
test:
	@echo "Running tests..."
	uv run pytest

.PHONY: clean
clean:
	@echo "Cleaning up output directory..."
	rm -rf $(OUTPUT_DIR)

.PHONY: register
register:
	@if [ -n "$(slug)" ]; then \
		echo "Registering dataset slug $(slug)"; \
		uv run multiverse register-dataset --slug "$(slug)"; \
	elif [ -n "$(manifest)" ]; then \
		echo "Registering manifest $(manifest)"; \
		uv run multiverse register-dataset --manifest "$(manifest)"; \
	else \
		echo "Usage: make register slug=<dataset-slug> OR make register manifest=/path/to/dataset.yaml"; \
		exit 1; \
	fi

.PHONY: register-model
register-model:
	@if [ -n "$(slug)" ]; then \
		echo "Registering model slug $(slug)"; \
		uv run multiverse register-model --slug "$(slug)"; \
	elif [ -n "$(manifest)" ]; then \
		echo "Registering model manifest $(manifest)"; \
		uv run multiverse register-model --manifest "$(manifest)"; \
	else \
		echo "Usage: make register-model slug=<model-slug> OR make register-model manifest=/path/to/model.yaml"; \
		exit 1; \
	fi

.PHONY: build-sif
build-sif:
	uv run multiverse build-sif --slug $(slug)

.PHONY: register-models
register-models:
	@echo "Registering all built-in models..."
	uv run multiverse register-model --slug pca
	uv run multiverse register-model --slug mofa
	uv run multiverse register-model --slug multivi
	uv run multiverse register-model --slug mowgli
	uv run multiverse register-model --slug cobolt
	uv run multiverse register-model --slug totalvi
