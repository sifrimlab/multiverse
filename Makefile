# Makefile for the Multi-verse project

# --- Dependency Management ---

.PHONY: install
install:
	@echo "Installing dependencies using uv..."
	uv sync --group dev

.PHONY: setup
setup: install
	@echo "Starting Multiverse Setup Wizard (Streamlit)..."
	uv run streamlit run multiverse/gui.py

# --- Docker Image Builds ---

.PHONY: build-all
build-all: build-pca build-multivi build-mowgli build-mofa
	@echo "All model images built."

.PHONY: build-pca
build-pca:
	@echo "Building PCA image..."
	docker build -f containers/pca/dockerfile -t multiverse-pca .

.PHONY: build-multivi
build-multivi:
	@echo "Building MultiVI image..."
	docker build -f containers/multivi/dockerfile -t multiverse-multivi .

.PHONY: build-mowgli
build-mowgli:
	@echo "Building Mowgli image..."
	docker build -f containers/mowgli/dockerfile -t multiverse-mowgli .

.PHONY: build-mofa
build-mofa:
	@echo "Building MOFA image..."
	docker build -f containers/mofa/dockerfile -t multiverse-mofa .


# --- Orchestrator Runner ---

# Define some default paths for the runner
# Users should override these as needed
INPUT_DIR ?= ./sample_data/input
OUTPUT_DIR ?= ./results
MODELS_TO_RUN ?= pca multivi mowgli
CONFIG_FILE ?= config_alldatasets.json

.PHONY: run
run:
	@echo "Running Multi-verse pipeline..."
	uv run python runner.py $(CONFIG_FILE)

.PHONY: test
test:
	@echo "Running tests..."
	uv run pytest

.PHONY: clean
clean:
	@echo "Cleaning up output directory..."
	rm -rf $(OUTPUT_DIR)
