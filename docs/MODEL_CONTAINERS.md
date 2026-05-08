# Model Containers

This document describes the current container build/run contract for Multi-verse models.

## Build Locations

Each built-in model package lives under:

- `store/models/<slug>/model.yaml`
- `store/models/<slug>/container/Dockerfile`

You can build individual images from the repository root with Make targets:

```bash
make build-pca
make build-mofa
make build-multivi
make build-mowgli
make build-cobolt
make build-totalvi
```

Or build all:

```bash
make build-all
```

## Runtime Contract (Zero-Path)

Containers are expected to use fixed in-container paths:

- Input (read-only): `/input/data.h5mu`
- Output directory (read-write): `/output/`
- Job specification (JSON): `/output/job_spec.json`

Required outputs written by the model:

- `/output/embeddings.h5`
- `/output/metrics.json`

## Manual Run (Debug)

Although the orchestrator handles normal execution, you can run manually for debugging:

```bash
docker run --rm \
  -v /abs/path/to/data.h5mu:/input/data.h5mu:ro \
  -v /abs/path/to/output_dir:/output:rw \
  multiverse-pca:1.0.0
```

The container reads `/output/job_spec.json` for runtime parameters. For manual runs, create this file in the mounted output directory before starting the container.
