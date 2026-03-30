# Model Containers

This document describes how to build and run the individual model containers for the Multi-verse project.

## Building the Images

Each model has its own Dockerfile in the `docker/` directory. To build the image for a specific model, use the `docker build` command from the root of the repository.

For example, to build the PCA model image:
```bash
docker build -f docker/pca.Dockerfile -t multiverse-pca .
```

Similarly, for the other models:
```bash
docker build -f docker/multivi.Dockerfile -t multiverse-multivi .
docker build -f docker/mowgli.Dockerfile -t multiverse-mowgli .
```

## Running a Container

The model containers are designed to be run by the orchestrator. However, you can also run them manually for testing or debugging.

To run a container, you need to mount an input directory (containing `data.h5ad` or `data.h5mu`) and an output directory.

Example for the PCA model:
```bash
docker run --rm \
    -v /path/to/your/input:/data/input:ro \
    -v /path/to/your/output/pca:/data/output:rw \
    multiverse-pca
```

The container will read the data from `/data/input`, run the model, and write the results (`embeddings.h5ad`, `metrics.json`, `log.txt`) to `/data/output`.
