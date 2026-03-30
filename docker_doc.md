# Multi-Verse Docker Container

This guide provides instructions for building and using a Docker container that runs the Multi-Verse repository with all dependencies pre-installed in a conda environment.

## Overview
The container is based on the `continuumio/anaconda3` image and includes:
- Required system packages (`cmake`, `build-essential`, `gfortran`, `pkg-config`, `git`).
- The `multi-verse` repository, cloned from GitHub.
- A conda environment (`multiverse`) with dependencies defined in the `environment.yml` file from the repository.

## Prerequisites
- Docker installed on your system.
- Internet connection to clone the repository and fetch dependencies.

## Build the Docker Image
1. Clone this repository or save the provided `Dockerfile` locally.
2. Build the image using the following command:
   ```bash
   docker build -t multiverse .
   ```

## Run the Container
1. Start a container using the built image:
   ```bash
   docker run -it --name multiverse -p 8888:8888 multiverse
   ```
2. The container will start and provide an interactive shell.

## Repository and Environment
- The `multi-verse` repository is cloned into the `/home/multi-verse` directory inside the container.
- The `multiverse` conda environment is pre-installed.

### Activating the Conda Environment
Inside the container, activate the `multiverse` environment:
```bash
conda activate multiverse
```

## Using Jupyter Notebook
1. Start a Jupyter Notebook server inside the container:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
   ```
2. Copy the provided URL with the token (e.g., `http://127.0.0.1:8888/?token=...`) and open it in your browser.
3. Access the notebooks in the `/home/multi-verse` directory.

## Stopping the Container
To stop the running container:
```bash
docker stop multiverse
```

## Restarting the Container
To restart a stopped container:
```bash
docker start -ai multiverse
```

## Cleaning Up
To remove the container and free up resources:
```bash
docker rm multiverse
```
To delete the image:
```bash
docker rmi multiverse
```
