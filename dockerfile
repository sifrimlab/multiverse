# Start from the continuumio/anaconda3 base image
FROM continuumio/anaconda3

# Set metadata
LABEL maintainer="Saptarshi Chakrabarti <saptarshi.chakrabarti@kuleuven.be>"
LABEL description="Docker image based on continuumio/anaconda3 with additional tools and a predefined conda environment."

# Update and install necessary packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        cmake \
        build-essential \
        gfortran \
        pkg-config \
        git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the repository into the container
WORKDIR /home
RUN git clone https://github.com/sifrimlab/multi-verse.git

# Install dependencies using pip from the cloned repository's requirements.txt
RUN pip install -r /home/multi-verse/requirements.txt

# Create the conda environment from the repository's environment.yml file (if needed for additional dependencies)
RUN conda env create -f /home/multi-verse/environment.yml && \
    conda clean -afy

# Activate the conda environment by default
ENV PATH=/opt/conda/envs/multiverse/bin:$PATH

# Expose a port (e.g., Jupyter default port 8888)
EXPOSE 8888

# Set the default command to start a bash shell
CMD ["bash"]
