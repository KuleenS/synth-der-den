# Use an official Ubuntu as the base image
FROM ubuntu:22.04

# Set environment variable to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl git bzip2 && \
    apt-get clean

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh

# Set path to conda
ENV PATH=/opt/miniconda/bin:$PATH

# Install Python and create a Conda environment
RUN conda init bash && \
    conda create -y --name synth-pipeline python=3.10

# Activate the environment and install dependencies
# This assumes you have a requirements.txt file in your project directory
COPY ./synth-der-den /app/synth-der-den
COPY ./processing_pipeline.sh /app/processing_pipeline.sh

COPY biobert /app/biobert
COPY pipeline /app/pipeline
COPY prototypes /app/prototypes

RUN /bin/bash -c "conda activate synth-pipeline && conda env create -f /app/synth-der-den/environ.yml"

# Set the working directory in the container
WORKDIR /app


# Give execution permission to the script
RUN chmod +x /app/processing_pipeline.sh

# Set the default command to run the application
CMD ["/bin/bash"]
