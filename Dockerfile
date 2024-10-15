# Use an official Ubuntu as the base image
FROM ubuntu:22.04
FROM continuumio/miniconda3

# Set environment variable to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget curl git bzip2 build-essential && \
    apt-get clean

# Download and install Miniconda
COPY requirements.txt /app/requirements.txt
COPY environ.yml /app/environ.yml
RUN conda env create --f /app/environ.yml
# Activate the environment and install dependencies
# This assumes you have a requirements.txt file in your project directory
COPY src /app/src/
COPY processing_pipeline.sh /app/processing_pipeline.sh

COPY new_biobert /app/biobert
COPY pipeline /app/pipeline
COPY prototypes /app/prototypes

# Set the working directory in the container
# Give execution permission to the script
RUN chmod +x /app/processing_pipeline.sh

# Ensure this is the root of your project
WORKDIR /app/
# Add src to the Python path
ENV PYTHONPATH=/app/src/  

# Set the default command to run the application
ENTRYPOINT ["conda", "run", "-n", "generation", "/app/processing_pipeline.sh"]
