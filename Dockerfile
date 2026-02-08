# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV KATAGO_MODELS_DIR=/models

# Install system dependencies required for building KataGo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    zlib1g-dev \
    libzip-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create directories for persistent models and SGF files
RUN mkdir /models && mkdir /sgfs

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Initialize git submodules for KataGo and sgfmill
# We need to configure git to allow submodule init from a detached HEAD in CI/Docker environments
RUN git config --global --add safe.directory /app
RUN git submodule update --init --recursive

# Build the KataGo engine
WORKDIR /app/katago/cpp
RUN cmake . -DUSE_BACKEND=OPENCL && make

# Go back to the app root and set it as the default workdir
WORKDIR /app

# Set the entrypoint to run the feature extraction script
ENTRYPOINT ["python", "extract_katago_features.py"]
