# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV KATAGO_MODELS_DIR=/models

# Install git, which is required to check out the submodules
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
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

# Clone the git submodules for KataGo and sgfmill directly
RUN git clone https://github.com/lightvector/KataGo.git katago
RUN git clone https://github.com/mattheww/sgfmill.git sgfmill

# The C++ KataGo engine is not needed for this feature extraction script.
# We return to the app root, which is the default workdir.
WORKDIR /app

# Set the entrypoint to run the feature extraction script
ENTRYPOINT ["python", "extract_katago_features.py"]
