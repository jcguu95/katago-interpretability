# KataGo Feature Extractor

## Overview

This repository provides tools to work with the internal representations of the KataGo Go engine. The primary functionality is to extract feature tensors (activations) from a trained KataGo model for given board states.

### Project Structure

The project is organized into two main components:
- **`feature_extraction/`**: A self-contained module for extracting feature tensors from KataGo. It has its own tests and can be used independently of the SAE pipeline.
- **SAE Training Pipeline (root)**: A set of scripts and a `Makefile` for generating data, collecting activations, and training a Sparse Autoencoder.

## Quick Start

This project uses `Makefile`s to automate setup and testing.

1.  **Clone the Repository**:
    ```bash
    git clone --recurse-submodules https://github.com/your-username/katago-feature-extractor.git
    cd katago-feature-extractor
    ```
    *If you have already cloned the repository without the submodules, you can initialize them by running `git submodule update --init --recursive`.*

2.  **Install Dependencies**: This command creates a Python virtual environment in `venv/` and installs all dependencies.
    ```bash
    make install
    ```

3.  **Run Tests**: There are two sets of tests:
    - **Feature Extractor Tests**: To test only the standalone feature extraction component. The `-full` version cleans everything and downloads the model.
      ```bash
      make test-extractor-full
      ```
    - **Full Pipeline Test**: To run a clean, end-to-end test of the entire SAE training pipeline.
      ```bash
      make test-pipeline-full
      ```

## Data Pipeline for SAE Training

The primary workflow for this project is the data pipeline, which automates the process from generating sample data to training a preliminary SAE model.

1.  **Run the Full Pipeline**: This single command will:
    - Generate a small set of synthetic SGF files (if they don't exist).
    - Download the necessary KataGo model (if it doesn't exist).
    - Collect feature activations from the SGFs.
    - Train a small sparse autoencoder on the activations.

    ```bash
    make data-pipeline
    ```
    At the end of the process, you will have a trained model file at `activations/sae.pt`.

## Visualizing SAE Features

After training an SAE model with `make data-pipeline`, you can run a basic analysis script to inspect its behavior on a single, random activation vector from the dataset.

```bash
make visualize-sae
```

This will print:
- The reconstruction error (MSE) for the sample.
- The number of "active" features in the SAE's hidden layer.
- The index and activation value of the most active features.

This serves as a starting point for deeper analysis of what the SAE has learned.

## Manual Usage

If you need to extract features for specific SGF files manually, you can run the main extraction script. You must first activate the virtual environment.

1.  **Activate the Environment**:
    ```bash
    source venv/bin/activate
    ```

2.  **Set `PYTHONPATH`**:
    ```bash
    export PYTHONPATH=$(pwd)/katago/python
    ```

3.  **Run the Script**:
    ```bash
    # Create a directory for your SGF files if you don't have one
    mkdir -p my_sgfs
    # (Place your SGF files in 'my_sgfs')

    # Example: Extract features from two different SGF files
    python feature_extraction/extract_katago_features.py \
      --sgf-node my_sgfs/test.sgf "" \
      --sgf-node my_sgfs/test2.sgf "0,0,1"
    ```
    *Note: The first time you run this script, it will download a default KataGo model, which may take some time.*

## Reproducibility and Compatibility

The project is designed to be highly reproducible. Dependencies are managed as follows:

-   **Python Packages**: All Python dependencies (e.g., `torch`, `numpy`) are pinned to specific versions in `requirements.txt`.
-   **KataGo and sgfmill**: The exact versions of the KataGo and `sgfmill` source code are pinned using `git submodules`, which lock them to a specific commit hash.
-   **Test Model**: The small KataGo model used for testing is downloaded automatically by running `make test-full`.

### Model Compatibility

**Warning**: This script is designed to work with PyTorch-native KataGo models (`.ckpt` format). The default model and the test model are known to be compatible. Using KataGo's native C++ engine format (e.g., `.bin.gz`) or older legacy formats (such as `.txt.gz`) is **not supported** and will result in errors.

### Development Environment

This project is developed and tested on a Linux environment. The Python package versions are strictly pinned in `requirements.txt` and are known to work on this platform. Running on other operating systems (like macOS) may result in dependency installation issues (e.g., for `torch`), as specific package versions may not be available for all platforms.

-   **Tested Platform**: `Linux-6.12.54-linuxkit-aarch64-with-glibc2.36`
-   **Python Version**: 3.10 (as defined by the virtual environment setup)

There are no external system dependencies required to run this project on the tested Linux platform, as it only uses the Python components of its submodules. If you are attempting to run on an unsupported platform (such as macOS) and `pip` needs to build packages like `numpy` or `torch` from source, you may need to install additional build tools (e.g., `ninja`).

#### Development with Docker and Aider

For a reproducible development environment, you can use a Docker container. This is the environment used for recent development on this project.

1.  **Start the container**: The following command starts an `aider` session within a container, mounting the current directory.
    ```bash
    docker run -it --rm \
        --user $(id -u):$(id -g) \
        --volume $(pwd):/app \
        --env GEMINI_API_KEY="YOUR_API_KEY" \
        paulgauthier/aider-full \
        --model gemini \
        --no-stream \
        --weak-model gemini/gemini-2.0-flash-lite
    ```
    *Note: Replace `"YOUR_API_KEY"` with your actual API key. You can exit the `aider` session with `/exit` to get a regular shell prompt inside the container.*

2.  **Run tests inside the container**: Once you have a shell inside the container, you can run the full pipeline test:
    ```bash
    make test-pipeline-full
    ```
