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
    At the end of the process, you will have a trained model file at `activations/sae.pt`. This file contains both the model weights and the hyperparameters used for training.

## Visualizing SAE Features

After training an SAE model with `make data-pipeline`, you can run a basic analysis script to inspect its behavior on a single, random activation vector from the dataset.

```bash
make visualize-sae
```

This will print:
- The hyperparameters used to train the model.
- The reconstruction error (MSE) for the sample.
- The number of "active" features in the SAE's hidden layer.
- The index and activation value of the most active features.

This serves as a starting point for deeper analysis of what the SAE has learned.

## SAE Architecture and Evaluation

This section details the current implementation of the Sparse Autoencoder (SAE) and provides guidance on its evaluation.

### Current Architecture

The current SAE is a simple, single-hidden-layer neural network with the following components:

-   **Input**: The network takes feature vectors from KataGo's `trunkfinal` layer. The activations tensor, with an original shape of `(N, C, H, W)`, is reshaped so that each spatial location `(H, W)` for each sample `N` is treated as an independent data point. This results in an input shape of `(N*H*W, C)`, where `C` is the number of input features.
-   **Encoder**: A single linear layer that maps the `C` input features to a larger number of "dictionary" features, `D`. The size of `D` is controlled by the `--dict-size-factor` (default is `4 * C`). This is followed by a Rectified Linear Unit (ReLU) activation function, which introduces non-linearity and ensures the learned features are non-negative.
-   **Decoder**: A single linear layer that maps the `D` dictionary features back to the original `C`-dimensional space, attempting to reconstruct the original input vector.
-   **Loss Function**: The model is trained to minimize a composite loss function:
    `Loss = Reconstruction_Loss + λ * Sparsity_Loss`
    -   **Reconstruction Loss**: This is the Mean Squared Error (MSE) between the decoder's output and the original input vector. It pushes the model to learn a faithful representation of the data.
    -   **Sparsity Loss**: A penalty on the hidden layer activations to encourage sparsity. The type of penalty can be configured (e.g., L1 or Lp norm) using the `--sparsity-type` argument. The coefficient `λ` (`--sparsity-coeff`) controls the strength of this pressure.

### Alternative Training Approaches

While the current setup is a standard starting point, several other techniques could be explored:

-   **Different Sparsity Penalties**: Instead of L1, one could use a KL-divergence penalty to encourage the average activation of each feature over the whole dataset to be close to a small target value.
-   **More Complex Architectures**: Deeper autoencoders with multiple hidden layers could potentially learn more hierarchical features.
-   **Activation Preprocessing**: Techniques like whitening (decorrelating the input features and scaling them to have unit variance) can sometimes help training.

### How to Evaluate the SAE

The goal of this SAE is not just to reconstruct its input, but to provide *interpretable* features. The script `visualize_sae.py` provides a basic check of reconstruction error and sparsity on a single input. A more thorough evaluation involves:

1.  **Measuring the Sparsity/Reconstruction Trade-off**: A good SAE should achieve low reconstruction error while maintaining high sparsity (few active features per input). You can evaluate this by training models with different `l1-lambda` values and plotting the resulting MSE vs. the average number of active features (L0 norm).

2.  **Qualitative Feature Interpretation (The "Human-in-the-Loop" part)**: This is the most crucial part of evaluation. To understand what a feature has learned, you can:
    -   Find the data points (i.e., board states) from your dataset that cause the highest activation for a specific feature in the hidden layer.
    -   Visualize these board states.
    -   As a human Go player, look for common patterns. Does the feature activate for "a white group in atari"? Or "a black moyo forming on the right side"? Or "a key cutting point"?

This is where your domain expertise as a Go player is essential. You don't need to "match" the decoded information to human data in a formal, supervised sense. Instead, you use human-understandable concepts from Go to label and understand the learned features. Using SGFs of known situations (joseki, puzzles, pro games) is an excellent way to probe what the features represent.

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
