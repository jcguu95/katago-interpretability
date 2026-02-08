# KataGo Feature Extractor

## Overview

This script, `extract_katago_features.py`, demonstrates how to load a pre-trained KataGo neural network model and extract internal features from it for a given Go board position. Specifically, it extracts the output of the 'trunkfinal' layer, which represents the model's processed spatial features of the board state. This is the core functionality of this repository.

## Quick Start

This project uses a `Makefile` to automate setup and testing. For details on the commands being run, you can inspect the `Makefile`.

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

3.  **Run Tests**:
    - **Full Test (run this first)**: `make test-full`. This cleans the environment, installs dependencies, downloads the test model, and runs all tests.
      ```bash
      make test-full
      ```
    - **Quick Test**: `make test`. This runs the tests without re-downloading the model, assuming dependencies are already installed. It's much faster for repeated testing.
      ```bash
      make test
      ```

## Usage

After setting up the environment with `make install`, you can run the main extraction script. You must first activate the virtual environment that was created.

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
    python extract_katago_features.py \
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

**Warning**: This script is designed to work with modern KataGo models (e.g., `.bin.gz` or `.ckpt` format). The default model and the test model are known to be compatible. Using older, legacy model formats (such as `.txt.gz`) is **not supported** and will likely result in errors.

### Development Environment

This project is developed and tested on a Linux environment. The Python package versions are strictly pinned in `requirements.txt` and are known to work on this platform. Running on other operating systems (like macOS) may result in dependency installation issues (e.g., for `torch`), as specific package versions may not be available for all platforms.

-   **Tested Platform**: `Linux-6.12.54-linuxkit-aarch64-with-glibc2.36`
-   **Python Version**: 3.10 (as defined by the virtual environment setup)

There are no external system dependencies (like `ninja`) required to run this project, as it only uses the Python components of its submodules.
