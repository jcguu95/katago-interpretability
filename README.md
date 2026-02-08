# KataGo Feature Extractor

## Overview

This script, `extract_katago_features.py`, demonstrates how to load a pre-trained KataGo neural network model and extract internal features from it for a given Go board position. Specifically, it extracts the output of the 'trunkfinal' layer, which represents the model's processed spatial features of the board state. This is the core functionality of this repository.

## How to Use

This project is containerized using Docker to ensure a completely reproducible environment.

1.  **Prerequisites**: You must have Docker installed on your system.

2.  **Build the Docker Image**: Navigate to the root of the repository and run the following command in your host machine's terminal (not inside another Docker container). This will assemble the environment, compile the KataGo engine, and install all dependencies.
    ```bash
    docker build -t katago-extractor .
    ```

3.  **Run the Extractor**: On your host machine, create local directories to store your SGF files and the downloaded KataGo models. Then, run the extractor inside the container using the following command from your host terminal.

    ```bash
    # Create local directories for data and models
    mkdir -p my_sgfs
    mkdir -p katago_models

    # Place your SGF files (e.g., test.sgf, test2.sgf) inside the 'my_sgfs' directory
    # Now, run the extractor from your host terminal:
    docker run --rm \
      -v "$(pwd)/my_sgfs:/sgfs" \
      -v "$(pwd)/katago_models:/models" \
      katago-extractor \
      --sgf-node /sgfs/test.sgf "" \
      --sgf-node /sgfs/test2.sgf "0,0,1"
    ```

    *The first time you run this command, the script will download a KataGo model into your `katago_models` directory, which may take some time. Subsequent runs will be much faster as they will use the local copy.*

## Testing

The test suite can be run inside the container to verify its functionality against the controlled environment. Run this command from your host machine's terminal using the image you already built.

```bash
docker run --rm katago-extractor python -m unittest test_extract_katago_features.py
```
