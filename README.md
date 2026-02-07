# KataGo Feature Extractor

## Overview

This script, `extract_katago_features.py`, demonstrates how to load a pre-trained KataGo neural network model and extract internal features from it for a given Go board position. Specifically, it extracts the output of the 'trunkfinal' layer, which represents the model's processed spatial features of the board state. This is the core functionality of this repository.

## How to Use

1.  **Environment Setup**: Ensure you have a Python environment with `torch` installed. The script is designed to run with the `venv` provided in this repository. Activate it using:
    ```bash
    . venv/bin/activate
    ```

2.  **Run the script**: Execute the script to extract features from an SGF file.

    You can specify one or more nodes from an SGF file using a variation path. The path is a comma-separated list of indices that navigate the game tree. An empty path `""` refers to the root position.

    **Example: Extract features for specific nodes from different SGF files**
    ```bash
    python extract_katago_features.py --sgf-node test.sgf "" --sgf-node test.sgf "0,0" --sgf-node test2.sgf "0,0,1"
    ```
    This command specifies three different positions from two SGF files. The script will process all of them together in a single, efficient batch.

    On the first run, the script will automatically download a pre-trained KataGo model. This is a large file and may take some time. On subsequent runs, it will use the local copy.

## Developer Notes: Obstacles Overcome

This script was developed to work in a constrained, pure-Python environment (like a Docker container) without access to a C++ compiler or system package managers. Several challenges were encountered and overcome:

1.  **Model Format Mismatch**: The primary challenge was that KataGo's officially released models (`.bin.gz`) are in a custom C++ format and are not directly readable by PyTorch's `torch.load()`. Initial attempts to load these files failed with `pickle` errors.

2.  **C++ Dependency for Conversion**: The KataGo repository includes a Python script (`export_model_pytorch.py`) to convert the C++ model to a PyTorch checkpoint. However, this script itself depends on a compiled C++ helper module (`katago.cpp_model`). Attempts to compile this module failed due to a lack of build tools (`cmake`, `g++`) and system permissions (`apt-get`) within the target environment.

3.  **Subprocess Approach Failure**: An alternative strategy was to run a pre-compiled KataGo C++ executable as a subprocess and communicate with it via JSON. This also failed because the environment lacked the necessary tools (`wget`, `unzip`) to download the executable.

4.  **Solution: PyTorch Checkpoints**: The final, successful approach was to bypass the C++ models and executables entirely. By sourcing model files directly from the KataGo training website, we could obtain PyTorch-native training checkpoints (`.ckpt` files). These files are designed to be loaded by `torch.load()`, allowing for a pure-Python solution.

5.  **Download and Extraction Logic**: The script was enhanced to handle downloading the model from a URL, adding a `User-Agent` header to bypass `403 Forbidden` errors, and correctly extracting the model file from its `.zip` archive into a subdirectory.

This iterative process of diagnosing errors and adapting the strategy was key to achieving a working solution within the environment's constraints.

## Testing

This repository includes a test suite to verify the script's command-line interface and core functionality. To run the tests, execute the following command from the root directory:

```bash
python -m unittest test_extract_katago_features.py
```
