# KataGo Feature Extractor

## Overview

This script, `extract_katago_features.py`, demonstrates how to load a pre-trained KataGo neural network model and extract internal features from it for a given Go board position. Specifically, it extracts the output of the 'trunkfinal' layer, which represents the model's processed spatial features of the board state.

## How to Use

1.  **Environment Setup**: Ensure you have a Python environment with `torch` installed. The script is designed to run with the `venv` provided in this repository. Activate it using:
    ```bash
    . venv/bin/activate
    ```

2.  **Run the script**: Execute the script from the command line:
    ```bash
    python extract_katago_features.py
    ```
    On the first run, the script will automatically download a pre-trained KataGo model checkpoint from the official KataGo training website. This is a large file and may take some time. The script will then unzip it and load the model. On subsequent runs, it will use the already downloaded file.

    The script will output the shape and contents of the `trunkfinal` tensor for two different board states to demonstrate that the feature extraction is working correctly.

## Developer Notes: Obstacles Overcome

This script was developed to work in a constrained, pure-Python environment (like a Docker container) without access to a C++ compiler or system package managers. Several challenges were encountered and overcome:

1.  **Model Format Mismatch**: The primary challenge was that KataGo's officially released models (`.bin.gz`) are in a custom C++ format and are not directly readable by PyTorch's `torch.load()`. Initial attempts to load these files failed with `pickle` errors.

2.  **C++ Dependency for Conversion**: The KataGo repository includes a Python script (`export_model_pytorch.py`) to convert the C++ model to a PyTorch checkpoint. However, this script itself depends on a compiled C++ helper module (`katago.cpp_model`). Attempts to compile this module failed due to a lack of build tools (`cmake`, `g++`) and system permissions (`apt-get`) within the target environment.

3.  **Subprocess Approach Failure**: An alternative strategy was to run a pre-compiled KataGo C++ executable as a subprocess and communicate with it via JSON. This also failed because the environment lacked the necessary tools (`wget`, `unzip`) to download the executable.

4.  **Solution: PyTorch Checkpoints**: The final, successful approach was to bypass the C++ models and executables entirely. By sourcing model files directly from the KataGo training website, we could obtain PyTorch-native training checkpoints (`.ckpt` files). These files are designed to be loaded by `torch.load()`, allowing for a pure-Python solution.

5.  **Download and Extraction Logic**: The script was enhanced to handle downloading the model from a URL, adding a `User-Agent` header to bypass `403 Forbidden` errors, and correctly extracting the model file from its `.zip` archive into a subdirectory.

This iterative process of diagnosing errors and adapting the strategy was key to achieving a working solution within the environment's constraints.
