# KataGo Feature Extractor

## Overview

This script, `extract_katago_features.py`, demonstrates how to load a pre-trained KataGo neural network model and extract internal features from it for a given Go board position. Specifically, it extracts the output of the 'trunkfinal' layer, which represents the model's processed spatial features of the board state. This is the core functionality of this repository.

## How to Use

1.  **Setup**: Ensure you have a Python environment with `torch` installed.

2.  **Run**: Execute the script to extract features from SGF files. You can specify multiple nodes from different SGF files, and they will be processed together in an efficient batch.

    The `--sgf-node` argument takes two values: the SGF file path and a variation path. The variation path is a comma-separated list of indices that navigate the game tree (e.g., `"0,0,1"`). An empty path `""` refers to the root position.

    **Example**
    ```bash
    python extract_katago_features.py --sgf-node test.sgf "" --sgf-node test2.sgf "0,0,1"
    ```

    *On the first run, the script will automatically download and unzip a pre-trained KataGo model, which may take some time. Subsequent runs will use the local copy.*

## Testing

This repository includes a test suite to verify the script's functionality. To run the tests:

```bash
python -m unittest test_extract_katago_features.py
```
