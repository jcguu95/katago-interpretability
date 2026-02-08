# Changelog

## 0.6.0 - 2026-02-08

- **feat**: Implemented a full training pipeline for the Sparse Autoencoder (SAE).
  - `train_sae.py` now includes a training loop with MSE reconstruction loss and an L1 sparsity penalty.
  - The script is configurable via command-line arguments for hyperparameters like epochs, learning rate, and batch size.
  - The trained model is saved to a file (`activations/sae.pt` by default).
- **chore**: Updated the `Makefile` to run the full data pipeline, including SAE training, with the `make data-pipeline` command. This provides a complete end-to-end test from data generation to a trained model.

## 0.5.5 - 2026-02-08

- **chore**: Confirmed that all tests for the original feature extraction script (`make test-full`) are passing. The core functionality is stable before proceeding with SAE implementation.

## 0.5.4 - 2026-02-08

- **fix**: Updated test model URL again to resolve a persistent 403 Forbidden error. The project now uses a `b28c512nbt` model and the tests have been updated to reflect its different output shape.

## 0.5.3 - 2026-02-08

- **fix**: Improved the model zip extraction logic in `extract_katago_features.py` to be more robust. It now avoids creating nested directories and prevents re-extracting if the model file is already present.

## 0.5.2 - 2026-02-08

- **fix**: Updated test model URL to resolve a 403 Forbidden error during download. The project now uses a `b10c128` model for tests and as the default.
- **docs**: Added instructions for using a Docker-based development environment to `README.md`.

## 0.5.1 - 2026-02-08

- **fix**: Updated the test suite and data pipeline to use a smaller, compatible KataGo test model in the correct `.ckpt` format. The previous model was in an unsupported `.bin.gz` format that would cause tests to fail.
- **docs**: Clarified in `README.md` that the feature extractor only supports `.ckpt` model files, not `.bin.gz` files.

## 0.5.0 - 2026-02-08

- **feat**: Added a reproducible data generation and feature extraction pipeline for SAE training.
  - `generate_sgfs.py`: New script to create synthetic SGF files with random moves for testing the pipeline.
  - `collect_activations.py`: New script to process a directory of SGF files, extract `trunkfinal` activations for every board state, and save them to a `.pt` file with reproducibility metadata.
  - `train_sae.py`: New stub script that demonstrates loading the activations file, preparing for future SAE model training.
- **chore**: Integrated the new data pipeline scripts into the `Makefile` under the `make data-pipeline` target.

## 0.4.0 - 2026-02-08

- **feat**: Began pivot towards training a Sparse Autoencoder (SAE) on KataGo's internal features. This is the first step in a larger effort to build interpretability tools for the model.
- **chore**: Made the core feature extraction script (`extract_katago_features.py`) and its test (`test_extract_katago_features.py`) read-only to ensure stability during the addition of new functionality.

## 0.3.0 - 2026-02-08

- **refactor**: Replaced the Docker-based setup with a more secure and lightweight `venv` and `git submodule` approach.
- **feat**: Added a `Makefile` to automate installation (`make install`), quick testing (`make test`), and clean testing with model download (`make test-full`).
- **fix**: Resolved all model compatibility issues by removing support for legacy formats (`.txt.gz`) and updating the test suite to use a modern, compatible KataGo model. The test suite now passes reliably.

## 0.2.1 - 2026-02-07

- **fix**: The script now exits with a non-zero status code if all specified SGF nodes fail to process, making error handling more robust for scripting.

## 0.2.0 - 2026-02-07

- Added a test suite (`test_extract_katago_features.py`) to verify command-line functionality and prevent regressions.

## 0.1.0 - 2026-02-07

- Implemented SGF parsing to extract features from any node in a game tree, including variations.
- Added `--sgf-file` and `--variation-path` arguments.
- Implemented batch processing to extract features for multiple game states in a single pass, improving performance.
- Replaced `--sgf-file` and `--variation-path` with `--sgf-node` to allow specifying exact file-path pairs, improving flexibility.
- Batch processing now combines all specified nodes into a single global batch for improved performance.

## 0.0.0 - 2026-02-07

- Initial release.
- The script can load a KataGo model and extract the 'trunkfinal' feature tensor for a given board state.
