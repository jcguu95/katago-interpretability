# Changelog

## 0.9.5 - 2026-02-08

- **fix**: The feature extractor test suite (`test_extract_katago_features.py`) no longer deletes the downloaded model archive on completion. This ensures the model is cached between test runs and fixes a `FileNotFoundError` when `make test-extractor` was run after a full test.
- **fix**: Corrected paths in `.gitignore` for temporary test SGF files and improved the root `clean` target.

## 0.9.4 - 2026-02-08

- **fix**: Fixed a `FileNotFoundError` in the feature extractor test suite by correcting the hardcoded script path in `test_extract_katago_features.py`.

## 0.9.3 - 2026-02-08

- **fix**: Corrected test execution path in `feature_extraction/Makefile`. The tests are now run from the project root to ensure they can find the downloaded model file, resolving a `FileNotFoundError`.

## 0.9.2 - 2026-02-08

- **fix**: Corrected a recurring `SyntaxError` in the `Makefile`'s model extraction command by simplifying the Python one-liner and removing a redundant conditional.

## 0.9.1 - 2026-02-08

- **fix**: Resolved a `SyntaxError` in the `Makefile`'s model extraction command. The Python one-liner was using a `with` statement, which is not valid in that context. It has been replaced with an equivalent `open`/`close` sequence.

## 0.9.0 - 2026-02-08

- **refactor**: Reorganized the project to separate feature extraction from the SAE pipeline.
  - Moved `extract_katago_features.py` and its test into a new `feature_extraction/` directory.
  - Created a separate `feature_extraction/Makefile` for managing the standalone extractor component.
- **chore**: Overhauled the root `Makefile` for clarity.
  - Renamed `test-all` to `test-pipeline-full` to better describe the end-to-end SAE pipeline test.
  - Added `test-extractor` and `test-extractor-full` to delegate testing to the new `feature_extraction/` component.
- **docs**: Updated `README.md` to reflect the new project structure and testing commands.

## 0.8.4 - 2026-02-08

- **chore**: Added a `make test-all` command to run a full clean test of all project functionality, including the data pipeline, visualization, and original test suite.

## 0.8.3 - 2026-02-08

- **chore**: Added a `make retrain-sae` command to the `Makefile` to provide a convenient way to delete the existing model and re-run the training process.

## 0.8.2 - 2026-02-08

- **fix**: Fixed a crash in `visualize_sae.py` where the script incorrectly inferred the model's input feature size, causing a dimension mismatch error when loading activations.

## 0.8.1 - 2026-02-08

- **fix**: Resolved linting errors by removing unused imports from several scripts and adding a missing `sys` import in `visualize_sae.py`.

## 0.8.0 - 2026-02-08

- **feat**: Added `visualize_sae.py`, a new script to analyze a trained SAE model. It loads the model and activations, and reports reconstruction error and feature sparsity for a random sample.
- **feat**: The SAE training script (`train_sae.py`) now displays a `tqdm` progress bar during training, showing an ETA and the current loss per batch.
- **chore**: Added `make visualize-sae` target to the `Makefile` and updated the `README.md` with instructions.
- **deps**: Added `tqdm` to `requirements.txt`.

## 0.7.0 - 2026-02-08

- **perf**: The data collection and SAE training pipelines now automatically use a CUDA-enabled GPU if one is available, falling back to CPU otherwise. This addresses a major performance bottleneck for users with capable hardware and prepares the project for scaling up to larger datasets.
- **docs**: Updated `README.md` to reflect the current state of the project, including the full data pipeline and its use for training an SAE.

## 0.6.7 - 2026-02-08

- **fix**: The SAE training script (`train_sae.py`) no longer flattens the entire activation map into the feature dimension. It now correctly treats each spatial location as a sample and the channels as the feature vector, which resolves the out-of-memory `Killed` error during model initialization.
- **fix**: Refactored the `Makefile` to prevent re-downloading the model if the extracted model directory exists but the `.zip` archive has been deleted.

## 0.6.6 - 2026-02-08

- **fix**: Resolved a `SyntaxError` in the `Makefile`'s model download command. The Python one-liner was using a `with` statement, which is not valid in that context. It has been replaced with an equivalent `open`/`close` sequence.

## 0.6.5 - 2026-02-08

- **fix**: The `Makefile`'s data pipeline now checks for the extracted model before attempting to download the model archive. This prevents re-downloading the model if it has been unzipped but the archive is missing.

## 0.6.4 - 2026-02-08

- **fix**: Made the model download process more robust by using a temporary file. The final model archive is only moved into place after the download is complete, preventing corruption from interrupted downloads.

## 0.6.3 - 2026-02-08

- **fix**: Corrected a dependency in the `Makefile` that caused the KataGo model to be re-downloaded on every run of `make data-pipeline` instead of only when missing.

## 0.6.2 - 2026-02-08

- **fix**: Fixed a crash in `collect_activations.py` caused by an incorrect API call to the `sgfmill` library. The script now correctly checks for child nodes when traversing SGF game trees, resolving an `AttributeError`.

## 0.6.1 - 2026-02-08

- **fix**: The `data-pipeline` now automatically downloads the required KataGo model if it is not found locally, preventing errors when run in a clean environment. The `clean` target no longer removes the downloaded model zip, only the extracted files.
- **feat**: SGF generation (`generate_sgfs.py`) now creates more plausible game openings by playing the first few moves on common hoshi and komoku points instead of being purely random.

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
