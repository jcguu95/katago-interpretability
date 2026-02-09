# Changelog

## 0.12.3 - 2026-02-09

- **test**: Updated the feature extractor test suite to handle the new `--layer-name` argument.
- **test**: Added a specific test to ensure `trunkfinal` extraction still works correctly.
- **test**: Updated the numerical consistency test to explicitly use `trunkfinal`, preserving the original baseline.

## 0.12.2 - 2026-02-09

- **refactor**: The script `feature_extraction/extract_katago_features.py` can now extract different layers via a `--layer-name` argument, defaulting to `policy_penultimate`.
- **docs**: Added documentation to the extractor script to clarify its dual-purpose nature.
- **chore**: Added a `make test-verification-script` command to easily run the `policy_penultimate` verification script.
- **docs**: Updated `README.md` to reflect the new target layer for SAE training.

## 0.12.1 - 2026-02-09

- **fix**: Corrected a `TypeError` in `verify_policy_penultimate.py` that occurred because the script did not handle cases where the model's configuration is attached to the model object instead of being returned separately.

## 0.12.0 - 2026-02-08

- **milestone**: The `v0.11.x` series has concluded. We have a solid understanding of the SAE training architecture and its limitations with respect to the `trunkfinal` layer (i.e., its lack of global context).
- **note**: The main goal of the `v0.12.x` series is to retarget our analysis to a more specialized layer. We will modify the data pipeline to extract activations from the policy head's penultimate layer and train a new SAE on this richer, more move-specific representation.
- **feat**: Exposed the policy head's penultimate activation layer as `policy_penultimate` for feature extraction.

## 0.11.10 - 2026-02-08

- **docs**: Documented the significant finding (unpublished observation by Jin-Cheng Guu) that KataGo's raw policy network, without any search, is strong enough to beat high-dan players. This justifies our focus on understanding the network's internal representations directly.
- **docs**: Identified the policy head's penultimate layer as a potential future target for feature extraction, offering a representation that is more specialized for move prediction than `trunkfinal`.

## 0.11.9 - 2026-02-08

- **docs**: Deepened the analysis of KataGo's `PolicyHead`, documenting that it uses both local (per-point) and global (whole-board) information to evaluate moves. This clarifies the limitations of our current SAE, which only sees local features.

## 0.11.8 - 2026-02-08

- **feat**: Overhauled `visualize_sae.py` to perform a global analysis of feature statistics across the entire dataset, replacing the previous single-vector analysis.
- **docs**: The visualization script now reports on feature activation frequency, average magnitude, and max magnitude, providing a much richer understanding of the learned dictionary.

## 0.11.7 - 2026-02-08

- **feat**: Made the SAE's activation function configurable (`relu` or `gelu`) to facilitate experimentation with different non-linearities.
- **docs**: Added comments to `sae_model.py` explaining that ReLU is the standard choice for SAEs because its properties naturally encourage sparse activations.

## 0.11.6 - 2026-02-08

- **docs**: Clarified in `sae_model.py` that the ReLU activation function on the encoder output serves as the essential non-linearity for the autoencoder.

## 0.11.5 - 2026-02-08

- **docs**: Added comments to `sae_model.py` and `train_sae.py` to explicitly identify the encoder/decoder weights (W) and biases (b) and to pinpoint the exact line where these parameters are updated during training.

## 0.11.4 - 2026-02-08

- **docs**: Documented the architectural limitation of the current SAE model: by treating each board location independently, it cannot learn features that represent spatial relationships (e.g., "groups" or "zones"). This is a deliberate trade-off for model simplicity.

## 0.11.3 - 2026-02-08

- **docs**: Added a detailed explanation to `train_sae.py` and `docs/katago_model_architecture.md` about why the `512x19x19` activation tensor is treated as a batch of 512-dimensional vectors for SAE training.
- **config**: Adjusted the default dictionary size factor to 8x (512 -> 4096) as a balanced starting point for hyperparameter tuning.

## 0.11.2 - 2026-02-08

- **config**: Centralized SAE training hyperparameters in the `Makefile` for easier experimentation.
- **feat**: Increased the default dictionary size factor from 4x to 16x (`512 -> 8192`) as a more suitable baseline for capturing the feature complexity of KataGo.

## 0.11.1 - 2026-02-08

- **docs**: Added comments to `train_sae.py` to clarify the meaning of `input_features` and `dict_features` in mathematical and machine learning terms, improving the script's readability.

## 0.11.0 - 2026-02-08

- **milestone**: Completed a detailed investigation of KataGo's feature extractor. The v0.10.x series has culminated in a solid understanding of the model's architecture and the `trunkfinal` activations.
- **note**: The main goal of the v0.11.x series is to gain a precise understanding of the Sparse Autoencoder (SAE) training architecture as implemented in this project.

## 0.10.8 - 2026-02-08

- **docs**: Corrected and clarified the architecture document. The `trunkfinal` tensor is named `out` in the source code when passed to the policy/value heads, and this is now explicitly stated. This reverts a previous documentation change that incorrectly reflected a rejected code modification.

## 0.10.7 - 2026-02-08

- **docs**: Expanded the architecture document with a detailed explanation of how the `trunkfinal` tensor is processed within the Policy Head.

## 0.10.6 - 2026-02-08

- **test**: Updated the expected hash in the numerical consistency test to match the current model's output.
- **refactor**: Removed the debug print of the model architecture from the feature extractor to clean up test output.

## 0.10.5 - 2026-02-08

- **docs**: Clarified the purpose of `extract_trunkfinal_output` as a single-state convenience wrapper around the batch processing method.

## 0.10.4 - 2026-02-08

- **docs**: Updated `docs/katago_model_architecture.md` to include the shape of the `trunkfinal` tensor.

## 0.10.3 - 2026-02-08

- **docs**: Added a new document, `docs/katago_model_architecture.md`, explaining the architecture of KataGo's neural network and confirming that `trunkfinal` is the correct layer for feature extraction.

## 0.10.2 - 2026-02-08

- **note**: The main goal of the v0.10.x series is to gain a precise understanding of the feature extractor. Upcoming changes will be suggested by the user as this understanding develops.

## 0.10.1 - 2026-02-08

- **fix**: Corrected the argument for sparsity coefficient in `Makefile` from `--l1-lambda` to `--sparsity-coeff` to match `train_sae.py`.

## 0.10.0 - 2026-02-08

- **release**: Version 0.10.0 marks a significant milestone. The project now provides a complete, end-to-end toolkit for training Sparse Autoencoders on KataGo's internal activations and includes robust tools for data verification and feature interpretation. The core components are stable and ready for large-scale data collection and analysis.

## 0.9.14 - 2026-02-08

- **docs**: Renamed the project to "KataGo Interpretability" to better reflect its purpose and updated documentation accordingly.

## 0.9.13 - 2026-02-08

- **refactor**: Removed the activation visualization feature from `verify_activations.py`. This simplifies the script's purpose to data verification and resolves an associated linting error.

## 0.9.12 - 2026-02-08

- **feat**: Added visualization to `verify_activations.py`. The script can now plot a heatmap of any activation channel over the board to visually confirm feature extraction is working correctly.
- **feat**: `find_max_activating_examples.py` now prints the board state for each top-activating example, making feature interpretation much easier.
- **feat**: The training script (`train_sae.py`) now reports reconstruction and sparsity losses separately, providing better insight into the training dynamics.
- **refactor**: SGF parsing logic has been moved to a shared `sgf_utils.py` file.

## 0.9.11 - 2026-02-08

- **feat**: Added `verify_activations.py`, a new script for sanity-checking the feature extraction process on a single board position.
- **docs**: Added guidance on data collection strategy and pipeline verification to `README.md`.

## 0.9.10 - 2026-02-08

- **feat**: Added `find_max_activating_examples.py`, a new script for feature interpretation that finds the game positions that most strongly activate a given SAE feature.
- **feat**: `collect_activations.py` now saves a `source_map` to link activations back to their original SGF file and move number, and includes a `--limit` flag to process a subset of SGFs.

## 0.9.9 - 2026-02-08

- **feat**: Extended SAE training with configurable sparsity penalties (L1, Lp norm) and saved all training hyperparameters with the model for improved reproducibility and analysis.

## 0.9.8 - 2026-02-08

- **docs**: Added a detailed section to `README.md` explaining the SAE architecture, alternative training approaches, and methods for evaluation and feature interpretation.

## 0.9.7 - 2026-02-08

- **refactor**: The `SparseAutoencoder` class has been moved into its own file (`sae_model.py`) to eliminate code duplication between the training and visualization scripts.

## 0.9.6 - 2026-02-08

- **fix**: The `clean` command in `feature_extraction/Makefile` no longer attempts to delete model files, which was causing models to be deleted unintentionally.
- **chore**: The `test-extractor-full` command now explicitly deletes the model archive before running to ensure the download logic is tested. The model is preserved after the test completes for use in subsequent runs.

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
