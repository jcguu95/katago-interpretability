# Changelog

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
