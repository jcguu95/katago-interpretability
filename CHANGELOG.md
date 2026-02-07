# Changelog

## 0.1.0 - 2026-02-07

- Implemented SGF parsing to extract features from any node in a game tree, including variations.
- Added `--sgf-file` and `--variation-path` arguments.
- Implemented batch processing to extract features for multiple game states in a single pass, improving performance.
- Replaced `--sgf-file` and `--variation-path` with `--sgf-node` to allow specifying exact file-path pairs, improving flexibility.
- Batch processing now combines all specified nodes into a single global batch for improved performance.

## 0.0.0 - 2026-02-07

- Initial release.
- The script can load a KataGo model and extract the 'trunkfinal' feature tensor for a given board state.
