# Makefile for KataGo Feature Extractor

.PHONY: all test test-full install reinstall clean help

# Use the system's python3.
PYTHON := python3

# Define the virtual environment directory.
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
INSTALL_STAMP := $(VENV_DIR)/.installed

# Default target when 'make' is run without arguments.
all: help

# Help target to display available commands.
help:
	@echo "Available commands:"
	@echo "  make install    - Set up the virtual environment and install dependencies if needed."
	@echo "  make reinstall  - Force reinstallation of dependencies."
	@echo "  make test       - Run the test suite."
	@echo "  make test-full  - Run a clean test, removing venv and models first."
	@echo "  make clean      - Remove virtual environment and other generated files."
	@echo ""
	@echo "Data Pipeline commands:"
	@echo "  make data-pipeline       - Run the full SGF generation and feature extraction pipeline."
	@echo "  make generate-sgfs       - Generate synthetic SGF files."
	@echo "  make collect-activations - Extract features from SGFs into an activations file."
	@echo "  make train-sae           - Run the SAE trainer stub on the activations file."

# The main install target. It depends on a stamp file.
install: $(INSTALL_STAMP)

# Rule to create the virtual environment if the python executable doesn't exist.
$(VENV_PYTHON):
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

# The stamp file is created after installation. It depends on the venv python and requirements.txt.
# If requirements.txt changes, this rule will re-run.
$(INSTALL_STAMP): $(VENV_PYTHON) requirements.txt
	@echo "Initializing Git submodules..."
	@git submodule update --init --recursive
	@echo "Installing dependencies..."
	"$(VENV_PYTHON)" -m pip install -r requirements.txt
	"$(VENV_PYTHON)" -m pip install ./sgfmill
	@touch $(INSTALL_STAMP)

# A target to force reinstallation.
reinstall:
	@rm -f $(INSTALL_STAMP)
	@$(MAKE) install

# Target to run tests. Assumes model is already downloaded.
test: install
	@echo "Running tests..."
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) -m unittest test_extract_katago_features.py

# Target for a full, clean test run that downloads the model.
test-full: clean install
	@echo "Running full tests (including model download)..."
	@ALLOW_MODEL_DOWNLOAD=1 PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) -m unittest test_extract_katago_features.py

# --- Data Pipeline ---

.PHONY: data-pipeline generate-sgfs collect-activations train-sae

# Variables for the data pipeline
SGF_DIR := generated_sgfs
ACTIVATIONS_DIR := activations
# This is the default model downloaded by the scripts if not present.
MODEL_ZIP := kata1-b6c96-s175439552-d46399393-checkpoint.zip
ACTIVATIONS_FILE := $(ACTIVATIONS_DIR)/activations.pt

# Target to run the full data generation and processing pipeline.
data-pipeline: $(ACTIVATIONS_FILE)
	@$(MAKE) train-sae

# Target to generate synthetic SGF data.
generate-sgfs: $(SGF_DIR)

$(SGF_DIR): install
	@echo "--- Generating synthetic SGF data ---"
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) generate_sgfs.py --output-dir $(SGF_DIR)

# Target to collect activations from the generated SGFs.
collect-activations: $(ACTIVATIONS_FILE)

$(ACTIVATIONS_FILE): $(SGF_DIR)
	@echo "--- Collecting activations ---"
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) collect_activations.py --sgf-dir $(SGF_DIR) --model-path $(MODEL_ZIP) --output-file $(ACTIVATIONS_FILE)

# Target to run the SAE training script (currently a stub).
train-sae: $(ACTIVATIONS_FILE)
	@echo "--- Running SAE training stub ---"
	@$(VENV_PYTHON) train_sae.py --activations-file $(ACTIVATIONS_FILE)


# Target to clean up the project directory.
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -f test.sgf test2.sgf
	rm -rf kata1-*
	rm -rf $(SGF_DIR) $(ACTIVATIONS_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Cleaning submodules..."
	@git submodule foreach --recursive git clean -fdx
	@git submodule foreach --recursive git reset --hard
