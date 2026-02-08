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
	@rm -f $(MODEL_ZIP)
	@ALLOW_MODEL_DOWNLOAD=1 PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) -m unittest test_extract_katago_features.py

# --- Data Pipeline ---

.PHONY: data-pipeline generate-sgfs collect-activations train-sae

# Variables for the data pipeline
SGF_DIR := generated_sgfs
ACTIVATIONS_DIR := activations
# This is the default model downloaded by the scripts if not present.
MODEL_URL := https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b28c512nbt-s12404017920-d5711392113.zip
MODEL_ZIP := $(notdir $(MODEL_URL))
ACTIVATIONS_FILE := $(ACTIVATIONS_DIR)/activations.pt
SAE_MODEL_FILE := $(ACTIVATIONS_DIR)/sae.pt

# Target to run the full data generation and processing pipeline.
data-pipeline: $(SAE_MODEL_FILE)
	@echo "--- Data pipeline complete. Trained model at $(SAE_MODEL_FILE) ---"

# Target to generate synthetic SGF data.
generate-sgfs: $(SGF_DIR)

$(SGF_DIR): install
	@echo "--- Generating synthetic SGF data ---"
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) generate_sgfs.py --output-dir $(SGF_DIR)

# Target to collect activations from the generated SGFs.
collect-activations: $(ACTIVATIONS_FILE)

$(MODEL_ZIP): $(INSTALL_STAMP)
	@echo "--- Downloading Model ---"
	@$(VENV_PYTHON) -c "import requests, sys, os; url='$(MODEL_URL)'; filename='$(MODEL_ZIP)'; tmp_filename=filename+'.part'; headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}; print(f'Downloading {url}...', file=sys.stderr); res = requests.get(url, headers=headers, stream=True); res.raise_for_status(); with open(tmp_filename, 'wb') as f: f.writelines(res.iter_content(8192)); os.rename(tmp_filename, filename)"

$(ACTIVATIONS_FILE): $(SGF_DIR) $(MODEL_ZIP)
	@echo "--- Collecting activations ---"
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) collect_activations.py --sgf-dir $(SGF_DIR) --model-path $(MODEL_ZIP) --output-file $(ACTIVATIONS_FILE)

# Target to run the SAE training script.
train-sae: $(SAE_MODEL_FILE)

$(SAE_MODEL_FILE): $(ACTIVATIONS_FILE)
	@echo "--- Training SAE model ---"
	@$(VENV_PYTHON) train_sae.py \
		--activations-file $(ACTIVATIONS_FILE) \
		--output-model-file $(SAE_MODEL_FILE) \
		--epochs 2 \
		--batch-size 32 \
		--lr 1e-4 \
		--l1-lambda 1e-3


# Target to clean up the project directory.
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -f test.sgf test2.sgf
	rm -rf $(patsubst %.zip,%,$(MODEL_ZIP))
	rm -rf $(SGF_DIR) $(ACTIVATIONS_DIR)
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Cleaning submodules..."
	@git submodule foreach --recursive git clean -fdx
	@git submodule foreach --recursive git reset --hard
