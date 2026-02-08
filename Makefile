# Makefile for KataGo Feature Extractor

.PHONY: all test install clean help

# Use the system's python3.
PYTHON := python3

# Define the virtual environment directory.
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

# Default target when 'make' is run without arguments.
all: help

# Help target to display available commands.
help:
	@echo "Available commands:"
	@echo "  make install    - Set up the virtual environment and install all dependencies."
	@echo "  make test       - Run the test suite (will also run install if needed)."
	@echo "  make clean      - Remove virtual environment and other generated files."
	@echo ""

# Target to set up the virtual environment directory if it doesn't exist.
$(VENV_DIR):
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

# Target to install dependencies into the virtual environment.
# This depends on the venv directory existing.
install: $(VENV_DIR)
	@echo "Initializing Git submodules..."
	@git submodule update --init --recursive
	@echo "Installing dependencies..."
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install ./sgfmill

# Target to run tests. This will automatically set up the environment first if needed.
test: install
	@echo "Running tests..."
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) -m unittest test_extract_katago_features.py

# Target to clean up the project directory.
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -f test.sgf test2.sgf
	rm -rf kata1-*
	find . -type d -name "__pycache__" -exec rm -r {} +
