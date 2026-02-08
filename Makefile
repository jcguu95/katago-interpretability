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

# Target to set up the virtual environment directory if it doesn't exist.
$(VENV_DIR):
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

# The main install target. It depends on a stamp file.
install: $(INSTALL_STAMP)

# The stamp file is created after installation. It depends on requirements.txt.
# If requirements.txt changes, this rule will re-run.
$(INSTALL_STAMP): $(VENV_DIR) requirements.txt
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

# Target to run tests. This will automatically set up the environment first if needed.
test: install
	@echo "Running tests..."
	@PYTHONPATH=$(shell pwd)/katago/python $(VENV_PYTHON) -m unittest test_extract_katago_features.py

# Target for a full, clean test run.
test-full: clean test

# Target to clean up the project directory.
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -f test.sgf test2.sgf
	rm -rf kata1-*
	find . -type d -name "__pycache__" -exec rm -r {} +
