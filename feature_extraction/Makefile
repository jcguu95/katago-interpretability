# Makefile for KataGo Feature Extractor component

.PHONY: all test test-full clean help

# Variables, relative to this directory
VENV_PYTHON := ../venv/bin/python

all: help

help:
	@echo "This Makefile is meant to be called from the root Makefile."

test:
	@echo "Running feature extractor tests..."
	@cd .. && PYTHONPATH=.:./katago/python $(subst ../,,$(VENV_PYTHON)) -m unittest feature_extraction.test_extract_katago_features

test-full:
	@echo "Running full feature extractor tests (including model download)..."
	@cd .. && ALLOW_MODEL_DOWNLOAD=1 PYTHONPATH=.:./katago/python $(subst ../,,$(VENV_PYTHON)) -m unittest feature_extraction.test_extract_katago_features

clean:
	@echo "Cleaning up feature_extraction directory..."
	rm -f test.sgf test2.sgf
	rm -f *.zip
	rm -rf kata1-*
	find . -type d -name "__pycache__" -exec rm -r {} +
