# Makefile for FUS replication
#
# Single-command reproduction targets:
#   make data        download and unzip MovieLens-100k into ml-100k/
#   make all         run all three streams end-to-end
#   make core        run the core FUS + CF implementation only
#   make cross_check run the independent FUS cross-check only
#   make baselines   run the PF + GIM reference baselines only
#   make figs        regenerate the four paper-style figures
#   make tests       run the sanity-check test suite
#   make clean       remove __pycache__ and stale results
#
# On Windows without GNU Make, the equivalent Python entry point is run_all.py.

PYTHON ?= python
ML100K_URL := https://files.grouplens.org/datasets/movielens/ml-100k.zip

.PHONY: all data core cross_check baselines figs tests clean help

help:
	@echo "Targets: data | all | core | cross_check | baselines | figs | tests | clean"

data:
	@if [ ! -f "ml-100k/u.data" ]; then \
		echo "Downloading MovieLens-100k..."; \
		curl -L -o ml-100k.zip $(ML100K_URL); \
		unzip -o ml-100k.zip; \
		rm ml-100k.zip; \
	else \
		echo "ml-100k/u.data already present, skipping download."; \
	fi

core:
	cd core/code && $(PYTHON) eval.py

cross_check:
	cd cross_check/code && $(PYTHON) eval_fus.py

baselines:
	cd baselines/code && $(PYTHON) eval.py

all: data cross_check baselines core
	@echo "All three streams complete. Results in core/, cross_check/, baselines/ results/ folders."

figs: data
	cd core/code && $(PYTHON) eval.py

tests:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Removed __pycache__ directories and .pyc files."
