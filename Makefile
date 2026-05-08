# Makefile for FUS replication
#
# Single-command reproduction targets:
#   make data    - download and unzip MovieLens-100k into ml-100k/
#   make all     - run all three streams (Person 1, 2, 3) end-to-end
#   make person1 - run Person 1 only (FUS + CF)
#   make person2 - run Person 2 only (independent FUS cross-check)
#   make person3 - run Person 3 only (PF + GIM)
#   make figs    - regenerate the four paper-style figures
#   make tests   - run the sanity-check test suite
#   make clean   - remove __pycache__ and stale results
#
# On Windows without GNU Make, the equivalent Python entry point is run_all.py.

PYTHON ?= python
ML100K_URL := https://files.grouplens.org/datasets/movielens/ml-100k.zip

.PHONY: all data person1 person2 person3 figs tests clean help

help:
	@echo "Targets: data | all | person1 | person2 | person3 | figs | tests | clean"

data:
	@if [ ! -f "ml-100k/u.data" ]; then \
		echo "Downloading MovieLens-100k..."; \
		curl -L -o ml-100k.zip $(ML100K_URL); \
		unzip -o ml-100k.zip; \
		rm ml-100k.zip; \
	else \
		echo "ml-100k/u.data already present, skipping download."; \
	fi

person1:
	cd Person_1_Faithful_Baseline/code && $(PYTHON) eval.py

person2:
	cd Person_2_Faithful_FUS/code && $(PYTHON) eval_fus.py

person3:
	cd Person_3_Reference_Baselines/code && $(PYTHON) eval.py

all: data person2 person3 person1
	@echo "All three streams complete. Results in Person_*/results/."

figs: data
	cd Person_1_Faithful_Baseline/code && $(PYTHON) eval.py

tests:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Removed __pycache__ directories and .pyc files."
