# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate

qutip-benchmark:
	bash -c "source $(PYENV) && python src/python/qutip_benchmarks.py"

dynamiqs-benchmark:
	bash -c "source $(PYENV) && python src/python/dynamiqs_benchmarks.py"

# Default target
all: publish

# run Quarto publish command
publish:
	quarto publish gh-pages --no-browser

# Help target
help:
	@echo "Usage:"
	@echo "  make publish      - Build and publish the Quarto website"