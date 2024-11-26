# Makefile for building and publishing the Quarto website

# Variables
PYENV=pyenv/bin/activate
ENV_VARS=_environment

# Default target
all: help

# Activate Python environment
activate:
	source $(PYENV)

# Target to load environment variables and run Quarto
publish:
	activate
	source $(ENV_VARS) && \
	quarto publish gh-pages --no-browser

# Help target
help:
	@echo "Usage:"
	@echo "  make activate     - Print the command to activate the Python environment"
	@echo "  make publish      - Build and publish the Quarto website"