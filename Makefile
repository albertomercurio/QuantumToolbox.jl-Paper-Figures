# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate
ENV_VARS:=_environment

# Default target
all: activate publish

# Activate Python environment and load environment variables
activate:
	@bash -c "source ${PYENV} && source ${ENV_VARS}"

# Target to load environment variables and run Quarto
publish:
	quarto publish gh-pages --no-browser

# Help target
help:
	@echo "Usage:"
	@echo "  make activate     - Print the command to activate the Python environment"
	@echo "  make publish      - Build and publish the Quarto website"