# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate

# Default target
all: activate publish

# Activate Python environment and load environment variables
activate:
	@bash -c "source ${PYENV} && set -a && source ${ENV_VARS} && set +a"

# run Quarto publish command
publish:
	quarto publish gh-pages --no-browser

# Help target
help:
	@echo "Usage:"
	@echo "  make activate     - Print the command to activate the Python environment"
	@echo "  make publish      - Build and publish the Quarto website"