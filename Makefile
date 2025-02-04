# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate

# Default target
all: publish

# run Quarto publish command
publish:
	quarto publish gh-pages --no-browser

# Help target
help:
	@echo "Usage:"
	@echo "  make publish      - Build and publish the Quarto website"