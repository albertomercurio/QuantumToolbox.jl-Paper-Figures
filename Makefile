# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate

qutip-benchmark:
	bash -c "source $(PYENV) && python src/python/qutip_benchmarks.py"

dynamiqs-benchmark:
	bash -c "source $(PYENV) && python src/python/dynamiqs_benchmarks.py"

render:
	quarto render

# run Quarto publish command
publish:
	quarto publish gh-pages --no-browser

# Default target
all-render: qutip-benchmark dynamiqs-benchmark render

all-publish: qutip-benchmark dynamiqs-benchmark render publish

# Help target
help:
	@echo "Usage:"
	@echo "  make publish      - Build and publish the Quarto website"