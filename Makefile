# Makefile for building and publishing the Quarto website

# Variables
PYENV:=pyenv/bin/activate

qutip-benchmark:
	bash -c "set -a && source _environment && set +a && python src/benchmarks/python/qutip_benchmarks.py"

dynamiqs-benchmark:
	bash -c "set -a && source _environment && set +a && python src/benchmarks/python/dynamiqs_benchmarks.py"

quantumoptics-benchmark:
	bash -c "set -a && source _environment && set +a && julia --project src/benchmarks/julia/quantumoptics.jl"

quantumtoolbox-benchmark:
	bash -c "set -a && source _environment && set +a && julia --project src/benchmarks/julia/quantumtoolbox.jl"

render:
	bash -c "set -a && source _environment && set +a && quarto render"

# run Quarto publish command
publish:
	bash -c "set -a && source _environment && set +a && quarto publish gh-pages --no-browser"

# Default target
all-render: qutip-benchmark dynamiqs-benchmark quantumoptics-benchmark quantumtoolbox-benchmark render

all-publish: qutip-benchmark dynamiqs-benchmark quantumoptics-benchmark quantumtoolbox-benchmark render publish

# Help target
help:
	@echo "Usage:"
	@echo "  make publish      - Build and publish the Quarto website"