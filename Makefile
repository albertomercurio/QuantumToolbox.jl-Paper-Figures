# Makefile for building and publishing the Quarto website

qutip-benchmark:
	bash -c "python src/benchmarks/python/qutip_benchmarks.py"

dynamiqs-benchmark:
	bash -c "python src/benchmarks/python/dynamiqs_benchmarks.py"

quantumoptics-benchmark:
	bash -c "julia --project src/benchmarks/julia/quantumoptics.jl"

quantumtoolbox-benchmark:
	bash -c "julia --project src/benchmarks/julia/quantumtoolbox.jl"

help:
	@echo "Usage:"
	@echo "  make qutip-benchmark         - Run the QuTiP benchmark"
	@echo "  make dynamiqs-benchmark      - Run the Dynamiqs benchmark"
	@echo "  make quantumoptics-benchmark - Run the QuantumOptics benchmark"
	@echo "  make quantumtoolbox-benchmark - Run the QuantumToolbox benchmark"
