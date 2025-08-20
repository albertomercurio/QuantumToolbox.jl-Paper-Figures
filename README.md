# QuantumToolbox.jl Paper Figures

This repository contains the code used to generate the figures in the paper about [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) package.

## How to run the benchmarks

1. Set the environment variable `JULIA_NUM_THREADS` to a suitable value, according to the number of CPU cores available on your machine.
2. Install a python environment and install the dependencies with `pip install -r requirements.txt`. For a CPU-only machine, use `pip install -r requirements-cpuonly.txt`.
3. Run the benchmarks using the `make` command:
    - `make qutip-benchmark`
    - `make dynamiqs-benchmark`
    - `make quantumoptics-benchmark`
    - `make quantumtoolbox-benchmark`

> [!NOTE]
> All the following commands should be run under the root folder of this repository: `/path/to/QuantumToolbox.jl-Paper-Figures/`
