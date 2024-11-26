# QuantumToolbox Benchmarks

Lectures on numerical simulation of Quantum Physics and Quantum Optics. Most of the code uses the [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) package.

## How to build the Website

The website is built using [Quarto](https://quarto.org). The steps to build the website are:

1. Activate the Python virtual environment: `source pyenv/bin/activate`
2. Load the environment variables: `source _environment`
3. Run Quarto: `quarto publish gh-pages --no-browser`
