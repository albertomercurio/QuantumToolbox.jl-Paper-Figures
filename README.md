# QuantumToolbox Benchmarks

Benchmarks for the [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) package.

## How to run the benchmarks for the package comparison

1. Create a virtual environment for Python: `python3 -m venv pyenv`
2. Activate the virtual environment: `source pyenv/bin/activate`
3. Install the requirements: `pip install -r requirements.txt`
4. Load the environment variables: `source _environment`
5. Run the benchmarks: `julia --project package_comparison.jl`

## How to build the Website

The website is built using [Quarto](https://quarto.org). The steps to build the website are:

1. Activate the Python virtual environment: `source pyenv/bin/activate`
2. Load the environment variables: `source _environment`
3. Run Quarto: `quarto publish gh-pages --no-browser`
