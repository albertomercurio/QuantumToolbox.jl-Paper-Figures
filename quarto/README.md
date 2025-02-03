# QuantumToolbox.jl Paper Figures

This repository contains the code used to generate the figures in the paper about [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) package.

## How to build the Website

The website is built using [Quarto](https://quarto.org). The steps to build the website are:

1. Install the Python virtual environment (only once): `python3 -m venv pyenv`
2. Activate the Python virtual environment and install the dependencies: `source pyenv/bin/activate && pip install -r requirements.txt`
3. Load the environment variables: `source _environment`
4. Run Quarto: `quarto publish gh-pages --no-browser`
