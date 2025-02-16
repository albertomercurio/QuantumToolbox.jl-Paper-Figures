# QuantumToolbox.jl Paper Figures

This repository contains the code used to generate the figures in the paper about [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) package.

## How to build the Website

The website is built using [Quarto](https://quarto.org). The steps to build the website are:

1. Install the Python virtual environment (only once): `python3 -m venv pyenv`
2. Activate the Python virtual environment and install the dependencies: `source pyenv/bin/activate && pip install -r requirements.txt`
3. Load the environment variables: `set -a && source _environment && set +a`
4. Run Quarto: `all-render` or `all-publish`

> [!NOTE]
> All the following commands should be run under the root folder of this repository: `/path/to/quantumtoolbox.jl-paper-figures/`

You can even just render the website locally by running `quarto render` and open the `index.html` file in your browser.
