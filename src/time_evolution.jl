# %% [markdown]
# ---
# title: "Time Evolution solvers: from the Schrödinger equation to the Lindblad master equation and Monte Carlo wave function method"
# author: "Alberto Mercurio"

# engine: julia
# ---

# ## Introduction

# The Hamiltonian of a quantum system gives us all the information about the system's dynamics. Indeed, given a quantum state $|\psi(0)\rangle$, the corresponding time-evolved state $|\psi(t)\rangle$ is given by the Schrödinger equation

# $$
# i\hbar\frac{d}{dt}|\psi(t)\rangle = \hat{H} |\psi(t)\rangle \, ,
# $$

# where $\hat{H}$ is the Hamiltonian operator. Aside from a few solution that can be found analytically, the Schrödinger equation is usually solved numerically. Thanks to the [`sesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.sesolve) function, QuantumToolbox.jl provides a simple way to solve the Schrödinger equation for a given Hamiltonian and initial state.

# As an example, let's consider the Jaynes-Cummings (JC) model, which describes the interaction between a two-level atom and a quantized electromagnetic field. The Hamiltonian of the JC model is given by

# $$
# \hat{H} = \omega_c \hat{a}^\dagger \hat{a} + \omega_a \hat{\sigma}_z + g (\hat{a} \hat{\sigma}_+ + \hat{a}^\dagger \hat{\sigma}_-) \,
# $$

# where $\omega_c$ and $\omega_a$ are the frequencies of the cavity and the atom, respectively, $g$ is the coupling strength, and $\hat{a}$, $\hat{a}^\dagger$, $\hat{\sigma}_z$, $\hat{\sigma}_+$, and $\hat{\sigma}_-$ are the annihilation, creation, and Pauli operators, respectively.

# Let's consider the case where $\omega_c = \omega_a = 1$, $g = 0.1$, and the initial state is the vacuum state of the cavity. In the following, we will solve the Schrödinger equation for this system and compute the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time.

# We start by importing the packages and all the settings of the plots.

# %%
include("_environment.jl")

using QuantumToolbox
using CairoMakie

# Settings for the plots
include("_cairomakie_setup.jl")

# %% [markdown]
# We now define the parameters and the Hamiltonian of the JC model.

# %%
ωc = 1
ωa = 1
g = 0.1

N = 10 # Cutoff of the cavity Hilbert space
a = tensor(destroy(N), eye(2)) # Annihilation operator of the cavity

σz = tensor(eye(N), sigmaz()) # Pauli-Z operator of the atom
σm = tensor(eye(N), sigmam()) # Annihilation operator of the atom
σp = tensor(eye(N), sigmap()) # Creation operator of the atom

H = ωc * a' * a + ωa / 2 * σz + g * (a * σp + a' * σm)

# %% [markdown]
# We now perform the time evolution of the system.

# %%
ψ0 = tensor(fock(N, 0), basis(2, 0)) # Initial state: vacuum state of the cavity and excited state of the atom

e_ops = [a' * a] # Expectation value of the number operator

tlist = range(0, 10*pi/g, 1000)
res_se = sesolve(H, ψ0, tlist, e_ops=e_ops)

# %% [markdown]
# Note how the syntax is mostly similar to the one of QuTiP. Let's now plot the results.

# %%

fig = Figure(size=(300, 150))
ax = Axis(fig[1, 1], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle")

lines!(ax, tlist, real(res_se.expect[1, :]), color=:navy)

fig

# %% [markdown]
# Its is important to note that the memory allocated does not depend on the number of points in the time grid. This is because the solver uses the most efficient algorithms given by the *DifferentialEquations.jl* package [@christopher_rackauckas_2024_differentialequations].