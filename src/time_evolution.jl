# %% [markdown]
# ---
# title: "Time Evolution solvers: from the Schrödinger equation to the Lindblad master equation and Monte Carlo wave function method"
# author: "Alberto Mercurio"

# engine: julia
# ---

# ## Introduction

# The Hamiltonian of a quantum system gives us all the information about the system's dynamics. Indeed, given a quantum state $|\psi(0)\rangle$, the corresponding time-evolved state $|\psi(t)\rangle$ is given by the Schrödinger equation ($\hbar = 1$)

# $$
# i\frac{d}{dt}|\psi(t)\rangle = \hat{H} |\psi(t)\rangle \, ,
# $$

# where $\hat{H}$ is the Hamiltonian operator. Aside from a few solution that can be found analytically, the Schrödinger equation is usually solved numerically. Thanks to the [`sesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.sesolve) function, *QuantumToolbox.jl* provides a simple way to solve the Schrödinger equation for a given Hamiltonian and initial state.

# As an example, let's consider the Jaynes-Cummings (JC) model, which describes the interaction between a two-level atom and a quantized electromagnetic field. The Hamiltonian of the JC model is given by

# $$
# \hat{H} = \omega_c \hat{a}^\dagger \hat{a} + \omega_a \hat{\sigma}_z + g (\hat{a} \hat{\sigma}_+ + \hat{a}^\dagger \hat{\sigma}_-) \,
# $$

# where $\omega_c$ and $\omega_a$ are the frequencies of the cavity and the atom, respectively, $g$ is the coupling strength, and $\hat{a}$, $\hat{a}^\dagger$, $\hat{\sigma}_z$, $\hat{\sigma}_+$, and $\hat{\sigma}_-$ are the annihilation, creation, and Pauli operators, respectively.

# Let's consider the case where $\omega_c = \omega_a = 1$, $g = 0.1$, and the initial state is the vacuum state of the cavity. In the following, we will solve the Schrödinger equation for this system and compute the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time.

# We start by importing the packages and all the settings of the plots.

# %%
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

xlims!(ax, tlist[1], tlist[end])
ylims!(ax, 0, nothing)

fig

# %% [markdown]
# Its is important to note that the memory allocated does not depend on the number of points in the time grid or the final time. This is because the solver uses the most efficient algorithms given by the *DifferentialEquations.jl* package [@christopher_rackauckas_2024_differentialequations], computing intermediate steps in-place. It only allocates extra memory if the user requests the storage of the states with the `saveat` keyword argument, specifying the times at which the states are saved.

# ## Master Equation

# The Schrödinger equation is a unitary evolution equation, which describes the dynamics of a closed quantum system. However, in many cases, the system is not isolated, as it interacts with an environment. In this case, the dynamics of the system is described by the Lindblad master equation

# $$
# \frac{d}{dt}\hat{\rho} = -i [\hat{H}, \hat{\rho}] + \sum_k \mathcal{D}[\hat{C}_k] \hat{\rho} \, ,
# $$

# where $\hat{\rho}$ is the density matrix of the system, $\hat{H}$ is the Hamiltonian operator, $\hat{C}_k$ are the collapse operators, and 

# $$
# \mathcal{D}[\hat{C}] \hat{\rho} = \hat{C} \hat{\rho} \hat{C}^\dagger - \frac{1}{2} \{ \hat{C}^\dagger \hat{C}, \hat{\rho} \}
# $$

# is the Lindblad dissipator. *QuantumToolbox.jl* provides the [`mesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.mesolve) function to solve the master equation for a given Hamiltonian, initial state, and collapse operators.

# As an example, let's consider the case where both the cavity and the atom are coupled to a zero-temperature bath. The collapse operators are given by

# $$
# \hat{L}_1 = \sqrt{\kappa} \hat{a} \, , \quad \hat{L}_2 = \sqrt{\gamma} \hat{\sigma}_- \, ,
# $$

# where $\kappa$ and $\gamma$ are the decay rates of the cavity and the atom, respectively. We will solve the master equation for this system and compute the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time.

# %%
κ = 0.01
γ = 0.01

L1 = sqrt(κ) * a
L2 = sqrt(γ) * σm

c_ops = [L1, L2]

tlist = range(0, 10*pi/g, 1000)
res_me = mesolve(H, ψ0, tlist, c_ops, e_ops=e_ops)

# %% [markdown]
# Let's plot the results.

# %%
fig = Figure(size=(300, 150))
ax = Axis(fig[1, 1], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle")

lines!(ax, tlist, real(res_me.expect[1, :]), color=:navy)
lines!(ax, tlist, exp.(-κ * tlist), color=:red, linestyle=:dash)

xlims!(ax, tlist[1], tlist[end])
ylims!(ax, 0, nothing)

fig

# %% [markdown]
# The dashed line represents the exponential decay of the cavity population due to the cavity decay rate $\kappa$. The master equation solver is able to capture this behavior, as well as the oscillations due to the atom-cavity coupling.

# ## Monte Carlo Quantum Trajectories

# The Monte Carlo quantum trajectories method is a stochastic method to solve the time evolution of an open quantum system [@Dalibard1992Wave_function],[@Dum1992Monte],[@Molmer1993Monte],[@Carmichael1993An]. The idea is to simulate the evolution of the system by sampling the trajectories of the quantum state through a non-unitary evolution of a pure state $|\psi(t)\rangle$. The Monte Carlo method is particularly useful when the Hilbert space of the system is large, as it avoids the direct computation of the density matrix and Liouvillian superoperator. *QuantumToolbox.jl* provides the [`mcsolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.mcsolve) function to solve the open quantum system dynamics using the Monte Carlo method. The system evolves according to the following Schrödinger equation

# $$
# i\hbar\frac{d}{dt}|\psi(t)\rangle = \hat{H}_\mathrm{eff} |\psi(t)\rangle \, , \qquad \hat{H}_\mathrm{eff} = \hat{H} - i \frac{1}{2} \sum_k \hat{C}_k^\dagger \hat{C}_k \, ,
# $$

# where $\hat{H}_\mathrm{eff}$ is the effective non-Hermitian Hamiltonian. Being the evolution non-unitary, the norm of the state is not conserved, decreasing in time due to the dissipation. A quantum jump occurs when the norm of the state reaches a random threshold, previously sampled from a uniform distribution between 0 and 1. If many jump operators are present, the probability of each jump is equal to

# $$
# P_k = \frac{\langle \psi(t) | \hat{C}_k^\dagger \hat{C}_k | \psi(t) \rangle}{\sum_j \langle \psi(t) | \hat{C}_j^\dagger \hat{C}_j | \psi(t) \rangle} \, .
# $$

# As an example, let's consider the same system as before, where both the cavity and the atom are coupled to a zero-temperature bath.

# %%
tlist = range(0, 10*pi/g, 1000)
res_mc = mcsolve(H, ψ0, tlist, c_ops, e_ops=e_ops, ntraj=100)

# %% [markdown]
# As can be seen, an additional keyword argument `ntraj` is passed to the `mcsolve` function, specifying the number of quantum trajectories to simulate. Let's plot the results.

# %%
fig = Figure(size=(300, 150))
ax = Axis(fig[1, 1], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle")

lines!(ax, tlist, real(res_mc.expect[1, :]), color=:navy)

xlims!(ax, tlist[1], tlist[end])
ylims!(ax, 0, nothing)

fig

# %% [markdown]
