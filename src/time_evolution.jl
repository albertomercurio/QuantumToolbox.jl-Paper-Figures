# %% [markdown]
# ---
# title: "Time Evolution solvers: from the Schrödinger equation to the Lindblad master equation and Monte Carlo wave function method"
# author: "Alberto Mercurio"

# engine: julia
# ---

# ## Introduction

# The Hamiltonian of a quantum system gives us all the information about the system's dynamics. Given a quantum state $|\psi(0)\rangle$, the corresponding time-evolved state $|\psi(t)\rangle$ is given by the Schrödinger equation ($\hbar = 1$)

# $$
# i\frac{d}{dt}|\psi(t)\rangle = \hat{H} |\psi(t)\rangle \, ,
# $$ {#eq-schrodinger}

# where $\hat{H}$ is the Hamiltonian operator. Aside from a few solutions that can be found analytically, the Schrödinger equation is usually solved numerically. Thanks to the [`sesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.sesolve) function, *QuantumToolbox.jl* provides a simple way to solve the Schrödinger equation for a given Hamiltonian and initial state.

# As a pedagogical example, let's consider the Jaynes-Cummings (JC) model [@Jaynes1963Comparison], which describes the interaction between a two-level atom and a quantized electromagnetic field. The Hamiltonian of the JC model is given by

# $$
# \hat{H} = \omega_c \hat{a}^\dagger \hat{a} + \omega_a \hat{\sigma}_z + g (\hat{a} \hat{\sigma}_+ + \hat{a}^\dagger \hat{\sigma}_-) \,
# $$

# where $\omega_c$ and $\omega_a$ are the frequencies of the cavity and the atom, respectively, $g$ is the coupling strength, and $\hat{a}$, $\hat{a}^\dagger$, and $\hat{\sigma}_j$ are the annihilation, creation, and Pauli operators, respectively.

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
a = tensor(destroy(N), qeye(2)) # Annihilation operator of the cavity

σz = tensor(qeye(N), sigmaz()) # Pauli-Z operator of the atom
σm = tensor(qeye(N), sigmam()) # Annihilation operator of the atom
σp = tensor(qeye(N), sigmap()) # Creation operator of the atom

H = ωc * a' * a + ωa / 2 * σz + g * (a * σp + a' * σm)

# %% [markdown]
# The time evolution can be simply simulated by

# %%
ψ0 = tensor(fock(N, 0), basis(2, 0)) # Initial state: vacuum state of the cavity and excited state of the atom

e_ops = [a' * a] # Expectation value of the number operator

tlist = range(0, 10*pi/g, 1000)
res_se = sesolve(H, ψ0, tlist, e_ops=e_ops)

# %% [markdown]
# Note how the syntax is mostly similar to the one of QuTiP.
#
# As can be seen from @fig-se-me-mc-solve-1, the number of photons oscillates between the cavity and the atom, showing the vacuum Rabi oscillations.
# 
# Its is important to note that the memory allocated does not depend on the number of points in the time grid or the final time. This is because the solver uses the most efficient algorithms given by the [*DifferentialEquations.jl*](https://github.com/SciML/DifferentialEquations.jl?tab=readme-ov-file) package [@christopher_rackauckas_2024_differentialequations] [@RackauckasDifferentialEquations_jl2017], computing intermediate steps in-place. It only allocates extra memory if the user requests the storage of the states with the `saveat` keyword argument, specifying the times at which the states are saved.

# ## Master Equation

# The Schrödinger equation is a unitary evolution equation, which describes the dynamics of a closed quantum system. However, in many cases, the system is not isolated, as it interacts with an environment. In the case the system is weakly interacting with a Markovian environment, the dynamics of the system is described by the Lindblad master equation

# $$
# \frac{d}{dt}\hat{\rho} = -i [\hat{H}, \hat{\rho}] + \sum_k \mathcal{D}[\hat{C}_k] \hat{\rho} \, ,
# $$

# where $\hat{\rho}$ is the density matrix of the system, $\hat{H}$ is the Hamiltonian operator, $\hat{C}_k$ are the collapse operators, and 

# $$
# \mathcal{D}[\hat{C}] \hat{\rho} = \hat{C} \hat{\rho} \hat{C}^\dagger - \frac{1}{2} \left( \hat{C}^\dagger \hat{C} \hat{\rho} - \hat{\rho} \hat{C}^\dagger \hat{C} \right)
# $$

# is the Lindblad dissipator. *QuantumToolbox.jl* provides the [`mesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.mesolve) function to solve the master equation for a given Hamiltonian, initial state, and collapse operators.

# As an example, let's consider the case where both the cavity and the atom are coupled to a zero-temperature bath. The collapse operators are given by

# $$
# \hat{C}_1 = \sqrt{\kappa} \hat{a} \, , \quad \hat{C}_2 = \sqrt{\gamma} \hat{\sigma}_- \, ,
# $$

# where $\kappa$ and $\gamma$ are the decay rates of the cavity and the atom, respectively. We will solve the master equation for this system and compute the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time.

# %%
κ = 0.01
γ = 0.01

C1 = sqrt(κ) * a
C2 = sqrt(γ) * σm

c_ops = [C1, C2]

res_me = mesolve(H, ψ0, tlist, c_ops, e_ops=e_ops)

# %% [markdown]
# @fig-se-me-mc-solve-2 shows the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time. Contrary to the Schrödinger equation, the cavity population decays exponentially due to the cavity decay rate $\kappa$. Indeed, the dashed gray line represents the exponential decay of the cavity population due to the cavity decay rate $\kappa$. The master equation solver is able to capture this behavior, as well as the oscillations due to the atom-cavity coupling.
#
# When only the steady state is of interest, the [`steadystate`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.steadystate) function can be used to directly compute the steady state, without the need to solve the master equation in time. This function allows several algorithms such as direct, iterative, and eigenvalue-based methods [@Nation2015Iterative][@Nation2015Steady_state].

# ## Monte Carlo Quantum Trajectories

# The Monte Carlo quantum trajectories' approach is a stochastic method to solve the time evolution of an open quantum system [@Dalibard1992Wave_function],[@Dum1992Monte],[@Molmer1993Monte],[@Carmichael1993An]. The idea is to simulate the evolution of the system by sampling the trajectories of the quantum state through a non-unitary evolution of a pure state $|\psi(t)\rangle$. The Monte Carlo method is particularly useful when the Hilbert space of the system is large, as it avoids the direct computation of the density matrix and Liouvillian superoperator. Moreover, each trajectory can be easily parallelized, especially when using the *Distributed.jl* package in Julia.  *QuantumToolbox.jl* provides the [`mcsolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.mcsolve) function to solve the open quantum system dynamics using the Monte Carlo method. The system evolves according to the Schrödinger equation expressed in (@eq-schrodinger), but with the non-Hermitian effective Hamiltonian

# $$
# \hat{H}_\mathrm{eff} = \hat{H} - i \frac{1}{2} \sum_k \hat{C}_k^\dagger \hat{C}_k \, .
# $$

# Being the evolution non-unitary, the norm of the state is not conserved, decreasing in time due to the dissipation. A quantum jump occurs when the norm of the state reaches a random threshold, previously sampled from a uniform distribution between 0 and 1. If many jump operators are present, the probability of each jump is equal to

# $$
# P_k (t) = \frac{\langle \psi(t) | \hat{C}_k^\dagger \hat{C}_k | \psi(t) \rangle}{\sum_j \langle \psi(t) | \hat{C}_j^\dagger \hat{C}_j | \psi(t) \rangle} \, .
# $$

# As an example, let's consider the same system as before, where both the cavity and the atom are coupled to a zero-temperature bath. In addition to other keyword arguments, we also use the `ntraj` and `rng` arguments to specify the number of quantum trajectories and the random seed, respectively. The second argument is optional, but it is useful for reproducibility.

# %%
using Random

# We use the optional random seed for reproducibility
rng = MersenneTwister(10)

res_mc = mcsolve(H, ψ0, tlist, c_ops, e_ops=e_ops, ntraj=100, rng=rng)

# %% [markdown]
# @fig-se-me-mc-solve-3 shows the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time. The solid blue line represents the average over 100 quantum trajectories, while the dashed light blue line represents a single quantum trajectory. The single trajectory shows the stochastic nature of the method, where the cavity population can jump due to the random quantum jumps.

# %%
#| label: fig-se-me-mc-solve
#| layout-ncol: 3
fig = Figure(size=(plot_figsize_width_pt, 0.35*plot_figsize_width_pt))

ax_se = Axis(fig[1, 1], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle", title="sesolve")
ax_me = Axis(fig[1, 2], xlabel="Time", title="mesolve")
ax_mc = Axis(fig[1, 3], xlabel="Time", title="mcsolve")

lines!(ax_se, tlist, real(res_se.expect[1, :]), color=:dodgerblue4)

lines!(ax_me, tlist, real(res_me.expect[1, :]), color=:dodgerblue4)
lines!(ax_me, tlist, exp.(-κ * tlist), color=:dimgray, linestyle=:dash, linewidth=1.25)
text!(ax_me, 40, 0.7, text=L"e^{-\kappa t}", color=:dimgray, fontsize=9)

lines!(ax_mc, tlist, real(res_mc.expect[1, :]), color=:dodgerblue4, label="100 Traj.")
lines!(ax_mc, tlist, real(res_mc.expect_all[1, :, 11]), color=:lightsteelblue, linewidth=1.25, linestyle=:dash, label="1 Traj.")

axislegend(ax_mc, position=:rt, padding=0)

xlims!(ax_se, tlist[1], tlist[end])
xlims!(ax_me, tlist[1], tlist[end])
xlims!(ax_mc, tlist[1], tlist[end])

ylims!(ax_se, 0, 1.2)
ylims!(ax_me, 0, 1.2)
ylims!(ax_mc, 0, 1.2)

text!(ax_se, 0.02, 0.98, text="(a)", space=:relative, align=(:left, :top), font=:bold)
text!(ax_me, 0.02, 0.98, text="(b)", space=:relative, align=(:left, :top), font=:bold)
text!(ax_mc, 0.02, 0.98, text="(c)", space=:relative, align=(:left, :top), font=:bold)

hideydecorations!(ax_me, ticks=false)
hideydecorations!(ax_mc, ticks=false)

linkyaxes!(ax_se, ax_me, ax_mc)

colgap!(fig.layout, 8)

# For the LaTeX document
# save("../figures/se-me-mc-solve.pdf", fig, pt_per_unit = 1.0)

fig
