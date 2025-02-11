# %% [markdown]
# ---
# title: "Time Evolution solvers: from the Schrödinger equation to the Lindblad master equation and Monte Carlo wave function method"
# author: "Alberto Mercurio"

# engine: julia
# ---

# ## Time-independent evolution

# ### Schrödinger Equation

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

N = 20 # Cutoff of the cavity Hilbert space
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
# As can be seen from @fig-se-me-mc-solve (a), the number of photons oscillates between the cavity and the atom, showing the vacuum Rabi oscillations.
# 
# Its is important to note that the memory allocated does not depend on the number of points in the time grid or the final time. This is because the solver uses the most efficient algorithms given by the [*DifferentialEquations.jl*](https://github.com/SciML/DifferentialEquations.jl?tab=readme-ov-file) package [@christopher_rackauckas_2024_differentialequations] [@RackauckasDifferentialEquations_jl2017], computing intermediate steps in-place. It only allocates extra memory if the user requests the storage of the states with the `saveat` keyword argument, specifying the times at which the states are saved.

# ### Master Equation

# The Schrödinger equation is a unitary evolution equation, which describes the dynamics of a closed quantum system. However, in many cases, the system is not isolated, as it interacts with an environment. In the case the system is weakly interacting with a Markovian environment, the dynamics of the system is described by the Lindblad master equation

# $$
# \frac{d}{dt}\hat{\rho} = \mathcal{L} \hat{\rho} = -i [\hat{H}, \hat{\rho}] + \sum_k \mathcal{D}[\hat{C}_k] \hat{\rho} \, ,
# $$

# where $\mathcal{L}$ is the Liouvillian superoperator, $\hat{\rho}$ is the density matrix of the system, $\hat{H}$ is the Hamiltonian operator, $\hat{C}_k$ are the collapse operators, and 

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
# @fig-se-me-mc-solve (b) shows the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time. Contrary to the Schrödinger equation, the cavity population decays exponentially due to the cavity decay rate $\kappa$. Indeed, the dashed gray line represents the exponential decay of the cavity population due to the cavity decay rate $\kappa$. The master equation solver is able to capture this behavior, as well as the oscillations due to the atom-cavity coupling.
#
# When only the steady state is of interest, the [`steadystate`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.steadystate) function can be used to directly compute the steady state, without the need to solve the master equation in time. This function allows several algorithms such as direct, iterative, and eigenvalue-based methods [@Nation2015Iterative][@Nation2015Steady_state].

# ### Monte Carlo Quantum Trajectories

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
# @fig-se-me-mc-solve (c) shows the expectation value of the number operator $\langle \hat{a}^\dagger \hat{a} \rangle$ as a function of time. The solid blue line represents the average over 100 quantum trajectories, while the dashed light blue line represents a single quantum trajectory. The single trajectory shows the stochastic nature of the method, where the cavity population can jump due to the random quantum jumps.

# ## Time-dependent evolution
# Although we have only considered the JC model, the solvers shown above are able to simulate a wide range of physical systems. However, in many cases, the Hamiltonian and-or the collapse operators are time-dependent. For instance, this can happen when a drive is applied to the system, or when dynamically changing the resonance frequency. Here, we will demonstrate how the time-dependent evolution can be seamlessly simulated using the same solvers of the time-independent case. As an example, we consider the driven optomechanical system [@Law1995Interaction][@Aspelmeyer2014Cavity][@Macri2018Nonperturbative]
# $$
# \hat{H} = \omega_c \hat{a}^\dagger \hat{a} + \omega_m \hat{b}^\dagger \hat{b} + \frac{g}{2} \left(\hat{a} + \hat{a}^\dagger\right)^2 \left(\hat{b} + \hat{b}^\dagger\right) + F \cos \left(\omega_\mathrm{d} t \right) \left( \hat{a} + \hat{a}^\dagger \right) \, ,
# $$ {#eq-optomechanical-driven}
# where $\hat{a}$ and $\hat{b}$ are the annihilation operators of the cavity and mechanical mode, respectively. The parameter $\omega_m$ represents the mechanical frequency, while $F$ and $\omega_\mathrm{d}$ denote the amplitude and frequency of the drive, respectively. It is worth noting that the time-dependence can not be removed by a unitary transformation, as the Hamiltonian contains counter-rotating terms on both the coupling and the drive. The time evolution is solved as before, with Hamiltonian being either a `Tuple` or a `QuantumObjectEvolution` type. Finally, we solve the open dynamics of the system by considering the cavity decay and the mechanical damping, with the collapse operators $\mathcal{D} [\hat{a}]$ and $\mathcal{D} [\hat{b}]$, respectively.

## %%
Nc = 10 # Cutoff of the cavity Hilbert space
Nm = 7 # Cutoff of the mechanical mode Hilbert space
ωc = 1
ωm = 2 * ωc
g = 0.05
κ = 0.01
γ = 0.01
F = 10*κ
ωd = ωc

# Functions for the time-dependent coefficients
coef(p, t) = p[1] * cos(p[2] * t)

a = tensor(destroy(Nc), qeye(Nm)) # Annihilation operator of the cavity
b = tensor(qeye(Nc), destroy(Nm)) # Annihilation operator of the mechanical mode

H = ωc * a' * a + ωm * b' * b + g/2 * (a + a')^2 * (b + b')
c_ops = [sqrt(κ) * a, sqrt(γ) * b]
e_ops = [a' * a]

H_td = (H, (a + a', coef))

ψ0 = tensor(fock(Nc, 0), fock(Nm, 0)) # Zero bare cavity and mechanical excitations

params = [F, ωd] # NamedTuple (F = F, ωd = ωd) is also supported

tlist2 = range(0, 5/κ, 5000)
res_me_td = mesolve(H_td, ψ0, tlist2, c_ops, e_ops=e_ops, params=params)

# %% [markdown]
# Notice that the `params` argument is also supported as a `NamedTuple` instead of a `Vector`. The evolution of the expectation value $\langle \hat{a}^\dagger \hat{a} \rangle$ is shown in @fig-se-me-mc-solve (d), where we can observe a stroboscopic behavior of the steady state due to the counter-rotating terms on both the coupling and the drive. This results in the impossibility of using the `steadystate` function to compute the steady state. To this end, *QuantumToolbox.jl* offers a solution by providing the `steadystate_floquet` function. The method extracts the time-averaged steady state by expanding it into Fourier components. Depending on the internal solver, it then solves either a matrix continued fraction problem, or a linear system [@majumdar2011probing][@papageorge2012bichromatic][@maragkou2013bichromatic][@Macri2022Spontaneous]. We first define the time-dependent equation of motion for the density matrix
# $$
# \frac{ d}{ d t} \hat{\rho} = \left[ \mathcal{L}_0 + \mathcal{L}_1 e^{i \omega_\mathrm{d} t} + \mathcal{L}_{-1} e^{-i \omega_\mathrm{d} t} \right] \hat{\rho} \, ,
# $$ {#eq-fourier-lindblad}
# where $\mathcal{L}_0$ is the time-independent Liouvillian superoperator, while $\mathcal{L}_{\pm 1}$ are the superoperators containing the drive terms. For example, in the case of the optomechanical system, the Liouvillian superoperators are given by
# $$
# \mathcal{L}_0 \hat{\rho} = -i \left[ \hat{H}, \hat{\rho} \right] + \kappa \mathcal{D}[\hat{a}] \hat{\rho} + \gamma \mathcal{D}[\hat{b}] \hat{\rho} \, ,
# $$
# $$
# \mathcal{L}_{\pm 1} \hat{\rho} = & -i \left[ \frac{F}{2} \left( \hat{a} + \hat{a}^\dagger \right), \hat{\rho} \right]
# $$
# At long times, all the transient dynamics are washed out, and the density matrix can be expanded in Fourier components of the form
# $$
# \hat{\rho} (t) = \sum_{n=-\infty}^{+\infty} \hat{\rho}_n e^{i n \omega_\mathrm{d} t} \, ,
# $$
# By substituting the expansion in the equation of motion, we obtain
# $$
# \sum_{n=-\infty}^{+\infty} i n \omega_\mathrm{d} \hat{\rho}_n e^{i n \omega_\mathrm{d} t} = \sum_{n=-\infty}^{+\infty} \left[ \mathcal{L}_0 + \mathcal{L}_1 e^{i \omega_\mathrm{d} t} + \mathcal{L}_{-1} e^{-i \omega_\mathrm{d} t} \right] \hat{\rho}_n e^{i n \omega_\mathrm{d} t} \, .
# $$
# Equating the coefficients of the series yields the tridiagonal recursion relation
# $$
# ( \mathcal{L}_0 - i n \omega_\mathrm{d} ) \hat{\rho}_n + \mathcal{L}_1 \hat{\rho}_{n-1} + \mathcal{L}_{-1} \hat{\rho}_{n+1} = 0 \, .
# $$
# After choosing a cutoff $n_\mathrm{max}$ for the Fourier components, the equation above defines a tridiagonal linear system of the form
# $$
# \mathbf{A} \cdot \mathbf{b} = 0
# $$
# where
# $$
# \mathbf{A} = \begin{pmatrix}
# \mathcal{L}_0 - i (-n_\mathrm{max}) \omega_\mathrm{d} & \mathcal{L}_{-1} & 0 & \cdots & 0 \\
# \mathcal{L}_1 & \mathcal{L}_0 - i (-n_\mathrm{max}+1) \omega_\mathrm{d} & \mathcal{L}_{-1} & \cdots & 0 \\
# 0 & \mathcal{L}_1 & \mathcal{L}_0 - i (-n_\mathrm{max}+2) \omega_\mathrm{d} & \cdots & 0 \\
# \vdots & \vdots & \vdots & \ddots & \vdots \\
# 0 & 0 & 0 & \cdots & \mathcal{L}_0 - i n_\mathrm{max} \omega_\mathrm{d}
# \end{pmatrix}
# $$
# and
# $$
# \mathbf{b} = \begin{pmatrix}
# \hat{\rho}_{-n_\mathrm{max}} \\
# \hat{\rho}_{-n_\mathrm{max}+1} \\
# \vdots \\
# \hat{\rho}_{0} \\
# \vdots \\
# \hat{\rho}_{n_\mathrm{max}-1} \\
# \hat{\rho}_{n_\mathrm{max}}
# \end{pmatrix}
# $$

# This will allow to simultaneously obtain all the Fourier components $\hat{\rho}_n$.

# To extract the time-averaged steady state, we need simply write

# %%
H_dr = F / 2 * (a + a')

# We compare the results for different values of n_max
ρss_td_1 = steadystate_floquet(H, H_dr, H_dr, ωd, c_ops, n_max=1, tol=1e-6)[1]
ρss_td_2 = steadystate_floquet(H, H_dr, H_dr, ωd, c_ops, n_max=2, tol=1e-6)[1]

# Number of photons at the steady state
expect_ss_td_1 = expect(a' * a, ρss_td_1)
expect_ss_td_2 = expect(a' * a, ρss_td_2)

# %% [markdown]

# Although we applied this method to the master equation, the same approach can be used for all the time evolution solvers of QuantumToolbox.jl.

# %%
#| label: fig-se-me-mc-solve
#| layout: [[33,34,33], [100]]
fig = Figure(size=(plot_figsize_width_pt, 0.5*plot_figsize_width_pt))

ax_se = Axis(fig[1, 1], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle", title="sesolve")
ax_me = Axis(fig[1, 2], xlabel="Time", title="mesolve")
ax_mc = Axis(fig[1, 3], xlabel="Time", title="mcsolve")
ax_td = Axis(fig[2, 1:3], xlabel="Time", ylabel=L"\langle \hat{a}^\dagger \hat{a} \rangle")

lines!(ax_se, tlist, real(res_se.expect[1, :]), color=:dodgerblue4)

lines!(ax_me, tlist, real(res_me.expect[1, :]), color=:dodgerblue4)
lines!(ax_me, tlist, exp.(-κ * tlist), color=:dimgray, linestyle=:dash, linewidth=1.25)
text!(ax_me, 40, 0.7, text=L"e^{-\kappa t}", color=:dimgray, fontsize=9)

lines!(ax_mc, tlist, real(res_mc.expect[1, :]), color=:dodgerblue4, label="100 Traj.")
lines!(ax_mc, tlist, real(res_mc.expect_all[1, :, 11]), color=:lightsteelblue, linewidth=1.25, linestyle=:dash, label="1 Traj.")

lines!(ax_td, tlist2, real(res_me_td.expect[1, :]), color=:dodgerblue4, label="Master Eq.")
hlines!(ax_td, [expect_ss_td_1], color=:crimson, linestyle=:dash, label=L"n_\mathrm{max} = 1")
hlines!(ax_td, [expect_ss_td_2], color=:darkorange, linestyle=:solid, label=L"n_\mathrm{max} = 2")

axislegend(ax_mc, position=:rt, padding=0)
axislegend(ax_td, position=:lb, padding=(10,0,0,0), orientation=:horizontal)

xlims!(ax_se, tlist[1], tlist[end])
xlims!(ax_me, tlist[1], tlist[end])
xlims!(ax_mc, tlist[1], tlist[end])
xlims!(ax_td, tlist2[1], tlist2[end])

ylims!(ax_se, 0, 1.3)
ylims!(ax_me, 0, 1.3)
ylims!(ax_mc, 0, 1.3)
ylims!(ax_td, 0, nothing)

text!(ax_se, 0.01, 0.99, text="(a)", space=:relative, align=(:left, :top), font=:bold)
text!(ax_me, 0.01, 0.99, text="(b)", space=:relative, align=(:left, :top), font=:bold)
text!(ax_mc, 0.01, 0.99, text="(c)", space=:relative, align=(:left, :top), font=:bold)
text!(ax_td, 0.01/3, 0.85, text="(d)", space=:relative, align=(:left, :top), font=:bold)

hideydecorations!(ax_me, ticks=false)
hideydecorations!(ax_mc, ticks=false)

linkyaxes!(ax_se, ax_me, ax_mc)

colgap!(fig.layout, 8)
rowgap!(fig.layout, 8)

# ------------ Inset Zoom Axis ------------

x_zoom = (420, 450)
y_zoom = (2.05, 2.25)

inset_ax = InsetAxis(fig, fig[2,1:3], ax_td, x_zoom, y_zoom)

lines!(inset_ax, tlist2, real(res_me_td.expect[1, :]), color=:dodgerblue4, label="Master Eq.", linewidth=2)
hlines!(inset_ax, [expect_ss_td_1], color=:crimson, linestyle=:dash, label=L"n_\mathrm{max} = 1", linewidth=2)
hlines!(inset_ax, [expect_ss_td_2], color=:darkorange, linestyle=:solid, label=L"n_\mathrm{max} = 2", linewidth=2)
hidedecorations!(inset_ax)
xlims!(inset_ax, x_zoom[1], x_zoom[2])
ylims!(inset_ax, y_zoom[1], y_zoom[2])

# -----------------------------------------

# For the LaTeX document
save("../figures/se-me-mc-solve.pdf", fig, pt_per_unit = 1.0)

fig
