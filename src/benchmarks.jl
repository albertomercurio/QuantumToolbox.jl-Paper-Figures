# %% [markdown]
# ---
# title: "Performance Comparison of Quantum Simulation Packages: Julia vs. Python"
# author: "Alberto Mercurio"

# engine: julia
# ---

# Here we compare the performance of [`QuantumToolbox.jl`](https://github.com/qutip/QuantumToolbox.jl) with other quantum simulation packages:
# - [`QuTiP`](https://github.com/qutip/qutip) (Python)
# - [`dynamiqs`](https://github.com/dynamiqs/dynamiqs) (Python - JAX)
# - [`QuantumOptics.jl`](https://github.com/qojulia/QuantumOptics.jl) (Julia)

# To allow reproducibility, this page is generated with [`Quarto`](https://quarto.org) based on [this repository](https://github.com/albertomercurio/QuantumToolbox-Benchmarks). Moreover, to keep the code clean, we use the [`PythonCall.jl`](https://github.com/JuliaPy/PythonCall.jl) package to call Python code from Julia. We tested that the overhead of calling Python code from Julia is negligible for the purpose of this benchmark.

# ## Importing the Required Packages

# %%
import QuantumToolbox
import QuantumOptics
using StochasticDiffEq
using CairoMakie
using BenchmarkTools
using JSON

include("_cairomakie_setup.jl")

# %% [markdown]
# ## Master Equation simulation

# Parameters:

# %%
N = 50 # Dimension of the Hilbert space
Δ = 0.1 # Detuning with respect to the drive
U = -0.05 # Nonlinearity
F = 2 # Amplitude of the drive
γ = 1 # Decay rate
nth = 0.2
ntraj = 100
stoc_dt = 1e-3
stoc_alg_quantumtoolbox = EM()
stoc_alg_quantumoptics = EM()

# %% [markdown]
# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumToolbox.fock(N, 0)

QuantumToolbox.mesolve(H, ψ0, tlist, c_ops, progress_bar=Val(false)).states[2] # Warm-up

mesolve_quantumtoolbox = @benchmark QuantumToolbox.mesolve($H, $ψ0, $tlist, $c_ops, progress_bar=Val(false)).states[2]


# %% [markdown]
# ### QuantumOptics.jl

# %%
bas = QuantumOptics.FockBasis(N)
a = QuantumOptics.destroy(bas)

H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumOptics.fockstate(bas, 0)

QuantumOptics.timeevolution.master(tlist, ψ0, H, c_ops)[2][2]

mesolve_quantumoptics = @benchmark QuantumOptics.timeevolution.master($tlist, $ψ0, $H, $c_ops, abstol=1e-8, reltol=1e-6)

# %% [markdown]
# ## Monte Carlo quantum trajectories simulation

# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumToolbox.fock(N, 0)

QuantumToolbox.mcsolve(H, ψ0, tlist, c_ops, progress_bar=Val(false), ntraj=ntraj).states[2] # Warm-up

mcsolve_quantumtoolbox =
    @benchmark QuantumToolbox.mcsolve($H, $ψ0, $tlist, $c_ops, progress_bar=Val(false), ntraj=ntraj).states[2]


# %% [markdown]
# ### QuantumOptics.jl

# %%
bas = QuantumOptics.FockBasis(N)
a = QuantumOptics.destroy(bas)

H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumOptics.fockstate(bas, 0)

function quantumoptics_mcwf(tlist, ψ0, H, c_ops, ntraj)
    Threads.@threads for i in 1:ntraj
        QuantumOptics.timeevolution.mcwf(tlist, ψ0, H, c_ops, display_beforeevent=true, display_afterevent=true, abstol=1e-8, reltol=1e-6)[2][2]
    end
end

quantumoptics_mcwf(tlist, ψ0, H, c_ops, ntraj) # Warm-up

mcsolve_quantumoptics = @benchmark quantumoptics_mcwf($tlist, $ψ0, $H, $c_ops, ntraj)

# %% [markdown]

# ## Stochastic Schrödinger Equation simulation

# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
sc_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = 0:stoc_dt*20:10 # We use this because dynamiqs only supports tsave to be multiple of dt
ψ0 = QuantumToolbox.fock(N, 0)

QuantumToolbox.ssesolve(H, ψ0, tlist, sc_ops, progress_bar=Val(false), ntraj=ntraj, alg=stoc_alg_quantumtoolbox, dt=stoc_dt).states[2] # Warm-up

ssesolve_quantumtoolbox =
    @benchmark QuantumToolbox.ssesolve($H, $ψ0, $tlist, $sc_ops, progress_bar=Val(false), ntraj=ntraj, alg=stoc_alg_quantumtoolbox, dt=stoc_dt).states[2]


# %% [markdown]
# ### QuantumOptics.jl

# %%
bas = QuantumOptics.FockBasis(N)
a = QuantumOptics.destroy(bas)

H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
sc_ops = [sqrt(γ * (1 + nth)) * a]

tlist = 0:stoc_dt*20:10
ψ0 = QuantumOptics.fockstate(bas, 0)

function quantumoptics_ssesolve(tlist, ψ0, H, sc_ops, ntraj, alg, dt)
    fdet_cm, fst_cm = QuantumOptics.stochastic.homodyne_carmichael(H, sc_ops[1], 0)
    Threads.@threads for i in 1:ntraj
        QuantumOptics.stochastic.schroedinger_dynamic(tlist, ψ0, fdet_cm, fst_cm; normalize_state=true, alg=alg, dt=dt, abstol=1e-2, reltol=1e-2)[2][2]
    end
end

quantumoptics_ssesolve(tlist, ψ0, H, sc_ops, ntraj, stoc_alg_quantumoptics, stoc_dt) # Warm-up

ssesolve_quantumoptics = @benchmark quantumoptics_ssesolve($tlist, $ψ0, $H, $sc_ops, ntraj, stoc_alg_quantumoptics, stoc_dt)

# && [markdown]
# ## Stochastic Master Equation

# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * nth) * a']
sc_ops = [sqrt(γ * (1 + nth)) * a]

tlist = 0:stoc_dt*20:10
ψ0 = QuantumToolbox.fock(N, 0)

QuantumToolbox.smesolve(H, ψ0, tlist, c_ops, sc_ops, ntraj=ntraj, progress_bar=Val(false), tstops=tlist, alg=stoc_alg_quantumtoolbox, dt=stoc_dt).states[2] # Warm-up

smesolve_quantumtoolbox = @benchmark QuantumToolbox.smesolve($H, $ψ0, $tlist, $c_ops, $sc_ops, ntraj=$ntraj, progress_bar=Val(false), tstops=$tlist, alg=stoc_alg_quantumtoolbox, dt=stoc_dt).states[2]

# %% [markdown]
# ### QuantumOptics.jl

# %%
bas = QuantumOptics.FockBasis(N)
a = QuantumOptics.destroy(bas)

H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']
sc_ops = [sqrt(γ * (1 + nth)) * a]

tlist = 0:stoc_dt*20:10
ψ0 = QuantumOptics.fockstate(bas, 0)

function quantumoptics_smesolve(tlist, ψ0, H, c_ops, sc_ops, ntraj, alg, dt)
    Threads.@threads for i in 1:ntraj
        QuantumOptics.stochastic.master(tlist, ψ0, H, c_ops, sc_ops; alg=alg, dt=dt)[2][2]
    end
end

quantumoptics_smesolve(tlist, ψ0, H, c_ops, sc_ops, ntraj, stoc_alg_quantumoptics, stoc_dt) # Warm-up

smesolve_quantumoptics = @benchmark quantumoptics_smesolve($tlist, $ψ0, $H, $c_ops, $sc_ops, $ntraj, $stoc_alg_quantumoptics, $stoc_dt)

# %% [markdown]
# ## Plotting the Results

# %%
qutip_results = JSON.parsefile("python/qutip_benchmark_results.json")
dynamiqs_results = JSON.parsefile("python/dynamiqs_benchmark_results.json")

mesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mesolve"]),)
mcsolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mcsolve"]),)
ssesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_ssesolve"]),)
smesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_smesolve"]),)

mesolve_dynamiqs = (times=Vector{Float64}(dynamiqs_results["dynamiqs_mesolve"]),)
smesolve_dynamiqs = (times=Vector{Float64}(dynamiqs_results["dynamiqs_smesolve"]),)

mesolve_times = [
    1e-6 * sum(m.times) / length(m.times) for
    m in [mesolve_quantumtoolbox, mesolve_quantumoptics, mesolve_qutip, mesolve_dynamiqs]
]
mcsolve_times =
    [1e-6 * sum(m.times) / length(m.times) for m in [mcsolve_quantumtoolbox, mcsolve_quantumoptics, mcsolve_qutip]]
ssesolve_times = [
    1e-6 * sum(m.times) / length(m.times) for
    m in [ssesolve_quantumtoolbox, ssesolve_quantumoptics, ssesolve_qutip]
]
smesolve_times = [
    1e-6 * sum(m.times) / length(m.times) for
    m in [smesolve_quantumtoolbox, smesolve_quantumoptics, smesolve_qutip, smesolve_dynamiqs]
]

mesolve_times_x = [1,2,3,4]
mcsolve_times_x = [1,2,3]
ssesolve_times_x = [1,2,3]
smesolve_times_x = [1,2,3,4]

labels = ["QuantumToolbox.jl", "QuantumOptics.jl", "QuTiP", "dynamiqs"]

# %%

fig = Figure(size=(plot_figsize_width_pt, plot_figsize_width_pt*0.4))

Label(fig[1, 1:3], "Performance Comparison with Other Packages (Lower is better)", tellwidth=false, halign=:center)

ax_mesolve = Axis(
    fig[2, 1],
    ylabel="Time (ms)",
    title="mesolve",
    xticks = (mesolve_times_x, labels[mesolve_times_x]),
    xticklabelrotation = π/4,
)
ax_mcsolve = Axis(
    fig[2, 2],
    title="mcsolve",
    xticks = (mcsolve_times_x, labels[mcsolve_times_x]),
    xticklabelrotation = π/4,
)
ax_smesolve = Axis(
    fig[2, 3],
    title="smesolve",
    xticks = (smesolve_times_x, labels[smesolve_times_x]),
    xticklabelrotation = π/4,
)

colors = Makie.wong_colors()

barplot!(
    ax_mesolve,
    mesolve_times_x,
    mesolve_times,
    # dodge = 1:length(mesolve_times),
    color=colors[mesolve_times_x]
)

barplot!(ax_mcsolve,
    mcsolve_times_x,
    mcsolve_times,
    # dodge=1:length(mcsolve_times), 
    color=colors[mcsolve_times_x]
)

barplot!(ax_smesolve,
    smesolve_times_x,
    smesolve_times,
    # dodge=1:length(smesolve_times), 
    color=colors[smesolve_times_x]
)

ylims!(ax_mesolve, 0, nothing)
ylims!(ax_mcsolve, 0, nothing)
ylims!(ax_smesolve, 0, nothing)

rowgap!(fig.layout, 2)

# For the LaTeX document
# save("figures/benchmarks.pdf", fig, pt_per_unit = 1.0)

# For the README file in the GitHub repository
# save("figures/benchmarks.svg", fig, pt_per_unit = 2.0)

fig

# %% [markdown]
# ## System Information

# %%
versioninfo()

# %% [markdown]
# ---

# %%
QuantumToolbox.about()

# %% [markdown]
# ---

# %%
qutip.about()
