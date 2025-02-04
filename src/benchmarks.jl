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

include("_cairomakie_setup.jl")

# %% [markdown]
# ## Master Equation simulation

# Parameters:

# %%
N = 50
Δ = 0.1
F = 2
γ = 1
nth = 0.8

# %% [markdown]
# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a + F * (a + a')
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

H = Δ * a' * a + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumOptics.fockstate(bas, 0)

QuantumOptics.timeevolution.master(tlist, ψ0, H, c_ops)[2][2]

mesolve_quantumoptics = @benchmark QuantumOptics.timeevolution.master($tlist, $ψ0, $H, $c_ops)

# %% [markdown]
# ## Monte Carlo quantum trajectories simulation

# Parameters:

# %%
N = 50
Δ = 0.1
F = 2
γ = 1
nth = 0.8
ntraj = 100

# %% [markdown]
# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a + F * (a + a')
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

H = Δ * a' * a + F * (a + a')
c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumOptics.fockstate(bas, 0)

function quantumoptics_mcwf(tlist, ψ0, H, c_ops, ntraj)
    Threads.@threads for i in 1:ntraj
        QuantumOptics.timeevolution.mcwf(tlist, ψ0, H, c_ops, display_beforeevent=true, display_afterevent=true)[2][2]
    end
end

quantumoptics_mcwf(tlist, ψ0, H, c_ops, ntraj) # Warm-up

mcsolve_quantumoptics = @benchmark quantumoptics_mcwf($tlist, $ψ0, $H, $c_ops, ntraj)

# %% [markdown]

# ## Stochastic Schrödinger Equation simulation

# Parameters:

# %%
sse_dt = 1e-3

# %% [markdown]

# ### QuantumToolbox.jl

# %%
a = QuantumToolbox.destroy(N)
H = Δ * a' * a + F * (a + a')
sc_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumToolbox.fock(N, 0)

QuantumToolbox.ssesolve(H, ψ0, tlist, sc_ops, progress_bar=Val(false), ntraj=ntraj, alg=EM(), dt=sse_dt).states[2] # Warm-up

ssesolve_quantumtoolbox =
    @benchmark QuantumToolbox.ssesolve($H, $ψ0, $tlist, $sc_ops, progress_bar=Val(false), ntraj=ntraj, alg=EM(), dt=sse_dt).states[2]


# %% [markdown]
# ### QuantumOptics.jl

# %%
bas = QuantumOptics.FockBasis(N)
a = QuantumOptics.destroy(bas)

H = Δ * a' * a + F * (a + a')
sc_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

tlist = range(0, 10, 100)
ψ0 = QuantumOptics.fockstate(bas, 0)

function quantumoptics_ssesolve(tlist, ψ0, H, sc_ops, ntraj, dt)
    fdet_cm, fst_cm = QuantumOptics.stochastic.homodyne_carmichael(H, sc_ops[1], 0)
    Threads.@threads for i in 1:ntraj
        QuantumOptics.stochastic.schroedinger_dynamic(tlist, ψ0, fdet_cm, fst_cm; normalize_state=true, alg=EM(), dt=dt)[2][2]
    end
end

quantumoptics_ssesolve(tlist, ψ0, H, sc_ops, ntraj, sse_dt) # Warm-up

ssesolve_quantumoptics = @benchmark quantumoptics_ssesolve($tlist, $ψ0, $H, $sc_ops, ntraj, sse_dt)

# %% [markdown]
# ## Plotting the Results

# %%
qutip_results = JSON.parsefile("src/python/qutip_benchmark_results.json")
dynamiqs_results = JSON.parsefile("src/python/dynamiqs_benchmark_results.json")

mesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mesolve"]),)
mcsolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mcsolve"]),)
ssesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_ssesolve"]),)

mesolve_dynamiqs = (times=Vector{Float64}(dynamiqs_results["dynamiqs_mesolve"]),)

mesolve_times = [
    1e-6 * sum(m.times) / length(m.times) for
    m in [mesolve_quantumtoolbox, mesolve_qutip, mesolve_dynamiqs, mesolve_quantumoptics]
]
mcsolve_times =
    [1e-6 * sum(m.times) / length(m.times) for m in [mcsolve_quantumtoolbox, mcsolve_qutip, mcsolve_quantumoptics]]
ssesolve_times = [
    1e-6 * sum(m.times) / length(m.times) for
    m in [ssesolve_quantumtoolbox, ssesolve_qutip, ssesolve_quantumoptics]
]

mesolve_times_x = [1,2,3,4]
mcsolve_times_x = [1,2,4]
ssesolve_times_x = [1,2,4]

labels = ["QuantumToolbox.jl", "QuTiP", "dynamiqs", "QuantumOptics.jl"]

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
ax_ssesolve = Axis(
    fig[2, 3],
    title="ssesolve",
    xticks = (ssesolve_times_x, labels[ssesolve_times_x]),
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

barplot!(ax_ssesolve,
    ssesolve_times_x,
    ssesolve_times,
    # dodge=1:length(ssesolve_times), 
    color=colors[ssesolve_times_x]
)

ylims!(ax_mesolve, 0, nothing)
ylims!(ax_mcsolve, 0, nothing)
ylims!(ax_ssesolve, 0, nothing)

rowgap!(fig.layout, 2)

# save("figures/benchmarks.pdf", fig, pt_per_unit = 1.0)

save("figures/benchmarks.svg", fig, pt_per_unit = 1.0)

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
