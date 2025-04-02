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
using CairoMakie
using BenchmarkTools
using JSON

include("../_cairomakie_setup.jl")

# %% [markdown]
# ## Plotting the Results

# %%
qutip_results = JSON.parsefile("python/qutip_benchmark_results.json")
dynamiqs_results = JSON.parsefile("python/dynamiqs_benchmark_results.json")
quantumoptics_results = JSON.parsefile("julia/quantumoptics_benchmark_results.json")
quantumtoolbox_results = JSON.parsefile("julia/quantumtoolbox_benchmark_results.json")

mesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mesolve"]),)
mcsolve_qutip = (times=Vector{Float64}(qutip_results["qutip_mcsolve"]),)
# ssesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_ssesolve"]),)
smesolve_qutip = (times=Vector{Float64}(qutip_results["qutip_smesolve"]),)

mesolve_dynamiqs = (times=Vector{Float64}(dynamiqs_results["dynamiqs_mesolve"]),)
smesolve_dynamiqs = (times=Vector{Float64}(dynamiqs_results["dynamiqs_smesolve"]),)

mesolve_quantumoptics = (times=Vector{Float64}(quantumoptics_results["quantumoptics_mesolve"]),)
mcsolve_quantumoptics = (times=Vector{Float64}(quantumoptics_results["quantumoptics_mcsolve"]),)
smesolve_quantumoptics = (times=Vector{Float64}(quantumoptics_results["quantumoptics_smesolve"]),)

mesolve_quantumtoolbox = (times=Vector{Float64}(quantumtoolbox_results["quantumtoolbox_mesolve"]),)
mcsolve_quantumtoolbox = (times=Vector{Float64}(quantumtoolbox_results["quantumtoolbox_mcsolve"]),)
smesolve_quantumtoolbox = (times=Vector{Float64}(quantumtoolbox_results["quantumtoolbox_smesolve"]),)

mesolve_times = [
    1e-9 * sum(m.times) / length(m.times) for
    m in [mesolve_quantumtoolbox, mesolve_quantumoptics, mesolve_qutip, mesolve_dynamiqs]
]
mcsolve_times =
    [1e-9 * sum(m.times) / length(m.times) for m in [mcsolve_quantumtoolbox, mcsolve_quantumoptics, mcsolve_qutip]]
# ssesolve_times = [
#     1e-9 * sum(m.times) / length(m.times) for
#     m in [ssesolve_quantumtoolbox, ssesolve_quantumoptics, ssesolve_qutip]
# ]
smesolve_times = [
    1e-9 * sum(m.times) / length(m.times) for
    m in [smesolve_quantumtoolbox, smesolve_quantumoptics, smesolve_qutip, smesolve_dynamiqs]
]

mesolve_times_x = [1,2,3,4]
mcsolve_times_x = [1,2,3]
ssesolve_times_x = [1,2,3]
smesolve_times_x = [1,2,3,4]

labels = ["QuantumToolbox.jl", "QuantumOptics.jl", "QuTiP", "dynamiqs"]

# %% [markdown]

# And now we plot the results.

# %%

fig = Figure(size=(plot_figsize_width_pt, plot_figsize_width_pt*0.4))

ax_mesolve = Axis(
    fig[1, 1],
    ylabel=L"Time ($s$)",
    title="mesolve",
    # xticks = (mesolve_times_x, labels[mesolve_times_x]),
    # xticklabelrotation = π/4,
)
ax_mcsolve = Axis(
    fig[1, 2],
    title="mcsolve",
    # xticks = (mcsolve_times_x, labels[mcsolve_times_x]),
    # xticklabelrotation = π/4,
)
ax_smesolve = Axis(
    fig[1, 3],
    title="smesolve",
    # xticks = (smesolve_times_x, labels[smesolve_times_x]),
    # xticklabelrotation = π/4,
)

colors = Makie.wong_colors()

barplot!(
    ax_mesolve,
    mesolve_times_x,
    mesolve_times,
    # dodge = 1:length(mesolve_times),
    color=colors[mesolve_times_x]
)
text!(ax_mesolve, 0.02, 0.98, text = "(a)", font = :bold, align = (:left, :top), space = :relative)

barplot!(ax_mcsolve,
    mcsolve_times_x,
    mcsolve_times,
    # dodge=1:length(mcsolve_times), 
    color=colors[mcsolve_times_x]
)
text!(ax_mcsolve, 0.02, 0.98, text = "(b)", font = :bold, align = (:left, :top), space = :relative)

barplot!(ax_smesolve,
    smesolve_times_x,
    smesolve_times,
    # dodge=1:length(smesolve_times), 
    color=colors[smesolve_times_x]
)
text!(ax_smesolve, 0.02, 0.98, text = "(c)", font = :bold, align = (:left, :top), space = :relative)

elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
Legend(fig[2, 1:3], elements, labels, orientation=:horizontal)

ylims!(ax_mesolve, 0, nothing)
ylims!(ax_mcsolve, 0, nothing)
ylims!(ax_smesolve, 0, nothing)

hidexdecorations!(ax_mesolve)
hidexdecorations!(ax_mcsolve)
hidexdecorations!(ax_smesolve)


# For the LaTeX document
# save("../figures/benchmarks.pdf", fig, pt_per_unit = 1.0)

# # For the README file in the GitHub repository
Label(fig[0, 1:3], "Performance Comparison with Other Packages (Lower is better)", tellwidth=false, halign=:center, fontsize=9)
save(joinpath(@__DIR__, "../../figures/benchmarks.svg"), fig, pt_per_unit = 2.0)

rowgap!(fig.layout, 5)

fig

# %% [markdown]
# ## System Information

# %%
versioninfo()

# %% [markdown]
# ---

# %%
QuantumToolbox.about()

