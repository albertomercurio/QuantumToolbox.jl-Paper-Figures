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

# Varying the Hilbert space dimension N
qutip_results_N = JSON.parsefile("python/qutip_benchmark_results_N.json")
dynamiqs_results_N = JSON.parsefile("python/dynamiqs_benchmark_results_N.json")
quantumtoolbox_results_N = JSON.parsefile("julia/quantumtoolbox_benchmark_results_N.json")

mesolve_qutip_N_cpu = (times=convert(Vector{Vector{Float64}}, qutip_results_N["qutip_mesolve_N_cpu"]),)
mesolve_qutip_N_gpu = (times=convert(Vector{Vector{Float64}}, qutip_results_N["qutip_mesolve_N_gpu"]),)

mesolve_dynamiqs_N_cpu = (times=convert(Vector{Vector{Float64}}, dynamiqs_results_N["dynamiqs_mesolve_N_cpu"]),)
mesolve_dynamiqs_N_gpu = (times=convert(Vector{Vector{Float64}}, dynamiqs_results_N["dynamiqs_mesolve_N_gpu"]),)

mesolve_quantumtoolbox_N_cpu = (times=convert(Vector{Vector{Float64}}, quantumtoolbox_results_N["quantumtoolbox_mesolve_N_cpu"]),)
mesolve_quantumtoolbox_N_gpu = (times=convert(Vector{Vector{Float64}}, quantumtoolbox_results_N["quantumtoolbox_mesolve_N_gpu"]),)

N_list = floor.(Int, range(10, 400, 25))

mesolve_times_N_cpu = [
    [1e-9 * sum(mm) / length(mm) for mm in m.times] for
    m in [mesolve_quantumtoolbox_N_cpu, mesolve_qutip_N_cpu, mesolve_dynamiqs_N_cpu]
]
mesolve_times_N_gpu = [
    [1e-9 * sum(mm) / length(mm) for mm in m.times] for
    m in [mesolve_quantumtoolbox_N_gpu, mesolve_qutip_N_gpu, mesolve_dynamiqs_N_gpu]
]

mesolve_times_x = [1,2,3,4]
mcsolve_times_x = [1,2,3]
ssesolve_times_x = [1,2,3]
smesolve_times_x = [1,2,3,4]

labels = ["QuantumToolbox.jl", "QuantumOptics.jl", "QuTiP", "dynamiqs"]

# %% [markdown]

# And now we plot the results.

# %%

fig = Figure(size=(plot_figsize_width_pt, plot_figsize_width_pt*0.6))

grid_plots = fig[2, 1]
grid_me_mc_sme = grid_plots[1, 1] = GridLayout()
grid_me_vs_N = grid_plots[2, 1] = GridLayout()

ax_mesolve = Axis(
    grid_me_mc_sme[1, 1],
    ylabel=L"Time ($\mathrm{s}$)",
    # title="mesolve",
)
ax_mcsolve = Axis(
    grid_me_mc_sme[1, 2],
    # title="mcsolve",
)
ax_smesolve = Axis(
    grid_me_mc_sme[1, 3],
    # title="smesolve",
)
ax_mesolve_vs_N_cpu = Axis(
    grid_me_vs_N[1, 1],
    ylabel="Time (s)",
    # xscale = log10,
    yscale = log10,
    xlabel = "Hilbert space dimension",
)
ax_mesolve_vs_N_gpu = Axis(
    grid_me_vs_N[1, 2],
    # xscale = log10,
    yscale = log10,
    xlabel = "Hilbert space dimension",
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

elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
Legend(fig[1, 1], elements, labels, orientation=:horizontal)

ylims!(ax_mesolve, 0, nothing)
ylims!(ax_mcsolve, 0, nothing)
ylims!(ax_smesolve, 0, nothing)

hidexdecorations!(ax_mesolve)
hidexdecorations!(ax_mcsolve)
hidexdecorations!(ax_smesolve)

# Hilbert space dimension N plots

markers = [:rect, :diamond, :pentagon]
for (i, m) in enumerate(mesolve_times_N_cpu)
    scatterlines!(ax_mesolve_vs_N_cpu, N_list, m, color=colors[[1,3,4]][i], marker=markers[i])
end
for (i, m) in enumerate(mesolve_times_N_gpu)
    scatterlines!(ax_mesolve_vs_N_gpu, N_list[1:length(m)], m, color=colors[[1,3,4]][i], marker=markers[i])
end

# Labels
text!(ax_mesolve, 0.0, 1.0, text = "(a)", align = (:left, :top), space = :relative, offset=(3, -3))
text!(ax_mcsolve, 0.0, 1.0, text = "(b)", align = (:left, :top), space = :relative, offset=(3, -3))
text!(ax_smesolve, 0.0, 1.0, text = "(c)", align = (:left, :top), space = :relative, offset=(3, -3))
text!(ax_mesolve_vs_N_cpu, 0.0, 1.0, text = "(d)", align = (:left, :top), space = :relative, offset=(3, -3))
text!(ax_mesolve_vs_N_gpu, 0.0, 1.0, text = "(e)", align = (:left, :top), space = :relative, offset=(3, -3))

text!(ax_mesolve, 0.5, 1.0, text = "mesolve", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)
text!(ax_mcsolve, 0.5, 1.0, text = "mcsolve", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)
text!(ax_smesolve, 0.5, 1.0, text = "smesolve", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)
text!(ax_mesolve_vs_N_cpu, 0.5, 1.0, text = "mesolve (CPU)", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)
text!(ax_mesolve_vs_N_gpu, 0.5, 1.0, text = "mesolve (GPU)", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)


# For the LaTeX document
# save(joinpath(@__DIR__, "../../figures/benchmarks.pdf"), fig, pt_per_unit = 1.0)

# For the README file in the GitHub repository
# Label(fig[0, 1], "Performance Comparison with Other Packages (Lower is better)", tellwidth=false, halign=:center, fontsize=9)
# save(joinpath(@__DIR__, "../../figures/benchmarks.svg"), fig, pt_per_unit = 2.0)

rowgap!(fig.layout, 5)
colgap!(grid_me_mc_sme, 7)
colgap!(grid_me_vs_N, 5)

fig

# %% [markdown]
# ## System Information

# %%
versioninfo()

# %% [markdown]
# ---

# %%
QuantumToolbox.about()

