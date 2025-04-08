using QuantumToolbox
# using StochasticDiffEq
using CairoMakie

include("_cairomakie_setup.jl")

# %%

# parameters
N = 20         # Fock space dimension
ω = 2π * 5     # cavity resonance frequency
κ = 2          # cavity decay rate
n = 4          # intensity of initial state
ntraj = 1000    # number of trajectories

# %%

tlist = range(0, 1.5, 400)

# operators
a = destroy(N)
H = ω * a' * a

# initial state
ψ0 = coherent(N, √n)
# %%

# collapse operators
sc_ops = [√(κ) * a]

x = (a + a') * √(κ)

sse_sol = ssesolve(
    H,
    ψ0,
    tlist,
    sc_ops[1],
    e_ops = [x],
    ntraj = ntraj,
    store_measurement = Val(true),
    # abstol=1e-3,
)

measurement_sse_avg = sum(sse_sol.measurement, dims=2) / size(sse_sol.measurement, 2)
measurement_sse_avg = dropdims(measurement_sse_avg, dims=2)

# %%

# temperature with average of 1 excitations
n_th = 1
c_ops  = [√(κ * n_th) * a']
sc_ops = [√(κ * (n_th + 1)) * a]

x_sme = (a + a') * √(κ * (n_th + 1))

sme_sol = smesolve(
    H,
    ψ0,
    tlist,
    c_ops,
    sc_ops[1],
    e_ops = [x_sme],
    ntraj = ntraj,
    store_measurement = Val(true)
)

measurement_sme_avg = sum(sme_sol.measurement, dims=2) / size(sme_sol.measurement, 2)
measurement_sme_avg = dropdims(measurement_sme_avg, dims=2)

# %%

fig = Figure(size = (plot_figsize_width_single_column_pt, 1.2 * plot_figsize_width_single_column_pt))

ax_sse = Axis(fig[1, 1], ylabel="Measurement")
ax_sme = Axis(fig[2, 1], ylabel="Measurement", xlabel = "Time")

lines!(ax_sse, tlist[1:end-1], real(measurement_sse_avg[1,:]), label = L"J_x", color = :lightsteelblue)
lines!(ax_sse, tlist, real(sse_sol.expect[1,:]),  label = L"\langle \hat{X} \rangle", color = :dodgerblue4)

lines!(ax_sme, tlist[1:end-1], real(measurement_sme_avg[1,:]), label = L"J_x", color = :lightsteelblue)
lines!(ax_sme, tlist, real(sme_sol.expect[1,:]),  label = L"\langle \hat{a} + \hat{a}^\dagger \rangle", color = :dodgerblue4)
axislegend(ax_sse, position = :rt, padding = 0)

# -- LIMITS --

xlims!(ax_sse, tlist[1], tlist[end])
xlims!(ax_sme, tlist[1], tlist[end])
ylims!(ax_sse, -11, 11)
ylims!(ax_sme, -11, 11)

# -- DECORATION --
hidexdecorations!(ax_sse, ticks=false)

# -- TEXT --
text!(ax_sse, 0.0, 1.0, text="(a)", space=:relative, align=(:left, :top), font=:bold, offset=(2, -1))
text!(ax_sme, 0.0, 1.0, text="(b)", space=:relative, align=(:left, :top), font=:bold, offset=(2, -1))

text!(ax_sse, 0.5, 1.0, text = "ssesolve", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)
text!(ax_sme, 0.5, 1.0, text = "smesolve", align = (:right, :top), space = :relative, offset=(-10, 0), font=:bold)

# -- SPACING --
rowgap!(fig.layout, 10)

# -- SAVE FIGURE --
save(joinpath(@__DIR__, "../figures/sse-sme.pdf"), fig, pt_per_unit = 1.0)

fig
