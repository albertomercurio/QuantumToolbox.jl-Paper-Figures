using QuantumToolbox
using CairoMakie

include("_cairomakie_setup.jl")

# %%

# parameters
N = 20      # Fock space dimension
ωc = 5      # cavity resonance frequency
ωq = 5      # qubit resonance frequency
g = 0.1     # coupling strength
κ = 1       # cavity decay rate
γ = 2       # qubit decay rate
α = 2       # coherence of initial state
ntraj = 500 # number of trajectories

# %%

tlist = range(0, 10π / ωc, 400)

# operators
a = tensor(destroy(N), qeye(2))
σm = tensor(qeye(N), sigmam())
σz = tensor(qeye(N), sigmaz())

H = ωc * a' * a + ωq * σz / 2 + g * (a' * σm + a * σm')

# initial state
ψ0 = tensor(coherent(N, α), basis(2, 0)) # |α, e⟩
# %%

# stochastic collapse operators
sc_ops = [√(κ) * a]

X = (a + a') * √(κ)

sse_sol = ssesolve(
    H,
    ψ0,
    tlist,
    sc_ops[1],
    e_ops = [X],
    ntraj = ntraj,
    store_measurement = Val(true),
)

measurement_sse_avg = sum(sse_sol.measurement, dims=2) / size(sse_sol.measurement, 2)
measurement_sse_avg = dropdims(measurement_sse_avg, dims=2)

# %%

# Including qubit losses
c_ops  = [√(γ) * σm]

sme_sol = smesolve(
    H,
    ψ0,
    tlist,
    c_ops,
    sc_ops[1],
    e_ops = [X],
    ntraj = ntraj,
    store_measurement = Val(true)
)

measurement_sme_avg = sum(sme_sol.measurement, dims=2) / size(sme_sol.measurement, 2)
measurement_sme_avg = dropdims(measurement_sme_avg, dims=2)

# %%

fig = Figure(size = (plot_figsize_width_single_column_pt, 1.2 * plot_figsize_width_single_column_pt))

ax_sse = Axis(fig[1, 1], ylabel="Measurement", yticks = [-5, 0, 5])
ax_sme = Axis(fig[2, 1], ylabel="Measurement", xlabel = "Time", yticks = [-5, 0, 5])

lines!(ax_sse, tlist[1:end-1], real(measurement_sse_avg[1,:]), label = L"J_x", color = :lightsteelblue)
lines!(ax_sse, tlist, real(sse_sol.expect[1,:]),  label = L"\langle \hat{X} \rangle", color = :dodgerblue4)

lines!(ax_sme, tlist[1:end-1], real(measurement_sme_avg[1,:]), label = L"J_x", color = :lightsteelblue)
lines!(ax_sme, tlist, real(sme_sol.expect[1,:]),  label = L"\langle \hat{a} + \hat{a}^\dagger \rangle", color = :dodgerblue4)
axislegend(ax_sse, position = :rt, padding = 0)

# -- LIMITS --

xlims!(ax_sse, tlist[1], tlist[end])
xlims!(ax_sme, tlist[1], tlist[end])
ylims!(ax_sse, -6, 6)
ylims!(ax_sme, -6, 6)

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
