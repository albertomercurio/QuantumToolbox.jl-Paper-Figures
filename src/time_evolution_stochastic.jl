using QuantumToolbox
using StochasticDiffEq
using CairoMakie

# %%

N = 50 # Dimension of the Hilbert space
Δ = 0.1 # Detuning with respect to the drive
U = -0.05 # Nonlinearity
F = 2 # Amplitude of the drive
γ = 1 # Decay rate
nth = 0.2
ntraj = 100
stoc_dt = 1e-3

# %%

a = destroy(N)
H = Δ * a' * a - U/2 * a'^2 * a^2 + F * (a + a')
sc_ops = [sqrt(γ * (1 + nth)) * a]

tlist = 0:stoc_dt*20:10 # We use this because dynamiqs only supports tsave to be multiple of dt
ψ0 = fock(N, 0)

prob_sse = ssesolveProblem(H, ψ0, tlist, sc_ops, e_ops=[a'*a], saveat=tlist, progress_bar=Val(false), store_measurement=Val(true)) # Warm-up

integrator = init(prob_sse.prob, SRA2())


function dWdt(integrator)
    @inbounds _dWdt = (integrator.W.u[end] .- integrator.W.u[end-1]) ./ (integrator.W.t[end] - integrator.W.t[end-1])

    return _dWdt
end

sol_sse = solve(prob_sse.prob, SRA2()) 

# %%

import QuantumToolbox: _get_m_expvals, SaveFuncSSESolve

_get_m_expvals(sol_sse, SaveFuncSSESolve)

ssesolve(prob_sse, SRA2())


sol_sse = ssesolve(H, ψ0, tlist, sc_ops, e_ops=[a'*a], saveat=tlist, ntraj=500, progress_bar=Val(false), store_measurement=Val(true), alg=SRA2())

# %%

fig = Figure()
ax = Axis(fig[1, 1])

measurement_avg = sum(sol_sse.measurement, dims=2) / size(sol_sse.measurement, 2)
measurement_avg = dropdims(measurement_avg, dims=2)

lines!(ax, tlist[1:end-1], measurement_avg[1, :])

fig

# %%











# %%

# parameters
DIM = 20               # Hilbert space dimension
DELTA = 5 * 2 * pi  # cavity detuning
KAPPA = 2              # cavity decay rate
INTENSITY = 4          # intensity of initial state
NUMBER_OF_TRAJECTORIES = 500

# operators
a = destroy(DIM)
x = a + a'
H = DELTA * a' * a

rho_0 = coherent(DIM,sqrt(INTENSITY))
times = 0:0.00025:1

stoc_solution = smesolve(
    H, rho_0, times,
    nothing,
    [sqrt(KAPPA) * a],
    e_ops=[x],
    ntraj=NUMBER_OF_TRAJECTORIES,
    store_measurement=Val(true),
)

# %%

measurement_avg = sum(stoc_solution.measurement, dims=2) / size(stoc_solution.measurement, 2)
measurement_avg = dropdims(measurement_avg, dims=2)

# %%

fig = Figure()
ax = Axis(fig[1, 1])

lines!(ax, times[1:end-1], measurement_avg[1, :])
lines!(ax, times, real.(stoc_solution.expect[1, :]))

fig

# %%

prob = ssesolveEnsembleProblem(H, rho_0, times, [sqrt(KAPPA) * a], e_ops=[x], ntraj=NUMBER_OF_TRAJECTORIES, store_measurement=Val(true))

solve(prob.prob, SRA2(), EnsembleThreads(), trajectories=2)

# %%

using QuantumToolbox

N = 20         # Fock space dimension
Δ = 5 * 2 * π  # cavity detuning
κ = 2          # cavity decay rate
α = 4          # intensity of initial state
ntraj = 500    # number of trajectories

tlist = 0:0.0025:1

# operators
a = destroy(N)
x = a + a'
H = Δ * a' * a

# initial state
ψ0 = coherent(N, √α)

# temperature with average of 0 excitations (absolute zero)
n_th = 0
# c_ops  = [√(κ * n_th) * a'] -> nothing
sc_ops = [√(κ * (n_th + 1)) * a]

sse_sol = ssesolve(
    H,
    ψ0,
    tlist,
    sc_ops,
    e_ops = [x],
    ntraj = ntraj,
    store_measurement = Val(true),
)

measurement_avg = sum(sse_sol.measurement, dims=2) / size(sse_sol.measurement, 2)
measurement_avg = dropdims(measurement_avg, dims=2)

# %%

using CairoMakie

# %%

# plot by CairoMakie.jl
fig = Figure(size = (500, 350))
ax = Axis(fig[1, 1], xlabel = "Time")
lines!(ax, tlist[1:end-1], real(measurement_avg[1,:]), label = L"J_x", color = :red, linestyle = :solid)
lines!(ax, tlist, real(sse_sol.expect[1,:]),  label = L"\langle x \rangle", color = :black, linestyle = :solid)

axislegend(ax, position = :rt)

fig

# %%
