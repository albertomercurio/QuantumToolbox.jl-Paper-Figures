using QuantumToolbox
using SciMLSensitivity
using Zygote
using Optimisers
using CairoMakie

include("_cairomakie_setup.jl")
include("interpolations.jl")

# %% --------- BELL STATE PREPARATION ---------
#
#
#

smooth_heaviside(t, t0, σ) = 1 / (1 + exp(-(t - t0) / σ))
coef_hadamard_smooth(p, t) = smooth_heaviside(t, 0, p[4]) * (1 - smooth_heaviside(t, p[1], p[4]))
coef_CNOT_smooth(p, t) = smooth_heaviside(t, p[2], p[4]) * (1 - smooth_heaviside(t, p[2] + p[3], p[4]))
coef_γ(p, t) = sqrt(p[5])

γ = 1e-4
const nsteps = 100
const tlist = range(0, 100, nsteps)
t_ratio = last(tlist) / π

σm1 = tensor(sigmam(), qeye(2))  # Lowering operator for qubit 1
σm2 = tensor(qeye(2), sigmam())  # Lowering operator for qubit 2
σx1 = tensor(sigmax(), qeye(2))  # Pauli X for qubit 1
σy1 = tensor(sigmay(), qeye(2))  # Pauli Y for qubit 1
σz1 = tensor(sigmaz(), qeye(2))  # Pauli Z for qubit 1
σx2 = tensor(qeye(2), sigmax())  # Pauli X for qubit 2
σy2 = tensor(qeye(2), sigmay())  # Pauli Y for qubit 2
σz2 = tensor(qeye(2), sigmaz())  # Pauli Z for qubit 2

# Initial state: |00>
const ψ0 = tensor(basis(2, 1), basis(2, 1))

# Target state
const Φp = ( tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)) ) / sqrt(2)

# %%
# ---- IDEAL CASE ----

# Hadamard gate Hamiltonian
H_hadamard = (σx1 - σz1) / (sqrt(2) * t_ratio)

# CNOT gate Hamiltonian
H_cnot = (1 + σz1) * (1 - σx2) / (2 * t_ratio)

# Time-dependent Hamiltonian
H_ideal = QobjEvo(H_hadamard, coef_hadamard_smooth) + QobjEvo(H_cnot, coef_CNOT_smooth)

c_ops = (sqrt(γ) * σm1, sqrt(γ) * σm2)

params_ideal = [t_ratio * π/2, t_ratio * π/2, t_ratio * π/2, 0.0001, γ]
sol = mesolve(H_ideal, ψ0, tlist, c_ops; params=params_ideal)

lines(tlist, coef_hadamard_smooth.(Ref(params_ideal), tlist), label="Hadamard")
lines(tlist, coef_CNOT_smooth.(Ref(params_ideal), tlist), label="CNOT")

1 - fidelity(Φp, sol.states[end])^2

# %%
# ---- OPTIMIZATION ----

# get_amp_idx(t) = clamp(searchsortedfirst(tlist, t), 1, nsteps)
function interp_p(p, t)
    # t_idx = get_amp_idx(t)
    # return p[t_idx]

    # return UniformQuadraticItp(tlist, p)(t)
    return UniformHermiteCR(tlist, p)(t)
    # return BSplineApprox(p, tlist, 3, 30, :Uniform, :Uniform)(t)
end

function coef_opt_hadamard(p, t)
    return interp_p(@view(p[1:nsteps]), t)
end
function coef_opt_CNOT(p, t)
    return interp_p(@view(p[1+nsteps:2nsteps]), t)
end

H = QobjEvo(H_hadamard, coef_opt_hadamard) + QobjEvo(H_cnot, coef_opt_CNOT)

const L = liouvillian(H, c_ops)

function loss(p)
    sol = mesolve(L, ψ0, tlist;
        params = p,
        progress_bar = Val(false),
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
    )

    return 1-fidelity(Φp, sol.states[end])^2
end

function optimize_parameters(params, params_ub, params_lb; maxiter=100)
    params_history = Array{Float64}(undef, length(params), maxiter)
    loss_history = Vector{Float64}(undef, maxiter)


    params0 = clamp.(params, params_lb, params_ub)
    state = Optimisers.setup(Optimisers.Adam(0.01), params0)

    params_history[:, 1] .= params0
    loss_history[1] = loss(params0)

    foreach(2:maxiter) do i
        grad = Zygote.gradient(loss, params)[1]
        state, params = Optimisers.update(state, params, grad)  # at every stepd

        params .= clamp.(params, params_lb, params_ub)

        loss_tmp = loss(params)
        println("Step $i: Loss = ", loss_tmp)
        flush(stdout)

        params_history[:, i] .= params
        loss_history[i] = loss_tmp
    end

    return params_history, loss_history
end

# %%
# for both Hadamard and CNOT parameters
# set the initial guess of starting and ending points at 0
params = [
    0.0; 
    fill(0.1, nsteps - 2) .+ 1e-3 .* randn(nsteps - 2);
    0.0;
    0.0; 
    fill(0.1, nsteps - 2) .+ 1e-3 .* randn(nsteps - 2);
    0.0;
]
params_lb = [
    -1e-3; 
    fill(-1.0, nsteps - 2);
    -1e-3;
    -1e-3; 
    fill(-1.0, nsteps - 2) .+ 1e-3 .* randn(nsteps - 2);
    -1e-3;
]
params_ub = [
    1e-3; 
    fill(1.0, nsteps - 2);
    1e-3;
    1e-3; 
    fill(1.0, nsteps - 2) .+ 1e-3 .* randn(nsteps - 2);
    1e-3;
]
loss(params)
Zygote.gradient(loss, params)[1]

params_history, loss_history = optimize_parameters(params, params_ub, params_lb; maxiter=200)

# %%

tlist_itp = range(tlist[1], tlist[end], 10*length(tlist))

coef_hadamard_ideal = coef_hadamard_smooth.(Ref(params_ideal), tlist_itp)
coef_CNOT_ideal = coef_CNOT_smooth.(Ref(params_ideal), tlist_itp)

coef_hadamard_opt= interp_p.( Ref(@view(params_history[1:nsteps, end])), tlist_itp)
coef_CNOT_opt= interp_p.( Ref(@view(params_history[1+nsteps:2nsteps, end])), tlist_itp)

# %%

fig = Figure(size=(plot_figsize_width_single_column_pt, 1.2 * plot_figsize_width_single_column_pt), figure_padding = (1,12,1,5))

ax_loss = Axis(fig[1, 1], xlabel = "Optimization Iteration", ylabel=L"1 - \mathcal{F}", yscale=log10)
ax_pulse_shape = Axis(fig[2, 1])
ax_pulse_shape_opt = Axis(fig[3, 1], xlabel = L"\gamma t")

lines!(ax_loss, axes(loss_history, 1), loss_history)

lines!(ax_pulse_shape, tlist_itp .* γ, coef_hadamard_ideal, label=L"f_\mathrm{H} (t)", color = Makie.wong_colors()[5])
lines!(ax_pulse_shape, tlist_itp .* γ, coef_CNOT_ideal, label=L"f_\mathrm{CNOT} (t)", linestyle=:dash, color = Makie.wong_colors()[6])

lines!(ax_pulse_shape_opt, tlist_itp .* γ, coef_hadamard_opt, label=L"f_\mathrm{H, opt} (t)", color=Makie.wong_colors()[5])
lines!(ax_pulse_shape_opt, tlist_itp .* γ, coef_CNOT_opt, label=L"f_\mathrm{CNOT, opt} (t)", linestyle=:dash, color=Makie.wong_colors()[6])

# ---- LEGENDS

axislegend(ax_pulse_shape, padding=(0, 2, 0, 5))
axislegend(ax_pulse_shape_opt, padding=(0, 2, 0, 5))

# ---- LIMITS

xlims!(ax_loss, 1, length(loss_history))
ylims!(ax_loss, nothing, 1)

dt = 0.5 # make the x limit slightly wider
xlims!(ax_pulse_shape, γ * (tlist[1] - dt), γ * (tlist[end] + dt))
xlims!(ax_pulse_shape_opt, γ * (tlist[1] - dt), γ * (tlist[end] + dt))

# ---- DECORATIONS

hidexdecorations!(ax_pulse_shape; ticks=false)

# ---- SPACING

rowgap!(fig.layout, 7)

# ---- LABELS

text!(ax_loss, 1.0, 0.0, text = "(a)", align = (:right, :bottom), space = :relative, offset=(-2, 5), font = :bold)
text!(ax_pulse_shape, 1.0, 0.0, text = "(b)", align = (:right, :bottom), space = :relative, offset=(-2, 5), font = :bold)
text!(ax_pulse_shape_opt, 1.0, 0.0, text = "(c)", align = (:right, :bottom), space = :relative, offset=(-2, 5), font = :bold)

# ---- SAVE

save(joinpath(@__DIR__, "../figures/autodiff_bell.pdf"), fig, pt_per_unit = 1.0)

fig

# %%

sol_test = mesolve(L, ψ0, tlist;
    params = params_history[:, end],
    progress_bar = Val(false),
    sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
)

1 - fidelity(Φp, sol_test.states[end])^2