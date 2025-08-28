using MKL
using QuantumToolbox
using SciMLSensitivity
using Zygote
using Optimisers
using DataInterpolations
using CairoMakie

include("_cairomakie_setup.jl")

# %%

# const N  = 20
# const a  = destroy(N)
# const ωc = 5.0
# const γ  = 1.0

# coef_a(p, t) = p[2] * exp(1im * p[1] * t)

# H = ωc * a' * a + QobjEvo(a, coef_a) + QobjEvo(a', conj ∘ coef_a)
# c_ops = [sqrt(γ) * a]
# const L = liouvillian(H, c_ops)
# const ψ0 = fock(N, 0)
# const tlist = range(0, 40, 100)

# function my_f_mesolve(p)
#     sol = mesolve(
#         L,
#         ψ0,
#         tlist,
#         progress_bar = Val(false),
#         params = p,
#         sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
#     )

#     return real(expect(a' * a, sol.states[end]))
# end

# # Analytical solution
# n_ss(ωd, F) = abs2(F / ((ωc - ωd) + 1im * γ / 2))
# n_ss_deriv(ωd, F) = [
#     2 * F^2 * (ωc - ωd) / ((ωc - ωd)^2 + γ^2 / 4)^2,
#     2 * F / ((ωc - ωd)^2 + γ^2 / 4),
# ]

# ωd = 1.0
# F = 1.0
# params = [ωd, F]

# grad_qt = Zygote.gradient(my_f_mesolve, params)[1]

# grad_exact = n_ss_deriv(ωd, F)

# # %%

# ωd_list = ωc .- range(-1, 1, 200)

# pr = ProgressBar(length(ωd_list))
# deriv_analytical_ωd = map(ωd_list) do ωd
#     res = n_ss_deriv(ωd, F)[1]
#     next!(pr)
#     return res
# end

# pr = ProgressBar(length(ωd_list))
# deriv_numerical_ωd = map(ωd_list) do ωd
#     res = Zygote.gradient(my_f_mesolve, [ωd, F])[1][1]
#     next!(pr)
#     return res
# end

# # %%

# fig = Figure(size=(plot_figsize_width_single_column_pt, 0.6 * plot_figsize_width_single_column_pt))
# ax = Axis(fig[1, 1], xlabel = L"\omega_c - \omega_d", ylabel = L"\partial \langle \hat{a}^\dagger \hat{a} \rangle_\mathrm{ss} / \partial \omega_d")

# lines!(ax, ωc .- ωd_list, deriv_analytical_ωd, color = :dodgerblue4, label = "Analytical")
# lines!(ax, ωc .- ωd_list, deriv_numerical_ωd, color = Makie.wong_colors()[2], label = "AD", linestyle = :dash)

# axislegend(ax, padding=0)

# # --- LIMITS ---
# xlims!(ax, ωc - ωd_list[1], ωc - ωd_list[end])

# # --- SAVING ---
# save(joinpath(@__DIR__, "../figures/autodiff.pdf"), fig, pt_per_unit = 1.0)

# fig

# %%

# smooth_heaviside(t, t0, σ) = 1 / (1 + exp(-(t - t0) / σ))
# coef_hadamard_smooth(p, t) = smooth_heaviside(t, 0, p[4]) * (1 - smooth_heaviside(t, p[1], p[4]))
# coef_CNOT_smooth(p, t) = smooth_heaviside(t, p[2], p[4]) * (1 - smooth_heaviside(t, p[2] + p[3], p[4]))

# coef_γ(p, t) = sqrt(p[5] * (p[6] + 1))
# coef_γT(p, t) = sqrt(p[5] * p[6])

# σm1 = tensor(sigmam(), qeye(2))  # Lowering operator for qubit 1
# σm2 = tensor(qeye(2), sigmam())  # Lowering operator for qubit 2
# σx1 = tensor(sigmax(), qeye(2))  # Pauli X for qubit 1
# σz1 = tensor(sigmaz(), qeye(2))  # Pauli Z for qubit 1
# σx2 = tensor(qeye(2), sigmax())  # Pauli X for qubit 2

# # Hadamard gate Hamiltonian
# H_hadamard = (σx1 - σz1) / sqrt(2)

# # CNOT gate Hamiltonian
# H_cnot = (1 + σz1) * (1 - σx2) / 2

# # Time-dependent Hamiltonian
# H = QobjEvo(H_hadamard, coef_hadamard_smooth) + QobjEvo(H_cnot, coef_CNOT_smooth)

# # Lindblad dissipation operators
# L1 = QobjEvo(σm1, coef_γ)
# L2 = QobjEvo(σm2, coef_γ)
# L3 = QobjEvo(σm1', coef_γT)
# L4 = QobjEvo(σm2', coef_γT)

# c_ops = (L1, L2, L3, L4)

# const L = liouvillian(H, c_ops)

# # Bell state
# const Φp = ( tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)) ) / sqrt(2)

# # Initial state: |00>
# const ψ0 = tensor(basis(2, 1), basis(2, 1))

# # Time list
# const tlist = range(0, π, 100)

# function bell_state_generation(p)
#     sol = mesolve(L, ψ0, tlist;
#         params = p,
#         progress_bar = Val(false),
#         sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP())
#         # sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP(), checkpointing=true)
#     )

#     return 1-fidelity(Φp, sol.states[end])
# end

# # %%

# bell_state_generation([π/2, π/2, π/2, 0.001, 0, 0, 0])

# @code_warntype bell_state_generation([π/2, π/2, 0.01, 0, 0, 0])

# Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), bell_state_generation, [π/2, π/2, 0, 0])

# Zygote.gradient(bell_state_generation, [π/2, π/2, 0, 0])

# Zygote.gradient(bell_state_generation, [π/4, π/2, 0.000001, 0.000001])

# Zygote.gradient(bell_state_generation, rand(6))

# using BenchmarkTools

# @benchmark bell_state_generation($[π/2, π/2, 0, 0])

# @benchmark Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), bell_state_generation, $[π/2, π/2, 0, 0])

# @benchmark Zygote.gradient(bell_state_generation, $[π/2, π/2, 0, 0])

# # %%

# function optimize_parameters(params, params_ub, params_lb; maxiter=100)
#     params_history = Array{Float64}(undef, length(params), maxiter)
#     loss_history = Vector{Float64}(undef, maxiter)

#     state = Optimisers.setup(Optimisers.Adam(0.01), clamp.(params, params_lb, params_ub))

#     for i in 1:maxiter
#         grad = Zygote.gradient(bell_state_generation, params)[1]
#         state, params = Optimisers.update(state, params, grad)  # at every step

#         # params .= abs.(params)  # enforce positivity
#         params .= clamp.(params, params_lb, params_ub)

#         loss_tmp = bell_state_generation(params)
#         println("Step $i: Loss = ", loss_tmp)

#         params_history[:, i] .= params
#         loss_history[i] = loss_tmp
#     end

#     return params_history, loss_history
# end

# # %%

# params = rand(6)
# params[5:6] .*= 0.1
# params_ub = [π, π, π, 0.1, 0.1, 0.1]
# params_lb = [1e-5, 1e-5, 1e-5, 1e-2, 0, 0]

# params_history, loss_history = optimize_parameters(params, params_ub, params_lb; maxiter=150)

# # %%

# fig = Figure(size=(2*plot_figsize_width_single_column_pt, 2*0.8 * plot_figsize_width_single_column_pt))
# ax_loss = Axis(fig[1, 1], yscale=log10)
# ax_params = Axis(fig[2, 1])
# ax_pulse_shape = Axis(fig[3, 1])

# lines!(ax_loss, axes(loss_history, 1), loss_history)

# lines!(ax_params, axes(params_history, 2), params_history[1, :], label=L"Hadamard $\Delta t$")
# lines!(ax_params, axes(params_history, 2), params_history[2, :], label=L"CNOT $t_0$")
# lines!(ax_params, axes(params_history, 2), params_history[3, :], label=L"CNOT $\Delta t$")
# lines!(ax_params, axes(params_history, 2), params_history[4, :], label=L"$\sigma$")
# lines!(ax_params, axes(params_history, 2), params_history[5, :], label=L"\gamma")
# lines!(ax_params, axes(params_history, 2), params_history[6, :], label=L"n_{th}")

# lines!(ax_pulse_shape, tlist, coef_hadamard_smooth.(Ref(params_history[:, end]), tlist), label="Hadamard")
# lines!(ax_pulse_shape, tlist, coef_CNOT_smooth.(Ref(params_history[:, end]), tlist), label="CNOT")

# axislegend(ax_params)
# axislegend(ax_pulse_shape)

# fig

# %%

struct UniformQuadraticItp{T<:AbstractVector}
    t0::Float64
    Δ::Float64
    y::T
end

function UniformQuadraticItp(t::AbstractVector, y::AbstractVector)
    @assert length(t) == length(y) "t and y size mismatch"
    @assert length(t) ≥ 3 "need at least 3 points for quadratic"
    Δ = t[2] - t[1]
    @assert all(abs.(diff(t) .- Δ) .< 1e-10*abs(Δ)) "t must be uniform"
    return UniformQuadraticItp{typeof(y)}(float(t[1]), float(Δ), y)
end

function (itp::UniformQuadraticItp)(x::Real)
    t0, Δ, y = itp.t0, itp.Δ, itp.y
    N = length(y)
    s  = (x - t0)/Δ
    j  = clamp(round(Int, s) + 1, 2, N-1)     # center index
    ξ  = s - (j-1)                             # in [-0.5, 0.5] typically
    Lm = 0.5*ξ*(ξ - 1.0)
    L0 = 1.0 - ξ*ξ
    Lp = 0.5*ξ*(ξ + 1.0)
    @inbounds return y[j-1]*Lm + y[j]*L0 + y[j+1]*Lp
end

struct UniformHermiteCR{T<:AbstractVector}
    t0::Float64; Δ::Float64; y::T
end
function UniformHermiteCR(t::AbstractVector, y::AbstractVector)
    @assert length(t)==length(y) && length(y)≥4
    Δ = t[2]-t[1]; @assert all(abs.(diff(t) .- Δ) .< 1e-10*abs(Δ))
    UniformHermiteCR{typeof(y)}(float(t[1]), float(Δ), y)
end
function (itp::UniformHermiteCR)(x::Real)
    t0, Δ, y = itp.t0, itp.Δ, itp.y; N = length(y)
    s = (x - t0)/Δ
    i = clamp(floor(Int, s)+1, 2, N-2)      # 2..N-2
    u = s - (i-1)                            # in [0,1)
    # tangents (Catmull–Rom): central differences
    @inbounds m_i   = 0.5*(y[i+1]-y[i-1])
    @inbounds m_ip1 = 0.5*(y[i+2]-y[i])
    u2 = u*u; u3 = u2*u
    h00 =  2u3 - 3u2 + 1
    h10 =      u3 - 2u2 + u
    h01 = -2u3 + 3u2
    h11 =      u3 -   u2
    @inbounds return h00*y[i] + h10*m_i + h01*y[i+1] + h11*m_ip1
end

# %%

ω0 = 4.0
γ = 1e-4
g = 0.01
const nsteps = 100
const tlist = range(0, 100, nsteps)

σm1 = tensor(sigmam(), qeye(2))  # Lowering operator for qubit 1
σm2 = tensor(qeye(2), sigmam())  # Lowering operator for qubit 2
σx1 = tensor(sigmax(), qeye(2))  # Pauli X for qubit 1
σy1 = tensor(sigmay(), qeye(2))  # Pauli Y for qubit 1
σz1 = tensor(sigmaz(), qeye(2))  # Pauli Z for qubit 1
σx2 = tensor(qeye(2), sigmax())  # Pauli X for qubit 2
σy2 = tensor(qeye(2), sigmay())  # Pauli Y for qubit 2
σz2 = tensor(qeye(2), sigmaz())  # Pauli Z for qubit 2

H0 = ω0 / 2 * (σz1 + σz2) + g * (σm1 * σm2' + σm1' * σm2)

get_amp_idx(t) = clamp(searchsortedfirst(tlist, t), 1, nsteps)
function interp_p(p, t)
    t_idx = get_amp_idx(t)
    return p[t_idx]

    # return UniformQuadraticItp(tlist, p)(t)
    # return UniformHermiteCR(tlist, p)(t)
    # return BSplineApprox(p, tlist, 3, 30, :Uniform, :Uniform)(t)
end
function coef_σx1(p, t)
    ωdt = interp_p(@view(p[1:nsteps]), t)
    Ωt = interp_p(@view(p[1+nsteps:2nsteps]), t)
    return Ωt * cos(ωdt * t)
end
function coef_σz1(p, t)
    Δzt = interp_p(@view(p[1+2nsteps:3nsteps]), t)
    return Δzt
end

H = H0 + QobjEvo(σx1, coef_σx1) + QobjEvo(σz1, coef_σz1)

c_ops = (sqrt(γ) * σm1, sqrt(γ) * σm2)

const L = liouvillian(H, c_ops)

# Initial state: |00>
const ψ0 = tensor(basis(2, 1), basis(2, 1))

# Target state
const Φp = ( tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)) ) / sqrt(2)

function loss(p)
    sol = mesolve(L, ψ0, tlist;
        params = p,
        progress_bar = Val(false),
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
        # sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP(), checkpointing=true)
    )

    # For smoothness
    Δ_param = sum(abs2, diff(@view(p[1:nsteps]))) + sum(abs2, diff(@view(p[1+nsteps:2nsteps]))) + sum(abs2, diff(@view(p[1+2nsteps:3nsteps])))

    return 1-fidelity(Φp, sol.states[end])^2 + 0.005 * Δ_param
end

function optimize_parameters(params, params_ub, params_lb; maxiter=100)
    params_history = Array{Float64}(undef, length(params), maxiter)
    loss_history = Vector{Float64}(undef, maxiter)

    # params0 = clamp.(params, params_lb, params_ub)
    params0 = copy(params)
    state = Optimisers.setup(Optimisers.Adam(), params0)

    params_history[:, 1] .= params0
    loss_history[1] = loss(params0)

    foreach(2:maxiter) do i
        grad = Zygote.gradient(loss, params)[1]
        state, params = Optimisers.update(state, params, grad)  # at every step

        # params .= clamp.(params, params_lb, params_ub)

        loss_tmp = loss(params)
        println("Step $i: Loss = ", loss_tmp)
        flush(stdout)

        params_history[:, i] .= params
        loss_history[i] = loss_tmp
    end

    return params_history, loss_history
end

# %%

params = fill(0.1, 3 * nsteps) .+ 1e-3 .* randn(3 * nsteps)
loss(params)
Zygote.gradient(loss, params)[1]

params_history, loss_history = optimize_parameters(params, nothing, nothing; maxiter=250)

# %%

smooth_heaviside(t, t0, σ) = 1 / (1 + exp(-(t - t0) / σ))
coef_hadamard_smooth(p, t) = smooth_heaviside(t, 0, p[4]) * (1 - smooth_heaviside(t, p[1], p[4]))
coef_CNOT_smooth(p, t) = smooth_heaviside(t, p[2], p[4]) * (1 - smooth_heaviside(t, p[2] + p[3], p[4]))

coef_γ(p, t) = sqrt(p[5])

# Hadamard gate Hamiltonian
H_hadamard = (σx1 - σz1) / sqrt(2)

# CNOT gate Hamiltonian
H_cnot = (1 + σz1) * (1 - σx2) / 2

# Time-dependent Hamiltonian
H_ideal = QobjEvo(H_hadamard, coef_hadamard_smooth) + QobjEvo(H_cnot, coef_CNOT_smooth)

params = [π/2, π/2, π/2, 0.0001, γ]
sol = mesolve(H_ideal, ψ0, tlist, c_ops; params=params)

1 - fidelity(Φp, sol.states[end])^2

# %%

tlist_itp = range(tlist[1], tlist[end], 10*length(tlist))

# ωdt_final = params_history[1:nsteps, end]
# Ωt_final = params_history[1+nsteps:2nsteps, end]
# Δzt_final = params_history[1+2nsteps:3nsteps, end]

# ωdt_final_itp = interp_p(@view(params_history[1:nsteps, end])).(tlist_itp)
# Ωt_final_itp = interp_p(@view(params_history[1+nsteps:2nsteps, end])).(tlist_itp)
# Δzt_final_itp = interp_p(@view(params_history[1+2nsteps:3nsteps, end])).(tlist_itp)

ωdt_final_itp = interp_p.( Ref(@view(params_history[1:nsteps, end])), tlist_itp)
Ωt_final_itp = interp_p.( Ref(@view(params_history[1+nsteps:2nsteps, end])), tlist_itp)
Δzt_final_itp = interp_p.( Ref(@view(params_history[1+2nsteps:3nsteps, end])), tlist_itp)

fig = Figure(size=(2*plot_figsize_width_single_column_pt, 2*0.8 * plot_figsize_width_single_column_pt))

ax_loss = Axis(fig[1, 1], xlabel = "Optimization Iteration", ylabel=L"1 - \mathcal{F}", yscale=log10)
ax_pulse_shape_final = Axis(fig[3, 1], xlabel = L"\gamma t")

lines!(ax_loss, axes(loss_history, 1), loss_history)

lines!(ax_pulse_shape_final, tlist_itp .* γ, Ωt_final_itp, label=L"\Omega (t)")
lines!(ax_pulse_shape_final, tlist_itp .* γ, ωdt_final_itp, label=L"\omega_\mathrm{d} (t)", linestyle=:dash)
lines!(ax_pulse_shape_final, tlist_itp .* γ, Δzt_final_itp, label=L"\Delta z^{(1)}", linestyle=:dashdot)

# lines!(ax_pulse_shape_final, tlist .* γ, ϕt_final, label=L"\phi_\mathrm{had} (t)")

Legend(fig[2, 1], ax_pulse_shape_final, orientation = :horizontal)

# xlims!(ax_pulse_shape_initial, γ * tlist[1], γ * tlist[end])
# ylims!(ax_pulse_shape_initial, 0, nothing)

xlims!(ax_pulse_shape_final, γ * tlist[1], γ * tlist[end])
ylims!(ax_pulse_shape_final, nothing, 0.37)

fig

# %%

N = 10
const xlist = range(0, 10, N)
const x_list_interp = range(0, 10, 5*N)

function test(p)
    p_interp = BSplineApprox(p, xlist, 3, 4, :Uniform, :Uniform)
    return sum(p_interp, x_list_interp)
end


test(rand(N))

Zygote.gradient(test, rand(N))[1]

Enzyme.gradient(Enzyme.Reverse, test, rand(N))
