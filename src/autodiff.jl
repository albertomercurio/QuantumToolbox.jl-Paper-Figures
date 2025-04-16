using QuantumToolbox
using SciMLSensitivity
using Zygote
using CairoMakie

include("_cairomakie_setup.jl")

# %%

const N = 20
const a = destroy(N)
const γ = 1.0

coef_Δ(p, t) = p[1]
coef_F(p, t) = p[2]

H = QobjEvo(a' * a, coef_Δ) + QobjEvo(a + a', coef_F)
c_ops = [sqrt(γ) * a]
const L = liouvillian(H, c_ops)

const ψ0 = fock(N, 0)

function my_f_mesolve(p)
    tlist = range(0, 40, 100)

    sol = mesolve(
        L,
        ψ0,
        tlist,
        progress_bar = Val(false),
        params = p,
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()),
    )

    return real(expect(a' * a, sol.states[end]))
end

# Analytical solution
n_ss(Δ, F) = abs2(F / (Δ + 1im * γ / 2))
n_ss_deriv(Δ, F) = [
    - 2 * F^2 * Δ / (Δ^2 + γ^2 / 4)^2,
    2 * F / (Δ^2 + γ^2 / 4),
]

Δ = 1.0
F = 1.0
params = [Δ, F]

# The factor 2 is due to a bug
grad_qt = Zygote.gradient(my_f_mesolve, params)[1]

grad_exact = n_ss_deriv(Δ, F)

# %%

Δ_list = range(-1, 1, 200)

deriv_analytical_Δ = map(Δ_list) do Δ
    n_ss_deriv(Δ, F)[1]
end

deriv_numerical_Δ = map(Δ_list) do Δ
    Zygote.gradient(my_f_mesolve, [Δ, F])[1][1]
end

# %%

fig = Figure(size=(plot_figsize_width_single_column_pt, 0.6 * plot_figsize_width_single_column_pt))
ax = Axis(fig[1, 1], xlabel = L"\Delta", ylabel = L"\partial \langle \hat{a}^\dagger \hat{a} \rangle_\mathrm{ss} / \partial \Delta")

lines!(ax, Δ_list, deriv_analytical_Δ, color = :dodgerblue4, label = "Analytical")
lines!(ax, Δ_list, deriv_numerical_Δ, color = Makie.wong_colors()[2], label = "AD", linestyle = :dash)

axislegend(ax, padding=0)

# --- LIMITS ---
xlims!(ax, Δ_list[1], Δ_list[end])

# --- SAVING ---
save(joinpath(@__DIR__, "../figures/autodiff.pdf"), fig, pt_per_unit = 1.0)

fig

# %%

