using MKL
using QuantumToolbox
using SciMLSensitivity
using Zygote
using CairoMakie

include("_cairomakie_setup.jl")

# %%

const N  = 20
const a  = destroy(N)
const ωc = 5.0
const γ  = 1.0

coef_a(p, t) = p[2] * exp(1im * p[1] * t)

H = ωc * a' * a + QobjEvo(a, coef_a) + QobjEvo(a', conj ∘ coef_a)
c_ops = [sqrt(γ) * a]
const L = liouvillian(H, c_ops)
const ψ0 = fock(N, 0)
const tlist = range(0, 40, 100)

function my_f_mesolve(p)
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
n_ss(ωd, F) = abs2(F / ((ωc - ωd) + 1im * γ / 2))
n_ss_deriv(ωd, F) = [
    2 * F^2 * (ωc - ωd) / ((ωc - ωd)^2 + γ^2 / 4)^2,
    2 * F / ((ωc - ωd)^2 + γ^2 / 4),
]

ωd = 1.0
F = 1.0
params = [ωd, F]

grad_qt = Zygote.gradient(my_f_mesolve, params)[1]

grad_exact = n_ss_deriv(ωd, F)

# %%

ωd_list = ωc .- range(-1, 1, 200)

pr = ProgressBar(length(ωd_list))
deriv_analytical_ωd = map(ωd_list) do ωd
    res = n_ss_deriv(ωd, F)[1]
    next!(pr)
    return res
end

pr = ProgressBar(length(ωd_list))
deriv_numerical_ωd = map(ωd_list) do ωd
    res = Zygote.gradient(my_f_mesolve, [ωd, F])[1][1]
    next!(pr)
    return res
end

# %%

fig = Figure(size=(plot_figsize_width_single_column_pt, 0.6 * plot_figsize_width_single_column_pt))
ax = Axis(fig[1, 1], xlabel = L"\omega_c - \omega_d", ylabel = L"\partial \langle \hat{a}^\dagger \hat{a} \rangle_\mathrm{ss} / \partial \omega_d")

lines!(ax, ωc .- ωd_list, deriv_analytical_ωd, color = :dodgerblue4, label = "Analytical")
lines!(ax, ωc .- ωd_list, deriv_numerical_ωd, color = Makie.wong_colors()[2], label = "AD", linestyle = :dash)

axislegend(ax, padding=0)

# --- LIMITS ---
xlims!(ax, ωc - ωd_list[1], ωc - ωd_list[end])

# --- SAVING ---
save(joinpath(@__DIR__, "../figures/autodiff.pdf"), fig, pt_per_unit = 1.0)

fig

# %%
