# %% [markdown]
# ---
# title: "The Dynamical Shifted Fock method"
# author: "Alberto Mercurio"

# engine: julia
# ---

# ## Introduction

# All the time-evolution solvers described above allow the use of user-defined callbacks. Indeed, thanks to the [DiffEqCallbacks.jl](https://github.com/SciML/DiffEqCallbacks.jl) package [@RackauckasDifferentialEquations_jl2017][@rackauckas_2024_differentialequations][@DiffEqCallbacks_jl], it is possible to add different actions to the state or the system depending on a given condition. For instance, the `SteadyStateODESolver` solver for the `steadystate` function terminates the integration once the state does not evolve anymore. To demonstrate the power of callbacks, here we show the Dynamical Fock Dimension (DFD) and Dynamical Shifted Fock (DSF) algorithms. The first one is a simple extension of [`mesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.mesolve) with an additional callback that continuously monitors the population of the Fock states in time, increasing or decreasing the dimension of the Hilbert space accordingly. Indeed, finding the correct cutoff dimension is a crucial point to obtain truthful results. This method is accessible through the [`dfd_mesolve`](https://qutip.org/QuantumToolbox.jl/stable/resources/api#QuantumToolbox.dfd_mesolve).

# Here we focus on the DSF method, which is a simple but powerful algorithm that allows the simulation of strongly-driven systems. Indeed, the numerical simulation of strongly-driven, nonlinear, and coupled systems presents several challenges. On one hand, the dynamics of such systems may necessitate the description of states that occupy a significant portion of the Hilbert space, making them difficult to simulate using standard numerical techniques. On the other hand, although semiclassical approximations (e.g., mean-field or cumulant expansion) partially address this issue, they may fail to accurately characterize quantum fluctuations. This is particularly problematic in several cases such as optical bistability, lasing, and the description of entangled states. The DSF is a numerical algorithm designed to efficiently tackle these challenges. We apply this method to the case of Lindblad master equation, but the same approach can be also applied to Monte Carlo quantum trajectories. 

# ## Description of the method

# The DSF algorithm is based on efficient manipulation of coherent states and displacement operators.

# A coherent state is defined as:
# $$
#     \vert \alpha \rangle = e^{-|\alpha|^2/2} \sum_{n=0}^{+ \infty} \frac{\alpha^n}{\sqrt{n!}} \vert n \rangle \, ,
# $$ {#eq-coherent-state}
# which can be also seen as the eigenstate of the bosonic annihilation operator $\hat{a} \vert \alpha \rangle = \alpha \vert \alpha \rangle$. Thus we define $\alpha = \mel{\alpha}{\hat{a}}{\alpha}$ as the coherence of the state.

# A coherent state can be obtained from the action of the displacement operator
# $$
#     \hat{D} (\alpha) = e^{\alpha \hat{a}^\dagger - \alpha^* \hat{a}} \, , \quad \, \hat{D}(\alpha) \hat{D}^\dagger (\alpha) = \hat{D} (\alpha) \hat{D} (-\alpha) = 1
# $$ {#eq-displacement-operator}
# to the vacuum state $\vert 0 \rangle$, namely $\vert \alpha \rangle = \hat{D} (\alpha) \vert 0 \rangle$. From this definition and from (@eq-displacement-operator), it follows that
# $$
#     \vert 0 \rangle = \hat{D}^\dagger (\alpha) \vert \alpha \rangle \, .
# $$ {#eq-coherent-to-vacuum}
# It can also be proven that, given any two coherent states $\ket{\alpha}$ and $\ket{\beta}$, one has
# $$
#      \ket{\alpha} \propto \hat{D} (\alpha-\beta)  \ket{\beta}\, .
# $$
# The equations above are the key points of the DSF algorithm. Indeed, a coherent state defined in the Fock basis as in (@eq-coherent-state) can have large occupation probabilities in the high-energy states. However, thanks to (@eq-coherent-to-vacuum), it is possible to rotate a high-energy coherent state into the vacuum state $\vert 0 \rangle$. Numerically speaking, this allows us to treat a coherent state with a vector where all the elements are zero, except for the first one. Hence, we can describe it with a smaller vector, given that we keep track of its coherence.

# This approach is also valid when the state is almost-coherent-like, with small quantum fluctuations around a macroscopic coherent state. In this case, given such a state $\vert \psi \rangle$, we have
# $$
#     \hat{D}^\dagger (\alpha) \vert \psi \rangle = \sum_n c_n \vert n \rangle \, ,
# $$
# where $\vert c_n \vert$ rapidly decreases as $n$ increases. Once again, this state can still be represented with a small Fock basis, and the accuracy of the DSF depends on the size of this basis.

# Now that we have explained the key ingredients, we present a simple demonstration of the algorithm. Let us start by considering a closed quantum system with the initial state $\vert \psi (t_0 = 0) \rangle$, whose unitary dynamics is governed by a generic Hamiltonian $\hat{H}$. For fixed-time-step integration, the algorithm is presented in the algorithm below, and its extension to adaptive-time-step integrators is straightforward as DiffEqCallbacks.jl supports very general integration algorithms.

# ::: {.algorithm}
# **Algorithm 1: Dynamical Shifted Fock Algorithm**

# 1. **Initialize:**  
#    Set the state $|\psi(0)\rangle$, Hamiltonian $\hat{H}$, final time $t_f$, and threshold $\delta_\alpha^{(\mathrm{max})}$. Also, set $t \gets 0$.

# 2. **While** $t < t_f$:
#    1. **Evolve:**  
#       Integrate 
#       $$
#       i\frac{d}{dt} |\psi(t)\rangle = \hat{H} |\psi(t)\rangle
#       $$
#       from $t$ to $t+\Delta t$.
#    2. **Compute Coherence:**  
#       $\alpha_{t+\Delta t} \gets \langle \psi(t+\Delta t) | \hat{a} | \psi(t+\Delta t) \rangle.$
#    3. **Compute Shift:**  
#       $\delta_\alpha \gets \alpha_{t+\Delta t} - \alpha_{t}.$
#    4. **If** $|\delta_\alpha| > \delta_\alpha^{(\mathrm{max})}$:
#       - **Displace state:**  
#         $|\psi(t+\Delta t)\rangle \gets \hat{D}(-\delta_\alpha)|\psi(t+\Delta t)\rangle.$
#       - **Update operators:**  
#         $\hat{H} \gets \hat{D}(-\delta_\alpha) \hat{H} \hat{D}(\delta_\alpha).$
#    5. **Increment Time:**  
#       Set $t \gets t+\Delta t$.
# :::

# To summarize, the DSF algorithm allows to simulate the dynamics of a system by keeping the Hilbert space dimension small, at the expense of tracking the coherence and performing the unitary transformations when reaching a given threshold. This overhead is usually very small, but it increases as the quantum fluctuations around the coherent state increase.

# ## Numerical simulation

# %%
using LinearAlgebra
import QuantumToolbox as qt
import QuantumCumulants
using ModelingToolkit
using CairoMakie

include("_cairomakie_setup.jl")

# %% [markdown]

# Parameters

# %%
# Parameters

F   = 5
Δc  = 0.0
Δa  = 50
γ   = 1
U = 0.002
g = 3

N0 = 200

a0 = qt.tensor(qt.destroy(N0), qt.qeye(2))

σm0 = qt.tensor(qt.qeye(N0), qt.sigmam())
σz0 = qt.tensor(qt.qeye(N0), qt.sigmaz())

# %%

# Define the Hamiltonian
function H(Δc, Δa, g, F, U, a, σz, σm)
    Δc*a'*a + Δa/2 * σz + U*(a^2)'*a^2 + g*(a*σm' + a'*σm) + F*(a+a')
end

# Define collapse operators
function c_ops(γ, a, σm)
    [sqrt(γ)*a, sqrt(γ)*σm]
end

# %% [markdown]
# We now study the number of photons at the steadystate as a function of the cavity-drive detuning

# %%
Δ_list = range(-5, 5, 60)

ss_photons = similar(Δ_list)

pr = qt.ProgressBar(length(Δ_list))
Threads.@threads for (i, Δ) in collect(enumerate(Δ_list))
    H0 = H(Δ, Δa, g, F, U, a0, σz0, σm0)
    c_ops0 = c_ops(γ, a0, σm0)

    ρss = qt.steadystate(H0, c_ops0, solver=qt.SteadyStateLinearSolver())
    ss_photons[i] = real(qt.expect(a0'*a0, ρss))

    qt.next!(pr)
end
# %%

fig = Figure(size=(400, 200))
ax = Axis(fig[1, 1])

lines!(ax, Δ_list, ss_photons)

vlines!(ax, [-0.5], color=:black)

fig

# %% [markdown]
# Full Master Equation

# %%
Δa = -0.2

# a0 = cu(a0)
# sm0 = cu(sm0)
# sp0 = cu(sp0)
# sz0 = cu(sz0)

H0     = H(Δc, Δa, g, F, U, a0, σz0, σm0)
c_ops0 = c_ops(γ, a0, σm0)

ψ0 = qt.tensor(qt.fock(N0, 0), qt.basis(2, 1))
# ψ0 = cu(ψ0)

tlist = range(0, 15/γ, 1000)
tlist_save = [tlist[1], 2.2, 0.9*tlist[end]]

sol = qt.mesolve(H0, ψ0, tlist, c_ops0, e_ops=[a0'*a0, a0], saveat=tlist_save);

# %% [markdown]
# Dynamical Shifted Fock

# %%
function H_dsf(op_l, p)
    Δc = p.Δc
    Δa = p.Δa
    U = p.U
    g = p.g
    F = p.F
    σm = p.σm
    σz = p.σz

    a = op_l[1]

    return H(Δc, Δa, g, F, U, a, σz, σm)
end

function c_ops_dsf(op_l, p)
    γ = p.γ
    σm = p.σm

    a = op_l[1]

    return [√γ*a, √γ*σm]
end

function e_ops_dsf(op_l, p)
    a = op_l[1]
    return [a'*a, a]
end

# %%

N_dsf = 15

a = qt.tensor(qt.destroy(N_dsf), qt.qeye(2))
σm = qt.tensor(qt.qeye(N_dsf), qt.sigmam())
σz = qt.tensor(qt.qeye(N_dsf), qt.sigmaz())

op_l = [a]
α0_l = [qt.expect(a0'*a0, ψ0)]

ψ0_dsf = qt.tensor(qt.fock(N_dsf, 0), qt.basis(2, 1))

p = (Δc=Δc, Δa=Δa, U=U, g=g, F=F, γ=γ, σm=σm, σz=σz)

sol_dsf = qt.dsf_mesolve(H_dsf, ψ0_dsf, tlist, c_ops_dsf, op_l, α0_l, p, e_ops=e_ops_dsf, krylov_dim=5, δα_list=[0.2])

# %% [markdown]
# We now find the last diagonal element of $\hat{\rho}_{t_1}$ that is larger than $10^{-4}$, that would be the selected Hilbert space dimension. The radius of the circle to draw will be the square root of that index.

# %%
population_tol = 1e-4

i = 2
ρt = qt.Qobj(Array(sol.states[i].data), dims=sol.states[i].dims)
ρt = qt.ptrace(ρt, 1)
# max_population = abs.(real.(diag(ρt.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(ρt.data))))
α2_max_full = sqrt(idx)

i = 3
ρt = qt.Qobj(Array(sol.states[i].data), dims=sol.states[i].dims)
ρt = qt.ptrace(ρt, 1)
# max_population = abs.(real.(diag(ρt.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(ρt.data))))
α3_max_full = sqrt(idx)

α_max_full = max(α2_max_full, α3_max_full)
# %%

xvec = range(-15, 15, 500)
yvec = range(-15, 15, 500)
xvec_gpu = (collect(xvec))
yvec_gpu = (collect(yvec))

i = 1
ρt = qt.Qobj(Array(sol.states[i].data), dims=sol.states[i].dims)
ρt = qt.ptrace(ρt, 1)
α1, δα1 = qt.get_coherence(ρt)
# max_population = abs.(real.(diag(δα1.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(δα1.data))))
α1_max = sqrt(idx)
wig1 = qt.wigner(ρt, xvec_gpu, yvec_gpu, g=2) |> Array
vmax1 = maximum(abs, wig1)

i = 2
ρt = qt.Qobj(Array(sol.states[i].data), dims=sol.states[i].dims)
ρt = qt.ptrace(ρt, 1)
α2, δα2 = qt.get_coherence(ρt)
# max_population = abs.(real.(diag(δα2.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(δα2.data))))
α2_max = sqrt(idx)
wig2 = qt.wigner(ρt, xvec_gpu, yvec_gpu, g=2) |> Array
vmax2 = maximum(abs, wig2)

i = 3
ρt = qt.Qobj(Array(sol.states[i].data), dims=sol.states[i].dims)
ρt = qt.ptrace(ρt, 1)
α3, δα3 = qt.get_coherence(ρt)
# max_population = abs.(real.(diag(δα3.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(δα3.data))))
α3_max = sqrt(idx)
wig3 = qt.wigner(ρt, xvec_gpu, yvec_gpu, g=2) |> Array
vmax3 = maximum(abs, wig3)

α_max = max(α1_max, α2_max, α3_max)

println(minimum(wig2), "  -  ", maximum(wig2))

# %%

full_case_color = :navy # Makie.wong_colors()[1]
dsf_case_color = Makie.wong_colors()[2]
qc_case_color = Makie.wong_colors()[3]

fig = Figure(size=(plot_figsize_width_pt, 0.6*plot_figsize_width_pt))

ax1 = Axis(fig[1, 1:3], xminorticksvisible=true, yminorticksvisible=true, xaxisposition=:top,
        xlabel = L"\gamma t", ylabel = L"\langle \hat{a}^\dagger \hat{a} \rangle")

ax2 = Axis(fig[2, 1], xticks=-10:10:10, yticks=-10:10:10, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)", ylabel = L"\mathrm{Im}(\alpha)")
ax3 = Axis(fig[2, 2], xticks=-10:10:10, yticks=-10:10:10, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)")
ax4 = Axis(fig[2, 3], xticks=-10:10:10, yticks=-10:10:10, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)")
# labelx = Label(fig[2, :], L"\mathrm{Re}(\alpha)")


lines!(ax1, tlist, real.(sol.expect[1, :]), label="Full", color=full_case_color)
lines!(ax1, tlist, real.(sol_dsf.expect[1, :]), label="DSF", linestyle=:dash, color=dsf_case_color)
# lines!(ax1, sol_qc.t, real.(sol_qc[a₁'*a₁]), label="QC", linestyle=:dashdot, color=qc_case_color)
# lines!(ax1, tlist, abs2.(sol.expect[2, :]))
# vlines!(ax1, [2.6], color=:red)
scatter!(ax1, tlist_save, real.(qt.expect(a0'*a0, sol.states)), color=:grey, markersize=5, marker=:rect)
lines!(ax1, fill(tlist_save[1], 2), [0, real(qt.expect(a0'*a0, sol.states[1]))], color=:grey, linestyle=:dash, linewidth=1)
lines!(ax1, fill(tlist_save[2], 2), [0, real(qt.expect(a0'*a0, sol.states[2]))], color=:grey, linestyle=:dash, linewidth=1)
lines!(ax1, fill(tlist_save[3], 2), [0, real(qt.expect(a0'*a0, sol.states[3]))], color=:grey, linestyle=:dash, linewidth=1)

axislegend(ax1, orientation=:horizontal, position=:cb)


# heatmap!(ax2, xvec, yvec, Array(wigner(fock(N0, 70), xvec_gpu, yvec_gpu))', rasterize=true, colorrange=(-vmax1, vmax1), colormap=:balance, interpolate=true)
heatmap!(ax2, xvec, yvec, wig1', rasterize=true, colorrange=(-vmax1, vmax1), colormap=:balance, interpolate=true)
heatmap!(ax3, xvec, yvec, wig2', rasterize=true, colorrange=(-vmax2, vmax2), colormap=:balance, interpolate=true)
heatmap!(ax4, xvec, yvec, wig3', rasterize=true, colorrange=(-vmax3, vmax3), colormap=:balance, interpolate=true)

# contour!(ax2, xvec, yvec, wig1', levels=25, colorrange=(-vmax1, vmax1), colormap=:balance)
# contour!(ax3, xvec, yvec, wig2', levels=25, colorrange=(-vmax2, vmax2), colormap=:balance)
# contour!(ax4, xvec, yvec, wig3', levels=25, colorrange=(-vmax3, vmax3), colormap=:balance)

arc!(ax2, Point2(0, 0), α_max_full, 0, 2π, color=full_case_color)
arc!(ax3, Point2(0, 0), α_max_full, 0, 2π, color=full_case_color)
arc!(ax4, Point2(0, 0), α_max_full, 0, 2π, color=full_case_color)

arc!(ax2, Point2(real(α1), imag(α1)), α_max, 0, 2π, linestyle=:dash, color=dsf_case_color)
arc!(ax3, Point2(real(α2), imag(α2)), α_max, 0, 2π, linestyle=:dash, color=dsf_case_color)
arc!(ax4, Point2(real(α3), imag(α3)), α_max, 0, 2π, linestyle=:dash, color=dsf_case_color)

xlims!(ax1, tlist[1]-0.05, tlist[end]+0.05)
xlims!(ax2, xvec[1], xvec[end])
xlims!(ax3, xvec[1], xvec[end])
xlims!(ax4, xvec[1], xvec[end])

ylims!(ax1, 0, nothing)
ylims!(ax2, yvec[1], yvec[end])
ylims!(ax3, yvec[1], yvec[end])
ylims!(ax4, yvec[1], yvec[end])

linkyaxes!(ax2, ax3, ax4)
ax3.yticklabelsvisible = false
ax4.yticklabelsvisible = false


rowsize!(fig.layout, 2, Aspect(2, 1))
# colsize!(fig.layout, 2, Aspect(1, 1))
colgap!(fig.layout, 3)
rowgap!(fig.layout, 20)




text!(ax1, 0, 1, text = "(a)", align = (:left, :top), offset = (3, -2), space = :relative, font=:bold)
text!(ax2, 0, 1, text = "(b)", align = (:left, :top), offset = (3, -2), space = :relative, font=:bold)
text!(ax3, 0, 1, text = "(c)", align = (:left, :top), offset = (3, -2), space = :relative, font=:bold)
text!(ax4, 0, 1, text = "(d)", align = (:left, :top), offset = (3, -2), space = :relative, font=:bold)



poly_kwargs = (color=cgrad(:balance)[128], strokecolor=:black, strokewidth=0.5, linestyle=:solid, alpha=1)

ax_origin = ax2.scene.viewport[].origin
ax_widths = ax2.scene.viewport[].widths
snapshot_point = posFig(ax1, tlist_save[1], 0)
poly_x = [snapshot_point[1], ax_origin[1], ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

ax_origin = ax3.scene.viewport[].origin
ax_widths = ax3.scene.viewport[].widths
snapshot_point = posFig(ax1, tlist_save[2], 0)
poly_x = [snapshot_point[1], ax_origin[1], ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

ax_origin = ax4.scene.viewport[].origin
ax_widths = ax4.scene.viewport[].widths
snapshot_point = posFig(ax1, tlist_save[3], 0)
poly_x = [snapshot_point[1], ax_origin[1], ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

# translate!(prova_scatter, 0, 0, 1)

# save(joinpath(data_folder, "DSF_master_equation.pdf"), fig, pt_per_unit=1)

save("../figures/dynamical_shifted_fock.pdf", fig, pt_per_unit=1)

fig

# %%