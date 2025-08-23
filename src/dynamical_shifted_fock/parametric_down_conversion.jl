using QuantumToolbox
using QuantumToolbox: Tsit5
using QuantumCumulants
using ModelingToolkit
using CairoMakie
using BenchmarkTools
using CUDA

include("../_cairomakie_setup.jl")

# %%

const F   = 7.0
const Δ1  = 0.5
const Δ2  = 0.5
const U   = 0.05
const κ1  = 1.0
const κ2  = 0.0
const κ2_2 = 0.01
const J   = 0.05

tlist = range(0, 15, 1000)
tlist_save = [0, 4, tlist[end]]

# %%
########## FULL SYSTEM ##########

const N01     = 140
const N02     = 40
a10     = kron(destroy(N01), eye(N02)) |> cu
a20     = kron(eye(N01), destroy(N02)) |> cu

H0     = Δ1 * a10'*a10 + Δ2 * a20'*a20 - U * (a20^2)' * a20^2 + F * (a10+a10') + J * (a10'*a20^2 + a10*(a20^2)')
c_ops0 = [√κ1 * a10, √κ2 * a20, √κ2_2 * a20^2]

ρ0   = kron(fock(N01, 0), fock(N02, 0)) |> cu

sol0 = mesolve(H0, ρ0, tlist, c_ops0, e_ops=[a10'*a10, a20'*a20, a10, a20], saveat=tlist_save);

# %%

fig = Figure()
ax1 = Axis(fig[1,1]; xlabel=L"t", ylabel="Photon number", title="Jaynes-Cummings model")

lines!(ax1, tlist, real.(sol0.expect[1,:]); label="Mode 1")
lines!(ax1, tlist, real.(sol0.expect[2,:]); label="Mode 2")

i = 3
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt = ptrace(ρt, 2)
fig, ax2 = plot_wigner(ρt, location=fig[2,1], xvec = xvec, yvec = yvec, rasterize=2)

ax2.aspect = AxisAspect(1)

fig

# %%
########## DYNAMICAL SHIFTED FOCK ##########

function H(op_list, p)
    a1 = op_list[1]
    Δ1 * a1'*a1 + Δ2 * a2'*a2 - U * (a2^2)' * a2^2 + F * (a1 + a1') + J * (a1'*a2^2 + a1*(a2^2)')
end
function c_ops(op_list,p)
    a1 = op_list[1]
    [√κ1 * a1, √κ2 * a2, √κ2_2 * a2^2]
end
function e_ops(op_list,p)
    a1 = op_list[1]
    [a1' * a1, a2' * a2, a1, a2]
end

const N1 = 4
const N2 = N02
const a1 = kron(destroy(N1), eye(N2))
const a2 = kron(eye(N1), destroy(N2))

# %%

op_list = [a1]
ψ0  = kron(fock(N1, 0), fock(N2, 0))

sol_dsf = dsf_mesolve(H, ψ0, tlist, c_ops, op_list, e_ops = e_ops);

# %%
########## QUANTUM CUMULANTS ##########

# Define hilbert space
h1 = FockSpace(:cavity1)
h2 = FockSpace(:cavity2)
h = QuantumCumulants.tensor(h1, h2)

# Define the fundamental operators
@qnumbers a₂1::Destroy(h,1) a₂2::Destroy(h,2)
@cnumbers Δ1₂ Δ2₂ F₂ U₂ J₂ κ1₂ κ2₂ κ2_2₂

# Hamiltonian
H₂ = Δ1₂ * a₂1'*a₂1 + Δ2₂ * a₂2'*a₂2 - U₂ * (a₂2^2)' * a₂2^2 + F₂ * (a₂1 + a₂1') + J₂ * (a₂1'*a₂2^2 + a₂1*(a₂2^2)')

# Collapse operators
Jumps₂ = [a₂1, a₂2, a₂2^2]
rates₂ = [κ1₂, κ2₂, κ2_2₂]

# Derive a set of equations
ops = [a₂1'*a₂1, a₂2'*a₂2, a₂1, a₂2]
eqs = meanfield(ops,H₂,Jumps₂;rates=rates₂,order=2)

# Expand the above equations to second order
eqs_expanded = ModelingToolkit.complete(eqs);

# Generate an ODESystem
@named sys = ODESystem(eqs_expanded);

# Solve the system using the OrdinaryDiffEq package
u0 = zeros(ComplexF64, length(eqs_expanded))
u0[1] = abs2(0)
u0[2] = abs2(0)
u0[3] = 0
u0[4] = 0

p = (Δ1₂, Δ2₂, F₂, U₂, κ1₂, κ2₂, κ2_2₂, J₂)
p0 = p .=> (Δ1, Δ2, F, U, κ1, κ2, κ2_2, J)
prob = ODEProblem(sys, u0,(tlist[1],tlist[end]),p0,saveat=tlist, abstol=1e-8, reltol=1e-6)
sol_qc = solve(prob, Tsit5());

# %%

fig = Figure()
ax = Axis(fig[1,1]; xlabel=L"t", ylabel="Photon number", title="Cat state generation")

lines!(ax, tlist, real.(sol0.expect[1,:]); label="Exact 1")
lines!(ax, tlist, real.(sol_dsf.expect[1,:]); label="DSF 1", linestyle=:dash)
lines!(ax, sol_qc.t, real.(sol_qc[a₂1'*a₂1]); label="Quantum cumulants 1", linestyle=:dot)
lines!(ax, tlist, real.(sol0.expect[2,:]); label="Exact 2")
lines!(ax, tlist, real.(sol_dsf.expect[2,:]); label="DSF 2", linestyle=:dash)
lines!(ax, sol_qc.t, real.(sol_qc[a₂2'*a₂2]); label="Quantum cumulants 2", linestyle=:dot)

# ylims!(ax, 0, 16)

axislegend(ax, labelsize=7, position=:rb)

fig

# %%

population_tol = 1e-4
xvec1 = range(-13, 13, 500)
yvec1 = range(-13, 13, 500)
xvec2 = range(-6, 6, 500)
yvec2 = range(-6, 6, 500)
xvec_gpu1 = (collect(xvec1))
yvec_gpu1 = (collect(yvec1))
xvec_gpu2 = (collect(xvec2))
yvec_gpu2 = (collect(yvec2))

i = 2
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt = ptrace(ρt, 1)
idx = findlast(>(population_tol), abs.(real.(diag(ρt.data))))
α2_max_full = sqrt(idx)

i = 3
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt = ptrace(ρt, 1)
idx = findlast(>(population_tol), abs.(real.(diag(ρt.data))))
α3_max_full = sqrt(idx)

α_max_full = max(α2_max_full, α3_max_full)

i = 1
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt1 = ptrace(ρt, 1)
ρt2 = ptrace(ρt, 2)
α1, δα1 = get_coherence(ρt1)
idx = findlast(>(population_tol), abs.(real.(diag(δα1.data))))
α1_max = sqrt(idx)
wig1_1 = wigner(ρt1, xvec_gpu1, yvec_gpu1, g=2) |> Array
wig1_2 = wigner(ρt2, xvec_gpu2, yvec_gpu2, g=2) |> Array
vmax1 = maximum(abs, wig1_1)

i = 2
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt1 = ptrace(ρt, 1)
ρt2 = ptrace(ρt, 2)
α2, δα2 = get_coherence(ρt1)
# max_population = abs.(real.(diag(δα2.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(δα2.data))))
α2_max = sqrt(idx)
wig2_1 = wigner(ρt1, xvec_gpu1, yvec_gpu1, g=2) |> Array
wig2_2 = wigner(ρt2, xvec_gpu2, yvec_gpu2, g=2) |> Array
vmax2 = maximum(abs, wig2_1)

i = 3
ρt = Qobj(Array(sol0.states[i].data), dims=sol0.states[i].dims)
ρt1 = ptrace(ρt, 1)
ρt2 = ptrace(ρt, 2)
α3, δα3 = get_coherence(ρt1)
# max_population = abs.(real.(diag(δα3.data))) |> maximum
idx = findlast(>(population_tol), abs.(real.(diag(δα3.data))))
α3_max = sqrt(idx)
wig3_1 = wigner(ρt1, xvec_gpu1, yvec_gpu1, g=2) |> Array
wig3_2 = wigner(ρt2, xvec_gpu2, yvec_gpu2, g=2) |> Array
vmax3 = maximum(abs, wig3_1)

α_max = max(α1_max, α2_max, α3_max)

# %%

full_case_color = :navy # Makie.wong_colors()[1]
dsf_case_color = Makie.wong_colors()[2]
qc_case_color = Makie.wong_colors()[3]

fig = Figure(size=(plot_figsize_width_pt, 0.5*plot_figsize_width_pt))

grid_cavity1 = GridLayout(fig[1, 1])
grid_cavity2 = GridLayout(fig[1, 2])

# --- Axes Cavity 1 ---

ax_evolution_cavity1 = Axis(grid_cavity1[1, 1:2], xminorticksvisible=true, yminorticksvisible=true, xaxisposition=:top,
        xlabel = L"\gamma t", ylabel = L"\langle \hat{a}_1^\dagger \hat{a}_1 \rangle")

ax_initial_wig_cavity1 = Axis(grid_cavity1[2, 1], xticks=-10:10:10, yticks=-10:10:10, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)", ylabel = L"\mathrm{Im}(\alpha)")
ax_final_wig_cavity1 = Axis(grid_cavity1[2, 2], xticks=-10:10:10, yticks=-10:10:10, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)")

# --- Axes Cavity 2 ---

ax_evolution_cavity2 = Axis(grid_cavity2[1, 1:2], xminorticksvisible=true, yminorticksvisible=true, xaxisposition=:top,
        xlabel = L"\gamma t", ylabel = L"\langle \hat{a}_2^\dagger \hat{a}_2 \rangle")

ax_initial_wig_cavity2 = Axis(grid_cavity2[2, 1], xticks=-5:5:10, yticks=-5:5:5, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)", ylabel = L"\mathrm{Im}(\alpha)")
ax_final_wig_cavity2 = Axis(grid_cavity2[2, 2], xticks=-5:5:5, yticks=-5:5:5, xminorticksvisible=true, yminorticksvisible=true, xlabel = L"\mathrm{Re}(\alpha)")

# --- Time Evolution Cavity 1 ---

lines!(ax_evolution_cavity1, tlist, real.(sol0.expect[1, :]), label="Full", color=full_case_color)
lines!(ax_evolution_cavity1, tlist, real.(sol_dsf.expect[1, :]), label="DSF", linestyle=:dash, color=dsf_case_color)
lines!(ax_evolution_cavity1, sol_qc.t, real.(sol_qc[a₂1'*a₂1]), label="QC", linestyle=:dashdot, color=qc_case_color)

axislegend(ax_evolution_cavity1, orientation=:horizontal, position=:cb, padding=0)

# --- Time Evolution Cavity 2 ---

lines!(ax_evolution_cavity2, tlist, real.(sol0.expect[2, :]), label="Full", color=full_case_color)
lines!(ax_evolution_cavity2, tlist, real.(sol_dsf.expect[2, :]), label="DSF", linestyle=:dash, color=dsf_case_color)
lines!(ax_evolution_cavity2, sol_qc.t, real.(sol_qc[a₂2'*a₂2]), label="QC", linestyle=:dashdot, color=qc_case_color)

# --- Wigner Functions Cavity 1 ---

heatmap!(ax_initial_wig_cavity1, xvec1, yvec1, wig1_1', rasterize=2, colorrange=(-vmax1, vmax1), colormap=:balance, interpolate=true)
heatmap!(ax_final_wig_cavity1, xvec1, yvec1, wig3_1', rasterize=2, colorrange=(-vmax3, vmax3), colormap=:balance, interpolate=true)

arc!(ax_initial_wig_cavity1, Point2(0, 0), α_max_full, 0, 2π, color=full_case_color)
arc!(ax_final_wig_cavity1, Point2(0, 0), α_max_full, 0, 2π, color=full_case_color)

arc!(ax_initial_wig_cavity1, Point2(real(α1), imag(α1)), α_max, 0, 2π, linestyle=:dash, color=dsf_case_color)
arc!(ax_final_wig_cavity1, Point2(real(α3), imag(α3)), α_max, 0, 2π, linestyle=:dash, color=dsf_case_color)

# --- Wigner Functions Cavity 2 ---

heatmap!(ax_initial_wig_cavity2, xvec2, yvec2, wig1_2', rasterize=2, colorrange=(-vmax1, vmax1), colormap=:balance, interpolate=true)
heatmap!(ax_final_wig_cavity2, xvec2, yvec2, wig3_2', rasterize=2, colorrange=(-vmax3, vmax3), colormap=:balance, interpolate=true)


# --- Limits Cavity 1 ---

xlims!(ax_evolution_cavity1, tlist[1]-0.05, tlist[end]+0.05)
xlims!(ax_initial_wig_cavity1, xvec1[1], xvec1[end])
xlims!(ax_final_wig_cavity1, xvec1[1], xvec1[end])

ylims!(ax_evolution_cavity1, 0, nothing)
ylims!(ax_initial_wig_cavity1, yvec1[1], yvec1[end])
ylims!(ax_final_wig_cavity1, yvec1[1], yvec1[end])

# linkyaxes!(ax_initial_wig_cavity1, ax_final_wig_cavity1)
ax_final_wig_cavity1.yticklabelsvisible = false

# --- Limits Cavity 2 ---

xlims!(ax_evolution_cavity2, tlist[1]-0.05, tlist[end]+0.05)
xlims!(ax_initial_wig_cavity2, xvec2[1], xvec2[end])
xlims!(ax_final_wig_cavity2, xvec2[1], xvec2[end])

ylims!(ax_evolution_cavity2, 0, nothing)
ylims!(ax_initial_wig_cavity2, yvec2[1], yvec2[end])
ylims!(ax_final_wig_cavity2, yvec2[1], yvec2[end])

# linkyaxes!(ax_initial_wig_cavity2, ax_final_wig_cavity2)
ax_final_wig_cavity2.yticklabelsvisible = false

# --- Spacing ---

colgap!(fig.layout, 7)

rowsize!(grid_cavity1, 2, Aspect(2, 1))
colgap!(grid_cavity1, 5)
rowgap!(grid_cavity1, 20)

rowsize!(grid_cavity2, 2, Aspect(2, 1))
colgap!(grid_cavity2, 5)
rowgap!(grid_cavity2, 20)

# --- Panels Labels ---

text!(ax_evolution_cavity1, 0, 1, text = "(a)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)
text!(ax_initial_wig_cavity1, 0, 1, text = "(b)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)
text!(ax_final_wig_cavity1, 0, 1, text = "(c)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)

text!(ax_evolution_cavity2, 0, 1, text = "(d)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)
text!(ax_initial_wig_cavity2, 0, 1, text = "(e)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)
text!(ax_final_wig_cavity2, 0, 1, text = "(f)", align = (:left, :top), offset = (2, -1), space = :relative, font=:bold)

# --- Zoom Effect Cavity 1 ---

poly_kwargs = (color=cgrad(:balance)[128], strokecolor=:black, strokewidth=0.5, linestyle=:solid, alpha=1)

snapshot_point1 = posFig(ax_evolution_cavity1, tlist_save[1], 0)
snapshot_point3 = posFig(ax_evolution_cavity1, tlist_save[3], 0)

ax_origin = ax_initial_wig_cavity1.scene.viewport[].origin
ax_widths = ax_initial_wig_cavity1.scene.viewport[].widths
poly_x = [snapshot_point1[1], ax_origin[1], ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point1[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

ax_origin = ax_final_wig_cavity1.scene.viewport[].origin
ax_widths = ax_final_wig_cavity1.scene.viewport[].widths
poly_x = [snapshot_point3[1], ax_origin[1] - 4, ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point3[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

scatter!(fig.scene, [snapshot_point1, snapshot_point3]; color=:grey, markersize=8, marker=:rect)

# --- Zoom Effect Cavity 2 ---

poly_kwargs = (color=cgrad(:balance)[128], strokecolor=:black, strokewidth=0.5, linestyle=:solid, alpha=1)

snapshot_point1 = posFig(ax_evolution_cavity2, tlist_save[1], 0)
snapshot_point3 = posFig(ax_evolution_cavity2, tlist_save[3], 0)

ax_origin = ax_initial_wig_cavity2.scene.viewport[].origin
ax_widths = ax_initial_wig_cavity2.scene.viewport[].widths
poly_x = [snapshot_point1[1], ax_origin[1], ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point1[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

ax_origin = ax_final_wig_cavity2.scene.viewport[].origin
ax_widths = ax_final_wig_cavity2.scene.viewport[].widths
poly_x = [snapshot_point3[1], ax_origin[1] - 4, ax_origin[1] + ax_widths[1]]
poly_y = [snapshot_point3[2], ax_origin[2] + ax_widths[1], ax_origin[2] + ax_widths[2]]
poly!(fig.scene, collect(zip(poly_x, poly_y)); poly_kwargs...)

scatter!(fig.scene, [snapshot_point1, snapshot_point3]; color=:grey, markersize=8, marker=:rect)

# translate!(prova_scatter, 0, 0, 1)

save(joinpath(@__DIR__, "../../figures/dynamical_shifted_fock_CAT.pdf"), fig, pt_per_unit=1)

fig

# %%
