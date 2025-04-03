using QuantumToolbox
using BenchmarkTools
using JSON
using CUDA
using CUDA.CUSPARSE
CUDA.allowscalar(false)

# %% [markdown]

# Parameters:

# %%

const N = 50 # Dimension of the Hilbert space
const Δ = 0.1 # Detuning with respect to the drive
const U = -0.05 # Nonlinearity
const F = 2 # Amplitude of the drive
const γ = 1 # Decay rate
const nth = 0.2
const ntraj = 100

# %%

function quantumtoolbox_mesolve(N)
    a = destroy(N)
    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')

    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    tlist = range(0, 10, 100)
    ψ0 = fock(N, 0)

    mesolve(H, ψ0, tlist[1:2], c_ops, e_ops = [a' * a], progress_bar = Val(false)) # Warm-up

    benchmark_result =
        @benchmark mesolve($H, $ψ0, $tlist, $c_ops, e_ops = $([a' * a]), progress_bar = Val(false)).expect

    return benchmark_result.times
end

function quantumtoolbox_mcsolve(N)
    a = destroy(N)
    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')

    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    tlist = range(0, 10, 100)
    ψ0 = fock(N, 0)

    mcsolve(H, ψ0, tlist, c_ops, e_ops = [a' * a], ntraj = 8, progress_bar = Val(false)) # Warm-up

    benchmark_result = @benchmark mcsolve(
        $H,
        $ψ0,
        $tlist,
        $c_ops,
        e_ops = $([a' * a]),
        ntraj = $ntraj,
        progress_bar = Val(false),
    ).expect

    return benchmark_result.times
end

function quantumtoolbox_smesolve(N)
    a = destroy(N)
    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')
    c_ops = [sqrt(γ * nth) * a']
    sc_ops = [sqrt(γ * (1 + nth)) * a]

    tlist = range(0, 10, 100)
    ψ0 = fock(N, 0)

    # `sc_ops` not a vector, to use diagonal noise solvers
    sol_qt_sme = smesolve(H, ψ0, tlist, c_ops, sc_ops[1], e_ops = [a' * a], ntraj = ntraj, progress_bar = Val(false)) # Warm-up
    sol_qt_me = mesolve(H, ψ0, tlist, vcat(c_ops, sc_ops), e_ops = [a' * a], progress_bar = Val(false))

    converged = sum(abs, vec(sol_qt_sme.expect) .- vec(sol_qt_me.expect)) / length(sol_qt_sme.expect)
    converged < 1e-1 || error("smesolve and mesolve results do not match")

    benchmark_result = @benchmark smesolve(
        $H,
        $ψ0,
        $tlist,
        $c_ops,
        $(sc_ops[1]),
        e_ops = $([a' * a]),
        ntraj = $ntraj,
        progress_bar = Val(false),
    ).expect

    return benchmark_result.times
end

function quantumtoolbox_mesolve_gpu(N)
    a = cu(destroy(N))
    H = Δ * a' * a - U / 2 * (a^2)' * a^2 + F * (a + a')

    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    tlist = range(0, 10, 100)
    ψ0 = cu(fock(N, 0))

    mesolve(H, ψ0, tlist[1:2], c_ops, e_ops = [a' * a], progress_bar = Val(false)) # Warm-up

    benchmark_result =
        @benchmark mesolve($H, $ψ0, $tlist, $c_ops, e_ops = $([a' * a]), progress_bar = Val(false)).expect

    return benchmark_result.times
end

# %%

result_mesolve = quantumtoolbox_mesolve(N)
result_mcsolve = quantumtoolbox_mcsolve(N)
result_smesolve = quantumtoolbox_smesolve(N)

# Save the results to a JSON file

results = Dict(
    "quantumtoolbox_mesolve" => result_mesolve,
    "quantumtoolbox_mcsolve" => result_mcsolve,
    "quantumtoolbox_smesolve" => result_smesolve,
)

output_path = joinpath(@__DIR__, "quantumtoolbox_benchmark_results.json")
open(output_path, "w") do file
    JSON.print(file, results)
end

# %% [markdown]

# Varying the Hilbert space dimension $N$

# %%

N_list = floor.(Int, logrange(10, 300, 25))

pr = ProgressBar(length(N_list))
quantumtoolbox_mesolve_N_cpu = map(N_list) do N
    next!(pr)

    quantumtoolbox_mesolve(N)
end

pr = ProgressBar(length(N_list))
quantumtoolbox_mesolve_N_gpu = map(N_list) do N
    next!(pr)

    quantumtoolbox_mesolve_gpu(N)
end

# Save the results to a JSON file

results = Dict(
    "quantumtoolbox_mesolve_N_cpu" => quantumtoolbox_mesolve_N_cpu,
    "quantumtoolbox_mesolve_N_gpu" => quantumtoolbox_mesolve_N_gpu,
)

output_path = joinpath(@__DIR__, "quantumtoolbox_benchmark_results_N.json")
open(output_path, "w") do file
    JSON.print(file, results)
end
