using LinearAlgebra
using QuantumToolbox
using BenchmarkTools
using JSON

run_gpu = get(ENV, "RUN_GPU_BENCHMARK", "false") == "true"

# %% [markdown]

# Parameters:

# %%

const Jx = 1
const hz = 0.2

const Δ = 0.1 # Detuning with respect to the drive
const U = -0.05 # Nonlinearity
const F = 2 # Amplitude of the drive
const nth = 0.2 # Thermal photons

const γ = 1 # Decay rate
const ntraj = 100

const N_list_cpu = 2:12
const N_list_gpu = 2:12

# %%

function local_op(op, i, N)
    dims = ntuple(d -> 2, N)
    I_pre = I(prod(dims[1:i-1]))
    I_post = I(prod(dims[i+1:end]))
    data = kron(I_pre, op.data, I_post)
    return QuantumObject(data, Operator(), dims)
end

function generate_system(N, ::Val{:ising})
    # iter = Iterators.product(1:N, 1:N)
    # iter_filtered = Iterators.filter(x -> x[1] < x[2], iter)

    Hz = hz * sum(i->local_op(sigmaz(), i, N), 1:N)
    # Hxx = Jx * sum(x -> local_op(sigmax(), x[1], N) * local_op(sigmax(), x[2], N), iter_filtered)
    Hxx = Jx * sum(i -> local_op(sigmax(), i, N) * local_op(sigmax(), i+1, N), 1:N-1)
    H = Hz + Hxx

    c_ops = [sqrt(γ) * local_op(sigmam(), i, N) for i in 1:N]

    return H, c_ops
end

initial_state(N, ::Val{:ising}) = tensor([basis(2, 1) for _ in 1:N]...)
initial_state(N, ::Val{:nho}) = fock(N, 0)

function generate_system(N, ::Val{:nho})
    a = destroy(N)
    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')

    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    return H, c_ops
end

function quantumtoolbox_mesolve(N, system_type::Val)
    H, c_ops = generate_system(N, system_type)

    tlist = range(0, 10, 100)
    ψ0 = initial_state(N, system_type)

    mesolve(H, ψ0, tlist[1:2], c_ops, e_ops = [H], progress_bar = Val(false)) # Warm-up

    benchmark_result =
        @benchmark mesolve($H, $ψ0, $tlist, $c_ops, e_ops = $([H]), progress_bar = Val(false)).expect

    return benchmark_result.times
end

function quantumtoolbox_mcsolve(N, system_type::Val)
    H, c_ops = generate_system(N, system_type)

    tlist = range(0, 10, 100)
    ψ0 = initial_state(N, system_type)

    mcsolve(H, ψ0, tlist, c_ops, e_ops = [H], ntraj = 8, progress_bar = Val(false)) # Warm-up

    benchmark_result = @benchmark mcsolve(
        $H,
        $ψ0,
        $tlist,
        $c_ops,
        e_ops = $([H]),
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

function quantumtoolbox_mesolve_gpu(N, system_type::Val)
    H_cpu, c_ops_cpu = generate_system(N, system_type)
    H = cu(H_cpu)
    c_ops = [cu(x) for x in c_ops_cpu]

    tlist = range(0, 10, 100)
    ψ0 = cu(initial_state(N, system_type))

    mesolve(H, ψ0, tlist[1:2], c_ops, e_ops = [H], progress_bar = Val(false)) # Warm-up

    benchmark_result =
        @benchmark mesolve($H, $ψ0, $tlist, $c_ops, e_ops = $([H]), progress_bar = Val(false)).expect

    return benchmark_result.times
end

# %%

if !run_gpu
    N = 50
    result_mesolve = quantumtoolbox_mesolve(N, Val(:nho))
    result_mcsolve = quantumtoolbox_mcsolve(N, Val(:nho))
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

    pr = ProgressBar(length(N_list_cpu))
    quantumtoolbox_mesolve_N_cpu = map(N_list_cpu) do N
        res = quantumtoolbox_mesolve(N, Val(:ising))
        next!(pr)
        res
    end

    # Save the results to a JSON file

    results = Dict(
        "quantumtoolbox_mesolve_N_cpu" => quantumtoolbox_mesolve_N_cpu,
    )

    output_path = joinpath(@__DIR__, "quantumtoolbox_benchmark_results_N_cpu.json")
    open(output_path, "w") do file
        JSON.print(file, results)
    end
else
    using CUDA
    using CUDA.CUSPARSE
    CUDA.allowscalar(false)

    pr = ProgressBar(length(N_list_gpu))
    quantumtoolbox_mesolve_N_gpu = map(N_list_gpu) do N
        res = quantumtoolbox_mesolve_gpu(N, Val(:ising))
        next!(pr)
        res
    end

    # Save the results to a JSON file

    results = Dict(
        "quantumtoolbox_mesolve_N_gpu" => quantumtoolbox_mesolve_N_gpu,
    )

    output_path = joinpath(@__DIR__, "quantumtoolbox_benchmark_results_N_gpu.json")
    open(output_path, "w") do file
        JSON.print(file, results)
    end
end

