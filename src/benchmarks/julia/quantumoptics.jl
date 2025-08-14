using QuantumOptics
import QuantumToolbox: ProgressBar, next!
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
const stoc_dt = 1e-3
const ntraj = 100

# %%

function generate_system(N, ::Val{:ising})
    b = SpinBasis(1//2)
    bases = tensor([b for _ in 1:N]...)

    iter = Iterators.product(1:N, 1:N)
    iter_filtered = Iterators.filter(x -> x[1] < x[2], iter)

    Hx = hz * sum(i -> embed(bases, i, sigmaz(b)), 1:N)
    Hzz = Jx * sum(x -> embed(bases, x[1], sigmax(b)) * embed(bases, x[2], sigmax(b)), iter_filtered)
    H = Hx + Hzz

    c_ops = [sqrt(γ) * embed(bases, i, sigmam(b)) for i in 1:N]
    return H, c_ops, bases
end

function generate_system(N, ::Val{:nho})
    bas = FockBasis(N)
    a = destroy(bas)

    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')
    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    return H, c_ops, bas
end

initial_state(N, bas, ::Val{:ising}) = basisstate(bas, ones(Int, N))
initial_state(N, bas, ::Val{:nho}) = fockstate(bas, 0)

function quantumoptics_mesolve(N, system_type::Val)
    H, c_ops, bas = generate_system(N, system_type)

    tlist = range(0, 10, 100)
    ψ0 = initial_state(N, bas, system_type)

    f_out = (t, state) -> expect(H, state)

    timeevolution.master(tlist[1:2], ψ0, H, c_ops, fout = f_out) # Warm-up

    benchmark_result =
        @benchmark timeevolution.master($tlist, $ψ0, $H, $c_ops, fout = $f_out, abstol = 1e-8, reltol = 1e-6)

    return benchmark_result.times
end

function quantumoptics_mcwf(tlist, ψ0, H, c_ops, e_ops, ntraj, f_out = (t, state) -> expect(e_ops[1], state))
    Threads.@threads for i in 1:ntraj
        timeevolution.mcwf(
            tlist,
            ψ0,
            H,
            c_ops,
            fout = f_out,
            display_beforeevent = true,
            display_afterevent = true,
            abstol = 1e-8,
            reltol = 1e-6,
        )
    end
end

function quantumoptics_mcsolve(N, system_type)
    H, c_ops, bas = generate_system(N, system_type)

    tlist = range(0, 10, 100)
    ψ0 = initial_state(N, bas, system_type)

    quantumoptics_mcwf(tlist, ψ0, H, c_ops, [H], ntraj) # Warm-up

    benchmark_result = @benchmark quantumoptics_mcwf($tlist, $ψ0, $H, $c_ops, $([H]), ntraj)

    return benchmark_result.times
end

function quantumoptics_smesolve(
    tlist,
    ψ0,
    H,
    c_ops,
    sc_ops,
    e_ops,
    ntraj,
    dt;
    f_out = (t, state) -> expect(e_ops[1], state),
)
    expect_result = zeros(ComplexF64, length(tlist), ntraj)
    Threads.@threads for i in 1:ntraj
        expvals = stochastic.master(tlist, ψ0, H, c_ops, sc_ops; fout = f_out, dt = dt)[2]
        if !isnothing(f_out)
            expect_result[:, i] .= expvals
        end
    end
    return dropdims(sum(expect_result, dims = 2), dims = 2) ./ ntraj
end

function quantumoptics_smesolve(N)
    bas = FockBasis(N)
    a = destroy(bas)

    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')
    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']
    sc_ops = [sqrt(γ * (1 + nth)) * a]

    tlist = 0:stoc_dt*20:10 # Since the solver has non-adaptive time steps
    ψ0 = fockstate(bas, 0)

    f_out = (t, state) -> expect(a' * a, state)

    sol_qo_me = timeevolution.master(tlist, ψ0, H, c_ops, fout = f_out)[2]
    expec_qo_sme = quantumoptics_smesolve(tlist, ψ0, H, c_ops, sc_ops, [a' * a], ntraj, stoc_dt) # Warm-up

    converged = sum(abs, expec_qo_sme .- vec(sol_qo_me)) ./ length(expec_qo_sme)
    converged < 1e-1 || error("smesolve and mesolve results do not match")

    benchmark_result =
        @benchmark quantumoptics_smesolve($tlist, $ψ0, $H, $c_ops, $sc_ops, $([a' * a]), $ntraj, $stoc_dt)

    return benchmark_result.times
end

function quantumoptics_mesolve_gpu(N, system_type::Val)
    H_cpu, c_ops_cpu, bas = generate_system(N, system_type)

    tlist = range(0, 10, 100)
    ψ0_cpu = initial_state(N, bas, system_type)

    H = Operator(H_cpu.basis_l, H_cpu.basis_r, CuSparseMatrixCSR(H_cpu.data))
    c_ops = [Operator(op.basis_l, op.basis_r, CuSparseMatrixCSR(op.data)) for op in c_ops_cpu]
    ψ0 = Ket(ψ0_cpu.basis, CuVector(ψ0_cpu.data))

    f_out = (t, state) -> expect(H, state)

    timeevolution.master(tlist[1:2], ψ0, H, c_ops, fout = f_out) # Warm-up

    benchmark_result =
        @benchmark timeevolution.master($tlist, $ψ0, $H, $c_ops, fout = $f_out, abstol = 1e-8, reltol = 1e-6)

    return benchmark_result.times
end

# %%

N_list = 2:10

if !run_gpu
    N = 50
    result_mesolve = quantumoptics_mesolve(N, Val(:nho))
    result_mcsolve = quantumoptics_mcsolve(N, Val(:nho))
    result_smesolve = quantumoptics_smesolve(N)

    # %%

    # Save the results to a JSON file

    results = Dict(
        "quantumoptics_mesolve" => result_mesolve,
        "quantumoptics_mcsolve" => result_mcsolve,
        "quantumoptics_smesolve" => result_smesolve,
    )

    output_path = joinpath(@__DIR__, "quantumoptics_benchmark_results.json")
    open(output_path, "w") do file
        JSON.print(file, results)
    end

    # %% [markdown]

    # Varying the Hilbert space dimension $N$

    # %%

    pr = ProgressBar(length(N_list))
    quantumoptics_mesolve_N_cpu = map(N_list) do N
        res = quantumoptics_mesolve(N, Val(:ising))
        next!(pr)
        res
    end

    # Save the results to a JSON file

    results = Dict(
        "quantumoptics_mesolve_N_cpu" => quantumoptics_mesolve_N_cpu,
    )

    output_path = joinpath(@__DIR__, "quantumoptics_benchmark_results_N_cpu.json")
    open(output_path, "w") do file
        JSON.print(file, results)
    end
else
    using CUDA
    using CUDA.CUSPARSE
    CUDA.allowscalar(false)

    # Overloaded expect function for GPU support
    function QuantumOptics.expect(op::DataOperator{B1,B2}, state::Operator{B2,B2, <:CuArray}) where {B1,B2}
        return tr(op * state)
    end

    pr = ProgressBar(length(N_list))
    quantumoptics_mesolve_N_gpu = map(N_list) do N
        res = quantumoptics_mesolve_gpu(N, Val(:ising))
        next!(pr)
        res
    end

    # Save the results to a JSON file

    results = Dict(
        "quantumoptics_mesolve_N_gpu" => quantumoptics_mesolve_N_gpu,
    )

    output_path = joinpath(@__DIR__, "quantumoptics_benchmark_results_N_gpu.json")
    open(output_path, "w") do file
        JSON.print(file, results)
    end
end
