using QuantumOptics
using BenchmarkTools
using JSON

# %% [markdown]
# Parameters:
# %%

const N = 50 # Dimension of the Hilbert space
const Δ = 0.1 # Detuning with respect to the drive
const U = -0.05 # Nonlinearity
const F = 2 # Amplitude of the drive
const γ = 1 # Decay rate
const nth = 0.2
const stoc_dt = 1e-3
const ntraj = 100

# %%

function quantumoptics_mesolve(N)
    bas = FockBasis(N)
    a = destroy(bas)

    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')
    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    tlist = range(0, 10, 100)
    ψ0 = fockstate(bas, 0)

    f_out = (t, state) -> expect(a' * a, state)

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

function quantumoptics_mcsolve(N)
    bas = FockBasis(N)
    a = destroy(bas)

    H = Δ * a' * a - U / 2 * a'^2 * a^2 + F * (a + a')
    c_ops = [sqrt(γ * (1 + nth)) * a, sqrt(γ * nth) * a']

    tlist = range(0, 10, 100)
    ψ0 = fockstate(bas, 0)

    quantumoptics_mcwf(tlist, ψ0, H, c_ops, [a' * a], ntraj) # Warm-up

    benchmark_result = @benchmark quantumoptics_mcwf($tlist, $ψ0, $H, $c_ops, $([a' * a]), ntraj)

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

# %%

result_mesolve = quantumoptics_mesolve(N)
result_mcsolve = quantumoptics_mcsolve(N)
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

N_list = floor.(Int, range(10, 400, 25))

quantumoptics_mesolve_N_cpu = map(enumerate(N_list)) do (i, N)
    println(i)

    quantumoptics_mesolve(N)
end

# Save the results to a JSON file

results = Dict(
    "quantumoptics_mesolve_N_cpu" => quantumoptics_mesolve_N_cpu,
)

output_path = joinpath(@__DIR__, "quantumoptics_benchmark_results_N.json")
open(output_path, "w") do file
    JSON.print(file, results)
end
