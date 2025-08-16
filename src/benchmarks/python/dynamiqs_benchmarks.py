# %%
import numpy as np
import jax
import jax.numpy as jnp
import dynamiqs
import timeit
from tqdm import tqdm
import json
import os

dynamiqs.set_device("cpu")
dynamiqs.set_precision("double") # Set the same precision as the others
run_gpu = os.getenv("RUN_GPU_BENCHMARK", "false") == "true"

# %% [markdown]

# Parameters:

# %%
Jx = 10.4
hz = 10.0

Δ = 0.1 # Detuning with respect to the drive
U = -0.05 # Nonlinearity
F = 2 # Amplitude of the drive
nth = 0.2 # Thermal photons

γ = 1 # Decay rate
ntraj = 100
stoc_dt = 1e-3

N_list_cpu = range(2, 13)
N_list_gpu = range(2, 13)

# %%

def local_op(op, i, N):
    ops = [dynamiqs.eye(2) for _ in range(N)]
    ops[i] = op
    return dynamiqs.tensor(*ops)

def generate_system(N, system_type):
    if system_type == "ising":
        Hz = hz * sum(local_op(dynamiqs.sigmaz(), i, N) for i in range(N))
        # Hxx = Jx * sum(local_op(dynamiqs.sigmax(), i, N) @ local_op(dynamiqs.sigmax(), j, N) for i in range(N) for j in range(i+1, N))
        Hxx = Jx * sum(local_op(dynamiqs.sigmax(), i, N) @ local_op(dynamiqs.sigmax(), i+1, N) for i in range(N-1))
        H = Hz + Hxx

        c_ops = [jnp.sqrt(γ) * local_op(dynamiqs.sigmam(), i, N) for i in range(N)]

        e_ops = [local_op(dynamiqs.sigmaz(), N-1, N)]

        return H, c_ops, e_ops
    elif system_type == "nho":
        a = dynamiqs.destroy(N)
        H = Δ * a.dag() @ a - U/2 * a.dag() @ a.dag() @ a @ a + F * (a + a.dag())

        c_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]

        e_ops = [a.dag() @ a]

        return H, c_ops, e_ops

def initial_state(N, system_type):
    if system_type == "ising":
        return dynamiqs.tensor(*[dynamiqs.basis(2, 0) for _ in range(N)])
    elif system_type == "nho":
        return dynamiqs.fock(N, 0)

def dynamiqs_mesolve(N, system_type, num_repeats=100):
    """Benchmark dynamiqs.mesolve using timeit for more accurate timing."""
    H, c_ops, e_ops = generate_system(N, system_type)

    tlist = jnp.linspace(0, 10, 100)
    ψ0 = initial_state(N, system_type)

    options = dynamiqs.Options(progress_meter = False, save_states=False)
    method = dynamiqs.method.Tsit5(rtol=1e-6, atol=1e-8)

    dynamiqs.mesolve(H, c_ops, ψ0, tlist[0:2], exp_ops=e_ops, options=options, method=method).states # Warm-up

    # Define the statement to benchmark
    def solve():
        dynamiqs.mesolve(H, c_ops, ψ0, tlist, exp_ops=e_ops, options=options, method=method).expects

    solve() # Warm-up

    # Run the benchmark using timeit. We run it one more time to remove the precompilation time of the first call
    times = timeit.repeat(solve, repeat=num_repeats+1, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times[1:]]  # List of times in nanoseconds

def dynamiqs_mcsolve(N, system_type, ntraj, num_repeats=100):
    H, c_ops, e_ops = generate_system(N, system_type)

    tlist = jnp.arange(0, 10, stoc_dt*20)
    ψ0 = initial_state(N, system_type)

    # define a certain number of PRNG key, one for each trajectory
    key = jax.random.PRNGKey(20)
    keys = jax.random.split(key, ntraj)

    options = dynamiqs.Options(save_states=False)
    method = dynamiqs.method.EulerJump(dt=stoc_dt)

    sol_mc = dynamiqs.jssesolve(H, c_ops, ψ0, tlist, keys, exp_ops=e_ops, options=options, method=method) # Warm-up
    sol_me = dynamiqs.mesolve(H, c_ops, ψ0, tlist, exp_ops=e_ops, options=options)
    # Test if the two methods give the same result up to sol tolerance
    expect_sse = jnp.sum(sol_mc.expects, axis=0)[0,:] / ntraj
    convergence_metric = jnp.sum(jnp.abs(expect_sse - sol_me.expects[0,:])) / len(tlist)
    print(f"mcsolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert jnp.allclose(expect_sse, sol_me.expects[0,:], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        dynamiqs.jssesolve(H, c_ops, ψ0, tlist, keys, exp_ops=e_ops, options=options, method=method).expects

    solve() # Warm-up

    # Run the benchmark using timeit. We run it one more time to remove the precompilation time of the first call
    times = timeit.repeat(solve, repeat=num_repeats+1, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times[1:]]  # List of times in nanoseconds

def dynamiqs_smesolve(N, ntraj, num_repeats=100):
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a - U/2 * a.dag() @ a.dag() @ a @ a + F * (a + a.dag())
    sc_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]
    etas = [1, 0]

    tlist = jnp.arange(0, 10, stoc_dt*20)
    ψ0 = dynamiqs.fock(N, 0)

    # define a certain number of PRNG key, one for each trajectory
    key = jax.random.PRNGKey(20)
    keys = jax.random.split(key, ntraj)

    method = dynamiqs.method.EulerMaruyama(dt=stoc_dt)
    options = dynamiqs.Options(save_states=False)

    sol_sme = dynamiqs.dsmesolve(H, sc_ops, etas, ψ0, tlist, keys, exp_ops=[a.dag() @ a], method=method, options=options) # Warm-up
    sol_me = dynamiqs.mesolve(H, [sc_ops[0], sc_ops[1]], ψ0, tlist, exp_ops=[a.dag() @ a], options=options)
    # Test if the two methods give the same result up to sol tolerance
    expect_sme = jnp.sum(sol_sme.expects, axis=0)[0,:] / ntraj
    convergence_metric = jnp.sum(jnp.abs(expect_sme - sol_me.expects[0,:])) / len(tlist)
    print(f"smesolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert jnp.allclose(expect_sme, sol_me.expects[0,:], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        dynamiqs.dsmesolve(H, sc_ops, etas, ψ0, tlist, keys, exp_ops=[a.dag() @ a], method=method, options=options).block_until_ready()

    solve() # Warm-up

    # Run the benchmark using timeit. We run it one more time to remove the precompilation time of the first call
    times = timeit.repeat(solve, repeat=num_repeats+1, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times[1:]]  # List of times in nanoseconds

# %%

if not run_gpu:
    N = 50

    # Benchmark all cases
    benchmark_results = {
        "dynamiqs_mesolve": dynamiqs_mesolve(N, "nho", num_repeats=100),
        "dynamiqs_mcsolve": dynamiqs_mcsolve(N, "nho", ntraj, num_repeats=10),
        "dynamiqs_smesolve": dynamiqs_smesolve(N, ntraj, num_repeats=5),
    }

    # %%

    print("Saving results to JSON...")

    # Save results to JSON
    with open("src/benchmarks/python/dynamiqs_benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)
    # %% [markdown]

    # Varying the Hilbert space dimension

    # %%

    dynamiqs_mesolve_N_cpu = []
    for N in tqdm(N_list_cpu):
        num_repeats = 2
        if N > 6:
            num_repeats = 1
        
        dynamiqs_mesolve_N_cpu.append(dynamiqs_mesolve(N, "ising", num_repeats=num_repeats))

    benchmark_results_N = {
        "dynamiqs_mesolve_N_cpu": dynamiqs_mesolve_N_cpu,
    }

    # %%

    print("Saving results to JSON...")
    # Save results to JSON

    with open("src/benchmarks/python/dynamiqs_benchmark_results_N_cpu.json", "w") as f:
        json.dump(benchmark_results_N, f, indent=4)
else:
    dynamiqs.set_device("gpu")

    dynamiqs_mesolve_N_gpu = []
    # In this way it is safe if it fails due to lack of GPU memory
    for N in tqdm(N_list_gpu):
        num_repeats = 1
        if N > 6:
            num_repeats = 2

        dynamiqs_mesolve_N_gpu.append(dynamiqs_mesolve(N, "ising", num_repeats=num_repeats))

    benchmark_results_N = {
        "dynamiqs_mesolve_N_gpu": dynamiqs_mesolve_N_gpu,
    }

    # %%

    print("Saving results to JSON...")
    # Save results to JSON

    with open("src/benchmarks/python/dynamiqs_benchmark_results_N_gpu.json", "w") as f:
        json.dump(benchmark_results_N, f, indent=4)