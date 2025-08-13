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
run_gpu = os.getenv("RUN_GPU_BENCHMARK", "False") == "True"

# %% [markdown]

# Parameters:

# %%
N = 50 # Dimension of the Hilbert space
Δ = 0.1 # Detuning with respect to the drive
U = -0.05 # Nonlinearity
F = 2 # Amplitude of the drive
γ = 1 # Decay rate
nth = 0.2
ntraj = 100
stoc_dt = 1e-3

# %%

def dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=100):
    """Benchmark dynamiqs.mesolve using timeit for more accurate timing."""
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a - U/2 * a.dag() @ a.dag() @ a @ a + F * (a + a.dag())
    c_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]

    tlist = jnp.linspace(0, 10, 100)
    ψ0 = dynamiqs.fock(N, 0)

    options = dynamiqs.Options(progress_meter = False, save_states=False)
    method = dynamiqs.method.Tsit5(rtol=1e-6, atol=1e-8)

    dynamiqs.mesolve(H, c_ops, ψ0, tlist[0:2], exp_ops=[a.dag() @ a], options=options, method=method).states # Warm-up

    # Define the statement to benchmark
    def solve():
        dynamiqs.mesolve(H, c_ops, ψ0, tlist, exp_ops=[a.dag() @ a], options=options, method=method).expects

    solve() # Warm-up

    # Run the benchmark using timeit. We run it one more time to remove the precompilation time of the first call
    times = timeit.repeat(solve, repeat=num_repeats+1, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times[1:]]  # List of times in nanoseconds

def dynamiqs_mcsolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a - U/2 * a.dag() @ a.dag() @ a @ a + F * (a + a.dag())
    c_ops = [jnp.sqrt(γ * (1 + nth)) * a]

    tlist = jnp.arange(0, 10, stoc_dt*20)
    ψ0 = dynamiqs.fock(N, 0)

    # define a certain number of PRNG key, one for each trajectory
    key = jax.random.PRNGKey(20)
    keys = jax.random.split(key, ntraj)

    options = dynamiqs.Options(save_states=False)
    method = dynamiqs.method.EulerJump(dt=stoc_dt)

    sol_mc = dynamiqs.jssesolve(H, c_ops, ψ0, tlist, keys, exp_ops=[a.dag() @ a], options=options, method=method) # Warm-up
    sol_me = dynamiqs.mesolve(H, c_ops, ψ0, tlist, exp_ops=[a.dag() @ a], options=options)
    # Test if the two methods give the same result up to sol tolerance
    expect_sse = jnp.sum(sol_mc.expects, axis=0)[0,:] / ntraj
    convergence_metric = jnp.sum(jnp.abs(expect_sse - sol_me.expects[0,:])) / len(tlist)
    print(f"mcsolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert jnp.allclose(expect_sse, sol_me.expects[0,:], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        dynamiqs.jssesolve(H, c_ops, ψ0, tlist, keys, exp_ops=[a.dag() @ a], options=options, method=method).expects

    solve() # Warm-up

    # Run the benchmark using timeit. We run it one more time to remove the precompilation time of the first call
    times = timeit.repeat(solve, repeat=num_repeats+1, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times[1:]]  # List of times in nanoseconds

def dynamiqs_smesolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
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

N_list = np.floor(np.linspace(10, 800, 10)).astype(int) 

if not run_gpu:
    # Benchmark all cases
    benchmark_results = {
        "dynamiqs_mesolve": dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=100),
        "dynamiqs_mcsolve": dynamiqs_mcsolve(N, Δ, F, γ, nth, ntraj, num_repeats=10),
        "dynamiqs_smesolve": dynamiqs_smesolve(N, Δ, F, γ, nth, ntraj, num_repeats=5),
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
    for N in tqdm(N_list):
        num_repeats = 40
        if N > 50:
            num_repeats = 20
        if N > 100:
            num_repeats = 10
        if N > 200:
            num_repeats = 2
        dynamiqs_mesolve_N_cpu.append(dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=num_repeats))

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
    for N in tqdm(N_list):
        num_repeats = 100
        if N > 50:
            num_repeats = 40
        if N > 100:
            num_repeats = 10
        if N > 200:
            num_repeats = 2
        dynamiqs_mesolve_N_gpu.append(dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=num_repeats))

    benchmark_results_N = {
        "dynamiqs_mesolve_N_gpu": dynamiqs_mesolve_N_gpu,
    }

    # %%

    print("Saving results to JSON...")
    # Save results to JSON

    with open("src/benchmarks/python/dynamiqs_benchmark_results_N_gpu.json", "w") as f:
        json.dump(benchmark_results_N, f, indent=4)