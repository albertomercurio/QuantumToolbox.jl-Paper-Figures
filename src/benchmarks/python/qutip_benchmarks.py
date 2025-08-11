# %%
import numpy as np
import jax
import jax.numpy as jnp
import qutip
import qutip_jax
import timeit
import json
from tqdm import tqdm
import os

num_threads = int(os.getenv("JULIA_NUM_THREADS", int(os.cpu_count() / 2)))

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
stoc_dt = 1e-3 # In case we run with a fixed timestep algorithm

# %%

def qutip_mesolve(N, Δ, F, γ, nth, num_repeats=100):
    """Benchmark qutip.mesolve using timeit for more accurate timing."""
    a = qutip.destroy(N)
    H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
    c_ops = [np.sqrt(γ * (1 + nth)) * a, np.sqrt(γ * nth) * a.dag()]

    tlist = np.linspace(0, 10, 100)
    ψ0 = qutip.fock(N, 0)
    options = {"store_final_state": True}

    qutip.mesolve(H, ψ0, tlist[0:2], c_ops, e_ops=[a.dag() * a], options=options) # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.mesolve(H, ψ0, tlist, c_ops, e_ops=[a.dag() * a], options=options).expect

    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def qutip_mcsolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    """Benchmark qutip.mcsolve using timeit for more accurate timing."""
    a = qutip.destroy(N)
    H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
    c_ops = [np.sqrt(γ * (1 + nth)) * a, np.sqrt(γ * nth) * a.dag()]

    tlist = np.linspace(0, 10, 100)
    ψ0 = qutip.fock(N, 0)

    options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads, "store_final_state": True}

    qutip.mcsolve(
        H,
        ψ0,
        tlist,
        c_ops,
        e_ops = [a.dag() * a],
        ntraj = ntraj,
        options = options,
    ) # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.mcsolve(
            H,
            ψ0,
            tlist,
            c_ops,
            e_ops = [a.dag() * a],
            ntraj = ntraj,
            options = options,
        ).expect
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def qutip_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    """Benchmark qutip.ssesolve using timeit for more accurate timing."""
    a = qutip.destroy(N)
    H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
    sc_ops = [np.sqrt(γ * (1 + nth)) * a]

    tlist = np.arange(0, 10, stoc_dt*20)
    ψ0 = qutip.fock(N, 0)

    options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads, "store_final_state": True}

    sol_sse = qutip.ssesolve(
        H,
        ψ0,
        tlist,
        sc_ops,
        e_ops=[a.dag() * a],
        ntraj=ntraj,
        options=options,
    ) # Warm-up
    sol_me = qutip.mesolve(H, ψ0, tlist, sc_ops, e_ops=[a.dag() * a])
    # Test if the two methods give the same result up to sol tolerance
    convergence_metric = np.sum(np.abs(sol_sse.expect[0] - sol_me.expect[0])) / len(tlist)
    print(f"ssesolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert np.allclose(sol_sse.expect[0], sol_me.expect[0], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        qutip.ssesolve(
            H,
            ψ0,
            tlist,
            sc_ops,
            e_ops=[a.dag() * a],
            ntraj=ntraj,
            options=options,
        ).expect
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def qutip_smesolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    """Benchmark qutip.ssesolve using timeit for more accurate timing."""
    a = qutip.destroy(N)
    H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
    c_ops = [np.sqrt(γ * nth) * a.dag()]
    sc_ops = [np.sqrt(γ * (1 + nth)) * a]

    tlist = np.arange(0, 10, stoc_dt*20)
    ψ0 = qutip.fock(N, 0)

    options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads, "store_final_state": True}

    sol_sme = qutip.smesolve(
        H,
        ψ0,
        tlist,
        c_ops,
        sc_ops,
        e_ops=[a.dag() * a],
        ntraj=ntraj,
        options=options,
    ) # Warm-up
    sol_me = qutip.mesolve(H, ψ0, tlist, [c_ops[0], sc_ops[0]], e_ops=[a.dag() * a])
    # Test if the two methods give the same result up to sol tolerance
    convergence_metric = np.sum(np.abs(sol_sme.expect[0] - sol_me.expect[0])) / len(tlist)
    print(f"smesolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert np.allclose(sol_sme.expect[0], sol_me.expect[0], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        qutip.smesolve(
            H,
            ψ0,
            tlist,
            c_ops,
            sc_ops,
            e_ops=[a.dag() * a],
            ntraj=ntraj,
            options=options,
        ).expect
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List

def qutip_mesolve_gpu(N, Δ, F, γ, nth, num_repeats=100):
    """Benchmark qutip.mesolve using timeit for more accurate timing."""
    with jax.default_device(jax.devices("gpu")[0]):
        with qutip.CoreOptions(default_dtype="jaxdia"):
            a = qutip.destroy(N)
        
            # We need to convert H to jaxdia again
            H = (Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())).to("jaxdia")
            c_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]

            tlist = jnp.linspace(0, 10, 100)

            ψ0 = qutip.fock(N, 0).to("jax") # Dense vector

            options = {
                "normalize_output":False,
                "store_states":False,
                "store_final_state":True,
                "method":"diffrax",
            }

            # Define the statement to benchmark
            def solve():
                qutip.mesolve(H, ψ0, tlist, c_ops, e_ops=[a.dag() * a], options=options).expect

            solve() # Warm-up


            # Run the benchmark using timeit
            times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

        return [t * 1e9 for t in times]  # List of times in nanoseconds

# %%

# Benchmark all cases
benchmark_results = {
    "qutip_mesolve": qutip_mesolve(N, Δ, F, γ, nth, num_repeats=100),
    "qutip_mcsolve": qutip_mcsolve(N, Δ, F, γ, nth, ntraj, num_repeats=20),
    "qutip_ssesolve": qutip_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=20),
    "qutip_smesolve": qutip_smesolve(N, Δ, F, γ, nth, ntraj, num_repeats=10),
}

# %%

print("Saving results to JSON...")

# Save results to JSON
with open("src/benchmarks/python/qutip_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=4)

# %% [markdown]
# Varying the Hilbert space dimension $N$

# %%

N_list = np.floor(np.linspace(10, 800, 10)).astype(int)

qutip_mesolve_N_cpu = []
for N in tqdm(N_list):
    num_repeats = 100
    if N > 50:
        num_repeats = 40
    if N > 100:
        num_repeats = 10
    if N > 200:
        num_repeats = 2
    try:
        qutip_mesolve_N_cpu.append(qutip_mesolve(N, Δ, F, γ, nth, num_repeats=num_repeats))
    except Exception as e:
        print(f"Failed for N={N} with error: {e}")
        break

qutip_mesolve_N_gpu = [] # In this way it is safe if it fails due to lack of GPU memory
for N in tqdm(N_list):
    num_repeats = 40
    if N > 50:
        num_repeats = 20
    if N > 100:
        num_repeats = 10
    if N > 200:
        num_repeats = 2
    try:
        qutip_mesolve_N_gpu.append(qutip_mesolve_gpu(N, Δ, F, γ, nth, num_repeats=num_repeats))
    except Exception as e:
        print(f"Failed for N={N} with error: {e}")
        break

benchmark_results_N = {
    "qutip_mesolve_N_cpu": qutip_mesolve_N_cpu,
    "qutip_mesolve_N_gpu": qutip_mesolve_N_gpu,
}

# %%

print("Saving results to JSON...")
# Save results to JSON

with open("src/benchmarks/python/qutip_benchmark_results_N.json", "w") as f:
    json.dump(benchmark_results_N, f, indent=4)

# %%
