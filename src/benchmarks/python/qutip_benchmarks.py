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
stoc_dt = 1e-3 # In case we run with a fixed timestep algorithm

N_list_cpu = range(2, 13)
N_list_gpu = range(2, 12)

# %%

def local_op(op, i, N):
    ops = [qutip.qeye(2) for _ in range(N)]
    ops[i] = op
    return qutip.tensor(ops)

def generate_system(N, system_type):
    if system_type == "ising":
        Hz = hz * sum(local_op(qutip.sigmaz(), i, N) for i in range(N))
        # Hxx = Jx * sum(local_op(qutip.sigmax(), i, N) * local_op(qutip.sigmax(), j, N) for i in range(N) for j in range(i+1, N))
        Hxx = Jx * sum(local_op(qutip.sigmax(), i, N) * local_op(qutip.sigmax(), i+1, N) for i in range(N-1))
        H = Hz + Hxx

        # c_ops = [np.sqrt(γ) * local_op(qutip.sigmam(), i, N) for i in range(N)]
        if isinstance(H.data, qutip_jax.jaxdia.JaxDia):
            c_ops = [jnp.sqrt(γ) * local_op(qutip.sigmam(), i, N) for i in range(N)]
        else:
            c_ops = [np.sqrt(γ) * local_op(qutip.sigmam(), i, N) for i in range(N)]

        e_ops = [local_op(qutip.sigmaz(), N-1, N)]

        return H, c_ops, e_ops
    elif system_type == "nho":
        a = qutip.destroy(N)
        H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
        c_ops = [np.sqrt(γ * (1 + nth)) * a, np.sqrt(γ * nth) * a.dag()]

        e_ops = [a.dag() * a]

        return H, c_ops, e_ops

def initial_state(N, system_type):
    if system_type == "ising":
        return qutip.tensor([qutip.basis(2, 0) for _ in range(N)])
    elif system_type == "nho":
        return qutip.fock(N, 0)

def qutip_mesolve(N, system_type, num_repeats=100):
    """Benchmark qutip.mesolve using timeit for more accurate timing."""
    H, c_ops, e_ops = generate_system(N, system_type)

    tlist = np.linspace(0, 10, 100)
    ψ0 = initial_state(N, system_type)
    options = {"store_final_state": True}

    qutip.mesolve(H, ψ0, tlist[0:2], c_ops, e_ops=e_ops, options=options) # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.mesolve(H, ψ0, tlist, c_ops, e_ops=e_ops, options=options).expect

    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def qutip_mcsolve(N, system_type, ntraj, num_repeats=100):
    """Benchmark qutip.mcsolve using timeit for more accurate timing."""
    H, c_ops, e_ops = generate_system(N, system_type)

    tlist = np.linspace(0, 10, 100)
    ψ0 = initial_state(N, system_type)

    options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads, "store_final_state": True}

    qutip.mcsolve(
        H,
        ψ0,
        tlist,
        c_ops,
        e_ops = e_ops,
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
            e_ops = e_ops,
            ntraj = ntraj,
            options = options,
        ).expect
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def qutip_smesolve(N, ntraj, num_repeats=100):
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

def qutip_mesolve_gpu(N, system_type, num_repeats=10):
    """Benchmark qutip.mesolve using timeit for more accurate timing."""
    with jax.default_device(jax.devices("gpu")[0]):
        with qutip.CoreOptions(default_dtype="jaxdia"):
            H, c_ops, e_ops = generate_system(N, system_type)

            tlist = jnp.linspace(0, 10, 100)

            ψ0 = initial_state(N, system_type) # Dense vector

            options = {
                "normalize_output":False,
                "store_states":False,
                "store_final_state":True,
                "method":"diffrax",
            }

            # Define the statement to benchmark
            def solve():
                qutip.mesolve(H, ψ0, tlist, c_ops, e_ops=e_ops, options=options).expect

            solve() # Warm-up

            # Run the benchmark using timeit
            times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

        return [t * 1e9 for t in times]  # List of times in nanoseconds

# %%
    
if not run_gpu:
    # Benchmark all cases
    N = 50
    benchmark_results = {
        "qutip_mesolve": qutip_mesolve(N, "nho", num_repeats=100),
        "qutip_mcsolve": qutip_mcsolve(N, "nho", ntraj, num_repeats=10),
        "qutip_smesolve": qutip_smesolve(N, ntraj, num_repeats=5),
    }

    # %%

    print("Saving results to JSON...")

    # Save results to JSON
    with open("src/benchmarks/python/qutip_benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    qutip_mesolve_N_cpu = []
    for N in tqdm(N_list_cpu):
        num_repeats = 2
        if N > 6:
            num_repeats = 1
        
        qutip_mesolve_N_cpu.append(qutip_mesolve(N, "ising", num_repeats=num_repeats))

    benchmark_results_N = {
        "qutip_mesolve_N_cpu": qutip_mesolve_N_cpu,
    }

    # %%

    print("Saving results to JSON...")
    # Save results to JSON

    with open("src/benchmarks/python/qutip_benchmark_results_N_cpu.json", "w") as f:
        json.dump(benchmark_results_N, f, indent=4)
else:
    qutip_mesolve_N_gpu = [] # In this way it is safe if it fails due to lack of GPU memory
    for N in tqdm(N_list_gpu):
        num_repeats = 2
        if N > 6:
            num_repeats = 1

        qutip_mesolve_N_gpu.append(qutip_mesolve_gpu(N, "ising", num_repeats=num_repeats))

    benchmark_results_N = {
        "qutip_mesolve_N_gpu": qutip_mesolve_N_gpu,
    }

    # %%

    print("Saving results to JSON...")
    # Save results to JSON

    with open("src/benchmarks/python/qutip_benchmark_results_N_gpu.json", "w") as f:
        json.dump(benchmark_results_N, f, indent=4)
