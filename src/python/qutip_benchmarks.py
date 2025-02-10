# %%
import numpy as np
import qutip
import timeit
import json
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
stoc_dt = 1e-3

# %%

def qutip_mesolve(N, Δ, F, γ, nth, num_repeats=100):
    """Benchmark qutip.mesolve using timeit for more accurate timing."""
    a = qutip.destroy(N)
    H = Δ * a.dag() * a - U/2 * a.dag()**2 * a**2 + F * (a + a.dag())
    c_ops = [np.sqrt(γ * (1 + nth)) * a, np.sqrt(γ * nth) * a.dag()]

    tlist = np.linspace(0, 10, 100)
    ψ0 = qutip.fock(N, 0)

    qutip.mesolve(H, ψ0, tlist, c_ops).states[1] # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.mesolve(H, ψ0, tlist, c_ops).states[1]

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

    qutip.mcsolve(
        H,
        ψ0,
        tlist,
        c_ops,
        ntraj = ntraj,
        options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads},
    ).states[1] # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.mcsolve(
            H,
            ψ0,
            tlist,
            c_ops,
            ntraj = ntraj,
            options = {"progress_bar": False, "map": "parallel", "num_cpus": num_threads},
        ).states[1]
    
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

    qutip.ssesolve(
        H,
        ψ0,
        tlist,
        sc_ops,
        ntraj=ntraj,
        options={"progress_bar": False, 
                     "map": "parallel", 
                     "num_cpus": num_threads,
                     "method": "euler",
                     "dt": stoc_dt,},
    ).states[1] # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.ssesolve(
            H,
            ψ0,
            tlist,
            sc_ops,
            ntraj=ntraj,
            options={"progress_bar": False, 
                     "map": "parallel", 
                     "num_cpus": num_threads,
                     "method": "euler",
                     "dt": stoc_dt,},
        ).states[1]
    
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

    qutip.smesolve(
        H,
        ψ0,
        tlist,
        c_ops,
        sc_ops,
        ntraj=ntraj,
        options={"progress_bar": False, 
                     "map": "parallel", 
                     "num_cpus": num_threads,
                     "method": "euler",
                     "dt": stoc_dt,},
    ).states[1] # Warm-up

    # Define the statement to benchmark
    def solve():
        qutip.smesolve(
            H,
            ψ0,
            tlist,
            c_ops,
            sc_ops,
            ntraj=ntraj,
            options={"progress_bar": False, 
                     "map": "parallel", 
                     "num_cpus": num_threads,
                     "method": "euler",
                     "dt": stoc_dt,},
        ).states[1]
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List
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
with open("src/python/qutip_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=4)
# %%
