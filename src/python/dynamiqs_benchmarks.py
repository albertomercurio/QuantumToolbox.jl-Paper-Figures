# %%
import numpy as np
import jax
import jax.numpy as jnp
import dynamiqs
import timeit
import json

dynamiqs.set_device("cpu")
dynamiqs.set_precision("double") # Set the same precision as the others

# %% [markdown]

# Parameters:

# %%
N = 50
Δ = 0.1
F = 2
γ = 1
nth = 0.8
ntraj = 100
sse_dt = 1e-3

# %%

def dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=100):
    """Benchmark dynamiqs.mesolve using timeit for more accurate timing."""
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a + F * (a + a.dag())
    c_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]

    tlist = jnp.linspace(0, 10, 100)
    ψ0 = dynamiqs.fock(N, 0)

    dynamiqs.mesolve(H, c_ops, ψ0, tlist, options = dynamiqs.Options(progress_meter = None)).states # Warm-up

    # Define the statement to benchmark
    def solve():
        dynamiqs.mesolve(H, c_ops, ψ0, tlist, options = dynamiqs.Options(progress_meter = None)).states
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def dynamiqs_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a + F * (a + a.dag())
    sc_ops = [jnp.sqrt(γ * (1 + nth)) * a, jnp.sqrt(γ * nth) * a.dag()]

    tlist = jnp.linspace(0, 10, 100)
    ψ0 = dynamiqs.fock(N, 0)

    # define a certain number of PRNG key, one for each trajectory
    key = jax.random.PRNGKey(20)
    keys = jax.random.split(key, ntraj)

    solver = dynamiqs.solver.EulerMaruyama(dt=sse_dt)

    dynamiqs.dssesolve(H, sc_ops, ψ0, tlist, keys, solver=solver, options=dynamiqs.Options(progress_meter=None)).states # Warm-up

    # Define the statement to benchmark
    def solve():
        dynamiqs.dssesolve(H, sc_ops, ψ0, tlist, keys, solver=solver, options=dynamiqs.Options(progress_meter=None)).states
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds


# %%

# Benchmark all cases
benchmark_results = {
    "dynamiqs_mesolve": dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=100),
    # "dynamiqs_ssesolve": dynamiqs_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=10), Not yet implemented
}

# %%

# Save results to JSON
with open("dynamiqs_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=4)
# %%
