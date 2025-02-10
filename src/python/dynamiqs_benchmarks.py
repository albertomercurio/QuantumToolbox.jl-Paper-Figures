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

    dynamiqs.mesolve(H, c_ops, ψ0, tlist, options = dynamiqs.Options(progress_meter = None)).states # Warm-up

    # Define the statement to benchmark
    def solve():
        dynamiqs.mesolve(H, c_ops, ψ0, tlist, options = dynamiqs.Options(progress_meter = None)).states
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

def dynamiqs_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=100):
    a = dynamiqs.destroy(N)
    H = Δ * a.dag() @ a - U/2 * a.dag() @ a.dag() @ a @ a + F * (a + a.dag())
    sc_ops = [jnp.sqrt(γ * (1 + nth)) * a]

    tlist = jnp.arange(0, 10, stoc_dt*20)
    ψ0 = dynamiqs.fock(N, 0)

    # define a certain number of PRNG key, one for each trajectory
    key = jax.random.PRNGKey(20)
    keys = jax.random.split(key, ntraj)

    solver = dynamiqs.solver.EulerMaruyama(dt=stoc_dt)

    sol_sse = dynamiqs.dssesolve(H, sc_ops, ψ0, tlist, keys, exp_ops=[a.dag() @ a], solver=solver, options=dynamiqs.Options(progress_meter=None)) # Warm-up
    sol_me = dynamiqs.mesolve(H, sc_ops, ψ0, tlist, exp_ops=[a.dag() @ a], options=dynamiqs.Options(progress_meter=None))
    # Test if the two methods give the same result up to sol tolerance
    expect_sse = jnp.sum(sol_sse.expects, axis=0)[0,:] / ntraj
    convergence_metric = jnp.sum(jnp.abs(expect_sse - sol_me.expects[0,:])) / len(tlist)
    print(f"ssesolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert jnp.allclose(expect_sse, sol_me.expects[0,:], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        dynamiqs.dssesolve(H, sc_ops, ψ0, tlist, keys, solver=solver, options=dynamiqs.Options(progress_meter=None)).states
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

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

    solver = dynamiqs.solver.EulerMaruyama(dt=stoc_dt)

    sol_sme = dynamiqs.dsmesolve(H, sc_ops, etas, ψ0, tlist, keys, exp_ops=[a.dag() @ a], solver=solver) # Warm-up
    sol_me = dynamiqs.mesolve(H, [sc_ops[0], sc_ops[1]], ψ0, tlist, exp_ops=[a.dag() @ a])
    # Test if the two methods give the same result up to sol tolerance
    expect_sme = jnp.sum(sol_sme.expects, axis=0)[0,:] / ntraj
    convergence_metric = jnp.sum(jnp.abs(expect_sme - sol_me.expects[0,:])) / len(tlist)
    print(f"smesolve convergenge check. {convergence_metric} should be smaller than 0.1")
    assert jnp.allclose(expect_sme, sol_me.expects[0,:], atol=1e-1 * len(tlist))

    # Define the statement to benchmark
    def solve():
        dynamiqs.dsmesolve(H, sc_ops, etas, ψ0, tlist, keys, solver=solver).states
    
    # Run the benchmark using timeit
    times = timeit.repeat(solve, repeat=num_repeats, number=1)  # number=1 ensures individual execution times

    return [t * 1e9 for t in times]  # List of times in nanoseconds

# %%

# Benchmark all cases
benchmark_results = {
    "dynamiqs_mesolve": dynamiqs_mesolve(N, Δ, F, γ, nth, num_repeats=100),
    # "dynamiqs_ssesolve": dynamiqs_ssesolve(N, Δ, F, γ, nth, ntraj, num_repeats=10), Not yet implemented
    "dynamiqs_smesolve": dynamiqs_smesolve(N, Δ, F, γ, nth, ntraj, num_repeats=10),
}

# %%

print("Saving results to JSON...")

# Save results to JSON
with open("src/python/dynamiqs_benchmark_results.json", "w") as f:
    json.dump(benchmark_results, f, indent=4)
# %%
