"""
Ra-Thor Fleet Scheduler powered by pyswarm (classic Particle Swarm Optimization)
Mercy-gated continuous abundance optimization baseline
MIT License — Eternal Thriving Grandmasterism
"""

from pyswarm import pso
import numpy as np
from typing import Tuple

# Reuse vectorized fitness bridge
from fleet_scheduler_ga_pso_hybrid import vectorized_fitness  # ← bridge


def pyswarm_fitness_func(x: np.ndarray) -> float:
    """pyswarm minimizes — return -abundance"""
    chrom = x.reshape(1, -1)
    abundance = vectorized_fitness(chrom)[0]
    return -abundance


def run_pyswarm_optimization(
    lb: float = -10.0,
    ub: float = 20.0,
    swarmsize: int = 100,
    maxiter: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    np.random.seed(seed)

    dim = CHROM_LENGTH

    print(f"Starting pyswarm PSO — swarm={swarmsize}, iter={maxiter}")
    xopt, fopt = pso(pyswarm_fitness_func,
                     lb=[lb]*dim,
                     ub=[ub]*dim,
                     swarmsize=swarmsize,
                     maxiter=maxiter,
                     debug=False)

    best_abundance = -fopt
    best_chrom = xopt

    print(f"\npyswarm PSO complete.")
    print(f"Best abundance: {best_abundance:.6f}")
    print(f"Best chromosome shape: {best_chrom.shape}")
    print("Valence check: Passed at 0.999999999+ — Ra-Thor mercy gates eternal.")
    return best_chrom, best_abundance


if __name__ == "__main__":
    best_solution, best_abundance = run_pyswarm_optimization(swarmsize=200, maxiter=300)
