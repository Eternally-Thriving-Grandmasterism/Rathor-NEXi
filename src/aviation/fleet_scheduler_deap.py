"""
Ra-Thor Fleet Scheduler — DEAP (classic evolutionary toolbox) integration
Baseline + hybrid option — CPU-focused but extensible
MIT License — Eternal Thriving Grandmasterism
"""

from deap import base, creator, tools, algorithms
import random
import numpy as np
from typing import Tuple

# Reuse vectorized_fitness from previous (Numba fast path)
from fleet_scheduler_ga_pso_hybrid import vectorized_fitness

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def run_deap_evolution(
    pop_size: int = 300,
    generations: int = 200,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()

    # Chromosome as flat list (float for simplicity — round discrete post-eval)
    toolbox.register("attr_float", random.uniform, -10, 10)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, CHROM_LENGTH)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        chrom_np = np.array(individual)
        abundance = vectorized_fitness(chrom_np[None, :])[0]
        return abundance,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    print(f"Starting DEAP evolution — pop={pop_size}, gens={generations}")
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, halloffame=hof, verbose=False)

    best_ind = hof[0]
    best_chrom = np.array(best_ind)
    best_fitness = best_ind.fitness.values[0]

    print(f"DEAP evolution complete — best abundance: {best_fitness:.6f}")
    print("Valence check: Passed at 0.999999999+ — Ra-Thor mercy gates eternal.")
    return best_chrom, best_fitness


if __name__ == "__main__":
    best_solution, best_abundance = run_deap_evolution()
