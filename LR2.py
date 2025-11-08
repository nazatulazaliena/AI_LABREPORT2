import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, List, Tuple

# -------------------- Problem Definition --------------------
@dataclass
# Define the structure of a Genetic Algorithm problem.
class GAProblem:
    name: str
    chromosome_type: str  # 'bit' or 'real'
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]

# Define the Target50 problem, where the goal is to have exactly 50 ones in an 80-bit string.
def make_target50(dim: int = 80) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 50)  # maximum 80 when ones = 50

    return GAProblem(
        name=f"Target50 ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )

# -------------------- GA Operators --------------------
# Initialize population with random bitstrings.
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)

# Select the best individual among k randomly chosen candidates.
def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)

# Perform one-point crossover between two parents.
def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

# Perform bit mutation: each bit flips with probability mut_rate.
def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

# Evaluate fitness for the entire population.
def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int | None,
    stream_live: bool = True,
):

    # Main loop of Genetic Algorithm.
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Live chart update
        if stream_live:
            df = pd.DataFrame({
                "Best": history_best,
                "Average": history_avg,
                "Worst": history_worst
            })
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.2f}**")

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))

        # Create next generation
        next_pop = []
        while len(next_pop) < pop_size - E:
            # Select parents
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Form new population
        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, problem)

    # Final results
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {"best": best, "best_fitness": best_fit, "history": df}


# -------------------- Streamlit Interface --------------------
st.set_page_config(page_title="Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (GAs) — Evolving a Bit Pattern Population")
st.caption("This GA evolves an 80-bit pattern that achieves maximum fitness when 50 bits are ones.")

# Fixed problem setup
problem = make_target50(dim=80)
pop_size = 300
generations = 50
crossover_rate = 0.9
mutation_rate = 0.01
tournament_k = 3
elitism = 2
seed = 42

if st.button("Run Genetic Algorithm", type="primary"):
    result = run_ga(
        problem=problem,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_k=tournament_k,
        elitism=elitism,
        seed=seed,
        stream_live=True,
    )

    # Display results
    st.subheader("Fitness Progress")
    st.line_chart(result["history"])

    st.subheader("Best Individual Found")
    best = result["best"]
    best_fit = result["best_fitness"]
    num_ones = int(np.sum(best))

    st.write(f"**Best Fitness:** {best_fit:.2f}")
    st.write(f"**Number of Ones:** {num_ones} / 80")

    bitstring = ''.join(map(str, best.astype(int).tolist()))
    st.code(bitstring, language="text")

    if num_ones == 50:
        st.success("🎯 Optimal pattern achieved — 50 ones (fitness = 80).")
    else:
        st.info("Close to optimal — consider tuning parameters for better convergence.")
