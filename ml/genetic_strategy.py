"""Genetic algorithm for automatic strategy rule discovery."""
from __future__ import annotations
import numpy as np
import random
from copy import deepcopy
from utils.logger import logger

# Gene pool: tunable strategy parameters
_GENE_SPACE = {
    "min_composite_score": (10.0, 60.0),
    "stop_loss_multiplier": (0.5, 3.0),
    "target_multiplier":    (1.2, 5.0),
    "min_adx":              (10.0, 40.0),
    "ml_prob_threshold":    (0.48, 0.75),
    "kelly_fraction":       (0.05, 0.40),
    "trail_atr_multiplier": (0.2, 1.5),
    "max_positions":        (2, 10),
    "min_rr_ratio":         (1.0, 3.0),
}


class Individual:
    """One candidate strategy (chromosome)."""

    def __init__(self, genes: dict = None):
        self.genes   = genes or self._random_genes()
        self.fitness = -np.inf
        self.metrics: dict = {}

    @staticmethod
    def _random_genes(rng=None) -> dict:
        rng = rng or random
        return {
            k: rng.uniform(lo, hi) if isinstance(lo, float) else rng.randint(int(lo), int(hi))
            for k, (lo, hi) in _GENE_SPACE.items()
        }

    def to_engine_params(self) -> dict:
        """Convert genes to BacktestEngine keyword arguments."""
        g = self.genes
        return {
            "min_composite_score":  float(g["min_composite_score"]),
            "stop_loss_multiplier": float(g["stop_loss_multiplier"]),
            "target_multiplier":    float(g["target_multiplier"]),
            "min_adx":              float(g["min_adx"]),
            "ml_prob_threshold":    float(g["ml_prob_threshold"]),
            "kelly_fraction":       float(g["kelly_fraction"]),
            "trail_atr_multiplier": float(g["trail_atr_multiplier"]),
            "max_positions":        int(g["max_positions"]),
            "min_rr_ratio":         float(g["min_rr_ratio"]),
        }

    def __repr__(self):
        return f"Individual(fitness={self.fitness:.3f}, genes={self.genes})"


class GeneticStrategyOptimizer:
    """
    Genetic Algorithm for strategy parameter discovery.

    Mimics natural selection:
      1. Generate random population of strategies
      2. Evaluate fitness (backtest performance)
      3. Select best performers
      4. Crossover (combine two parents)
      5. Mutate (small random changes)
      6. Repeat for N generations

    Advantages over grid/Bayesian search:
      - Explores large non-convex parameter spaces
      - Finds unexpected parameter combinations
      - Naturally avoids local optima
    """

    def __init__(
        self,
        population_size: int = 20,
        n_generations:   int = 10,
        elite_fraction:  float = 0.20,
        mutation_rate:   float = 0.15,
        crossover_rate:  float = 0.70,
        fitness_metric:  str = "sharpe",   # sharpe | calmar | return | pf
        random_state:    int = 42,
    ):
        self.pop_size       = population_size
        self.n_gen          = n_generations
        self.elite_frac     = elite_fraction
        self.mutation_rate  = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_metric = fitness_metric
        random.seed(random_state)
        np.random.seed(random_state)

        self._population: list[Individual] = []
        self._history:    list[dict] = []
        self.best: Individual | None = None

    # ------------------------------------------------------------------
    def evolve(self, fitness_fn) -> Individual:
        """
        Run genetic algorithm.

        fitness_fn: callable(params_dict) → float (higher = better)
        Returns best individual found.
        """
        # Initialize population
        self._population = [Individual() for _ in range(self.pop_size)]
        logger.info(f"GA: starting evolution, pop={self.pop_size}, gen={self.n_gen}")

        for gen in range(self.n_gen):
            # Evaluate fitness
            for ind in self._population:
                if ind.fitness == -np.inf:
                    try:
                        ind.fitness = float(fitness_fn(ind.to_engine_params()))
                    except Exception:
                        ind.fitness = -100.0

            # Sort by fitness
            self._population.sort(key=lambda x: x.fitness, reverse=True)
            best_gen = self._population[0]

            if self.best is None or best_gen.fitness > self.best.fitness:
                self.best = deepcopy(best_gen)

            self._history.append({
                "generation":    gen + 1,
                "best_fitness":  round(best_gen.fitness, 4),
                "mean_fitness":  round(np.mean([i.fitness for i in self._population]), 4),
                "best_genes":    best_gen.genes,
            })
            logger.info(f"GA gen {gen+1}/{self.n_gen}: best={best_gen.fitness:.4f}")

            if gen == self.n_gen - 1:
                break

            # Selection + reproduction
            n_elite    = max(1, int(self.pop_size * self.elite_frac))
            elites     = self._population[:n_elite]
            new_pop    = list(elites)

            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament_select(), self._tournament_select()
                if random.random() < self.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = deepcopy(p1)
                child = self._mutate(child)
                new_pop.append(child)

            self._population = new_pop

        logger.info(f"GA: done. Best fitness={self.best.fitness:.4f}")
        return self.best

    def evolve_backtest(self, engine_class, enriched_dfs: dict) -> Individual:
        """Evolve using backtest engine as fitness function."""
        def fitness_fn(params):
            try:
                engine = engine_class(**params)
                result = engine._run_with_dataframes(enriched_dfs)
                s      = result.get("summary", {})
                ret    = s.get("return_pct",       0)
                mdd    = s.get("max_drawdown_pct",  100)
                pf     = s.get("profit_factor",     0)
                wr     = s.get("win_rate",          0)
                trades = s.get("total_trades",      0)

                if trades < 10:
                    return -50.0   # penalize too few trades

                if self.fitness_metric == "sharpe":
                    return ret / max(mdd, 0.1)
                elif self.fitness_metric == "calmar":
                    return ret / max(mdd, 0.1)
                elif self.fitness_metric == "pf":
                    return float(pf)
                elif self.fitness_metric == "composite":
                    return (ret / max(mdd, 0.1)) * (wr / 100) * min(pf, 3)
                return ret
            except Exception:
                return -100.0

        return self.evolve(fitness_fn)

    def get_history(self) -> list[dict]:
        return self._history

    def pareto_front(self) -> list[Individual]:
        """Return Pareto-optimal individuals (approximate)."""
        return sorted(self._population, key=lambda x: x.fitness, reverse=True)[:5]

    # ------------------------------------------------------------------
    def _tournament_select(self, k: int = 3) -> Individual:
        """Tournament selection: pick best of k random individuals."""
        candidates = random.sample(self._population, min(k, len(self._population)))
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """Uniform crossover: each gene randomly inherited from p1 or p2."""
        child_genes = {}
        for k in _GENE_SPACE:
            child_genes[k] = p1.genes[k] if random.random() < 0.5 else p2.genes[k]
        return Individual(child_genes)

    def _mutate(self, ind: Individual) -> Individual:
        """Gaussian mutation: perturb random genes."""
        ind = deepcopy(ind)
        for k, (lo, hi) in _GENE_SPACE.items():
            if random.random() < self.mutation_rate:
                rng   = hi - lo
                delta = random.gauss(0, rng * 0.1)
                if isinstance(lo, float):
                    ind.genes[k] = float(np.clip(ind.genes[k] + delta, lo, hi))
                else:
                    ind.genes[k] = int(np.clip(round(ind.genes[k] + delta), lo, hi))
        ind.fitness = -np.inf   # needs re-evaluation
        return ind
