"""Genetic Algorithm Optimizer (Mejora 5, Modo Experto)
Evolves parameter combinations using historical data to find
optimal configuration for current market conditions.
"""
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.optimization.parameter_space import (
    BASE_SPACE, EXPERT_SPACE, FITNESS_WEIGHTS,
    decode_chromosome, ParamDef,
)

logger = logging.getLogger(__name__)

GEN_DEFAULTS = {
    "population_size": 50,
    "generations": 8,
    "tournament_size": 3,
    "crossover_rate": 0.8,
    "mutation_rate": 0.20,
    "mutation_strength": 0.15,
    "elite_count": 2,
    "early_stop_gens": 4,
    "islands": 1,
    "eval_timeout": 120,
}


class Individual:
    __slots__ = ("chromosome", "fitness", "age", "metadata")

    def __init__(self, chromosome: Dict, fitness: float = -float("inf")):
        self.chromosome = chromosome
        self.fitness = fitness
        self.age = 0
        self.metadata: Dict = {}


class GeneticOptimizer:
    def __init__(self, param_space: Dict = None, config: Dict = None,
                 expert_mode: bool = True):
        self.space = param_space or (EXPERT_SPACE if expert_mode else BASE_SPACE)
        self.cfg = {**GEN_DEFAULTS, **(config or {})}
        self.expert_mode = expert_mode
        self.population: List[Individual] = []
        self.hall_of_fame: List[Individual] = []
        self.generation = 0
        self.best_fitness = -float("inf")
        self._fitness_cache: Dict[str, Tuple[float, Dict]] = {}
        self._islands: List[List[Individual]] = []
        self._convergence_history: List[float] = []
        self._fitness_fn: Optional[Callable] = None

    def set_fitness_fn(self, fn: Callable[[Dict], Tuple[float, Dict]]):
        self._fitness_fn = fn

    def _random_chromosome(self) -> Dict:
        return {name: random.random() for name in self.space}

    def _init_population(self):
        pop_size = self.cfg["population_size"]
        self.population = [Individual(self._random_chromosome()) for _ in range(pop_size)]

        if self.expert_mode and self.cfg.get("islands", 3) > 1:
            n_islands = min(self.cfg["islands"], pop_size // 5)
            island_size = pop_size // n_islands
            self._islands = [
                [Individual(self._random_chromosome()) for _ in range(island_size)]
                for _ in range(n_islands)
            ]
            remainder = pop_size - island_size * n_islands
            self._islands[-1].extend(
                [Individual(self._random_chromosome()) for _ in range(remainder)]
            )
        else:
            self._islands = [self.population]

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        k = min(self.cfg["tournament_size"], len(pop))
        candidates = random.sample(pop, k)
        return max(candidates, key=lambda ind: ind.fitness)

    def _crossover(self, p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
        if random.random() > self.cfg["crossover_rate"]:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        c1, c2 = {}, {}
        for key in p1:
            if random.random() < 0.5:
                c1[key], c2[key] = p1[key], p2[key]
            else:
                c1[key], c2[key] = p2[key], p1[key]
        if self.expert_mode and random.random() < 0.3:
            alpha = random.uniform(0.3, 0.7)
            for key in p1:
                c1[key] = p1[key] * alpha + p2[key] * (1 - alpha)
                c2[key] = p2[key] * alpha + p1[key] * (1 - alpha)
        return c1, c2

    def _mutate(self, chrom: Dict) -> Dict:
        mutated = copy.deepcopy(chrom)
        mr = self.cfg["mutation_rate"]
        ms = self.cfg["mutation_strength"]

        if self.expert_mode:
            diversity = self._population_diversity()
            mr = mr * (1.0 + max(0, 0.5 - diversity) * 0.5)
            ms = ms * (0.5 + diversity)

        for key in mutated:
            if random.random() < mr:
                delta = random.gauss(0, ms)
                mutated[key] = max(0.0, min(1.0, mutated[key] + delta))
        return mutated

    def _population_diversity(self) -> float:
        if len(self.population) < 2:
            return 1.0
        keys = list(self.population[0].chromosome.keys())
        variances = []
        for k in keys:
            vals = [ind.chromosome[k] for ind in self.population]
            variances.append(np.var(vals))
        return float(np.mean(variances)) if variances else 0.0

    def _evaluate_population(self, pop: List[Individual],
                              progress_cb: Callable = None):
        to_eval = [(i, ind) for i, ind in enumerate(pop)
                   if ind.fitness == -float("inf")]
        if not to_eval:
            return

        total = len(to_eval)
        last_pct = 0
        done_idx = 0

        for idx, ind in to_eval:
            key = json.dumps(ind.chromosome, sort_keys=True)
            cached = self._fitness_cache.get(key)
            if cached is not None:
                ind.fitness, ind.metadata = cached
                done_idx += 1
                continue

            params = decode_chromosome(ind.chromosome, self.space)
            t0 = time.time()
            try:
                fitness, meta = self._fitness_fn(params)
            except Exception as e:
                logger.warning(f"Fitness eval [{idx+1}/{total}] failed: {e}")
                fitness, meta = -1000.0, {"error": str(e)}
            elapsed = time.time() - t0

            ind.fitness = fitness
            ind.metadata = meta
            self._fitness_cache[key] = (fitness, meta)
            done_idx += 1

            pct = (done_idx * 100) // total
            if pct // 10 > last_pct // 10:
                logger.info(
                    f"Optimización: {done_idx}/{total} ({pct}%) "
                    f"completadas | último fitness={fitness:.4f} "
                    f"({elapsed:.1f}s)"
                )
                last_pct = pct

            if progress_cb:
                progress_cb(idx, total)

    def _next_generation(self) -> List[Individual]:
        new_pop = []
        elite_count = min(self.cfg["elite_count"], len(self.population))

        elites = sorted(self.population, key=lambda i: i.fitness, reverse=True)[:elite_count]
        for e in elites:
            new_pop.append(copy.deepcopy(e))

        while len(new_pop) < len(self.population):
            p1 = self._tournament_select(self.population)
            p2 = self._tournament_select(self.population)
            c1, c2 = self._crossover(p1.chromosome, p2.chromosome)
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)
            new_pop.append(Individual(c1))
            if len(new_pop) < len(self.population):
                new_pop.append(Individual(c2))

        return new_pop[:len(self.population)]

    def _migrate_islands(self):
        if len(self._islands) < 2:
            return
        for i in range(len(self._islands)):
            j = (i + 1) % len(self._islands)
            if len(self._islands[i]) < 2 or len(self._islands[j]) < 2:
                continue
            emigrant = max(self._islands[i], key=lambda ind: ind.fitness)
            immigrant = random.choice(self._islands[j])
            idx_i = self._islands[i].index(emigrant)
            idx_j = self._islands[j].index(immigrant)
            self._islands[i][idx_i], self._islands[j][idx_j] = (
                copy.deepcopy(immigrant), copy.deepcopy(emigrant)
            )

    def run(self, fitness_fn: Callable[[Dict], Tuple[float, Dict]] = None,
            progress_cb: Callable = None) -> Dict:
        if fitness_fn:
            self.set_fitness_fn(fitness_fn)
        if self._fitness_fn is None:
            raise ValueError("Fitness function not set")

        self._init_population()
        self.generation = 0
        self.best_fitness = -float("inf")
        stall_count = 0

        for gen in range(self.cfg["generations"]):
            t0 = time.time()

            for island in self._islands:
                self._evaluate_population(island, progress_cb)

            if len(self._islands) > 1:
                all_inds = [ind for island in self._islands for ind in island]
            else:
                all_inds = self.population

            best = max(all_inds, key=lambda ind: ind.fitness)
            avg_fit = np.mean([ind.fitness for ind in all_inds])

            if best.fitness > self.best_fitness:
                self.best_fitness = best.fitness
                stall_count = 0
                self.hall_of_fame.append(copy.deepcopy(best))
                if len(self.hall_of_fame) > 20:
                    self.hall_of_fame = self.hall_of_fame[-20:]
            else:
                stall_count += 1

            self._convergence_history.append(avg_fit)

            iso_elapsed = time.time() - t0
            logger.info(
                f"Gen {gen + 1}/{self.cfg['generations']}: "
                f"best={best.fitness:.4f} avg={avg_fit:.4f} "
                f"diversity={self._population_diversity():.4f} "
                f"stall={stall_count} [{iso_elapsed:.1f}s]"
            )

            if stall_count >= self.cfg.get("early_stop_gens", 8):
                logger.info(f"Early stop at gen {gen + 1}")
                break

            if self.expert_mode and len(self._islands) > 1 and gen % 3 == 0:
                self._migrate_islands()
                self.population = [ind for island in self._islands for ind in island]
            else:
                self.population = self._next_generation()

            if self.expert_mode and len(self._islands) > 1:
                self._rebuild_islands()

        all_inds = [ind for island in self._islands for ind in island] if len(self._islands) > 1 else self.population
        best = max(all_inds, key=lambda ind: ind.fitness)
        params = decode_chromosome(best.chromosome, self.space)
        return {
            "best_params": params,
            "best_fitness": best.fitness,
            "metadata": best.metadata,
            "generations": self.generation,
            "hall_of_fame": [
                {
                    "params": decode_chromosome(ind.chromosome, self.space),
                    "fitness": ind.fitness,
                    "metadata": ind.metadata,
                }
                for ind in self.hall_of_fame[-5:]
            ],
            "convergence": self._convergence_history,
        }

    def _rebuild_islands(self):
        if len(self._islands) < 2:
            return
        combined = [ind for island in self._islands for ind in self.population]
        random.shuffle(combined)
        n = len(combined)
        n_islands = len(self._islands)
        size = n // n_islands
        self._islands = [combined[i * size:(i + 1) * size] for i in range(n_islands)]
        if n % n_islands:
            self._islands[-1].extend(combined[n_islands * size:])

    def warm_start(self, previous_best: Dict):
        """Continue optimization from previous best population"""
        if not self.hall_of_fame:
            return
        seed = decode_chromosome(previous_best, self.space)
        encoded = {}
        for name in self.space:
            pdef = self.space[name]
            encoded[name] = (seed[name] - pdef.lo) / (pdef.hi - pdef.lo) if pdef.hi > pdef.lo else 0.5
        self.population[0] = Individual(encoded, self.best_fitness)
        n_seed = max(3, len(self.population) // 10)
        for i in range(1, n_seed):
            noisy = {k: max(0, min(1, v + random.gauss(0, 0.05)))
                     for k, v in encoded.items()}
            self.population[i] = Individual(noisy)

    def save_checkpoint(self, path: Path):
        data = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "config": self.cfg,
            "hall_of_fame": [
                {"chromosome": ind.chromosome, "fitness": ind.fitness}
                for ind in self.hall_of_fame
            ],
            "convergence": self._convergence_history,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path):
        data = json.loads(path.read_text())
        self.generation = data["generation"]
        self.best_fitness = data["best_fitness"]
        self.cfg.update(data["config"])
        self.hall_of_fame = [
            Individual(d["chromosome"], d["fitness"])
            for d in data["hall_of_fame"]
        ]
        self._convergence_history = data["convergence"]
        logger.info(f"Checkpoint loaded: {path}")


class WalkForwardOptimizer:
    """Walk-forward optimization: train on window, validate on next window"""

    def __init__(self, optimizer: GeneticOptimizer, n_windows: int = 4,
                 window_size: int = 1000, step_size: int = 200):
        self.optimizer = optimizer
        self.n_windows = n_windows
        self.window_size = window_size
        self.step_size = step_size
        self.results: List[Dict] = []

    def run(self, df: pd.DataFrame, fitness_fn: Callable) -> Dict:
        n = len(df)
        if n < self.window_size + self.step_size:
            return {"error": "insufficient_data"}

        all_best = []
        for w in range(self.n_windows):
            train_end = n - (self.n_windows - w) * self.step_size
            train_start = max(0, train_end - self.window_size)
            test_end = min(n, train_end + self.step_size)

            if train_start >= train_end or train_end >= test_end:
                continue

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[train_end:test_end]

            def train_fn(params):
                return fitness_fn(params, train_df)

            result = self.optimizer.run(fitness_fn=train_fn)
            test_fitness, test_meta = fitness_fn(result["best_params"], test_df)

            self.results.append({
                "window": w,
                "train_range": f"{train_start}-{train_end}",
                "test_range": f"{train_end}-{test_end}",
                "train_fitness": result["best_fitness"],
                "test_fitness": test_fitness,
                "params": result["best_params"],
                "test_meta": test_meta,
            })
            all_best.append(result["best_params"])

        if not all_best:
            return {"error": "no_windows_completed"}

        avg_params = {}
        for key in all_best[0]:
            vals = [p[key] for p in all_best]
            avg_params[key] = float(np.mean(vals))

        avg_test_fit = np.mean([r["test_fitness"] for r in self.results])
        return {
            "avg_params": avg_params,
            "avg_test_fitness": float(avg_test_fit),
            "windows": self.results,
            "n_windows": len(self.results),
        }
