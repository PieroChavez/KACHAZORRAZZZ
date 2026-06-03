"""Tests for Genetic Optimizer (Modo Experto, ultra-rápido).
GA loop tests use mock fitness (no backtest). Fitness function tests
use minimal data to verify integration.
"""
import pytest
import numpy as np
import pandas as pd
from src.optimization.genetic_optimizer import GeneticOptimizer, WalkForwardOptimizer, Individual
from src.optimization.fitness import evaluate_fitness
from src.optimization.parameter_space import BASE_SPACE, EXPERT_SPACE, decode_chromosome, encode_params


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """100 filas — ~8 horas de datos 5min.
    No genera trades (random walk sin SMC), pero verifica
    que el pipeline backtest+fitness corre sin errores."""
    np.random.seed(42)
    n = 100
    ts = pd.date_range("2026-05-01", periods=n, freq="5min")
    close = 3000 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + abs(np.random.randn(n)) * 1.5
    low = close - abs(np.random.randn(n)) * 1.5
    open_ = pd.Series(close).shift(1).fillna(close[0])
    vol = np.random.randint(100, 1000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    }, index=ts)


@pytest.fixture
def fast_fitness():
    """Mock fitness que retorna score determinista sin backtest.
    O(1) por evaluación: ~10μs vs ~100ms del backtest real."""
    def fn(params):
        sharpe = params.get("min_reward_risk_ratio", 1.5) * 0.1
        wr = params.get("risk_per_trade", 0.02) * 5.0
        fitness = sharpe * 100 + wr * 10 + np.random.randn() * 0.5
        return round(float(fitness), 4), {
            "trades": 5,
            "total_return_pct": round(fitness, 2),
            "win_rate": 0.5,
            "profit_factor": 1.2,
        }
    return fn


# ─────────────────────────────────────────────
# PARAMETER SPACE
# ─────────────────────────────────────────────

class TestParameterSpace:
    def test_decode_chromosome(self):
        chrom = {"risk_per_trade": 0.5, "atr_multiplier_sl": 0.3}
        decoded = decode_chromosome(chrom, BASE_SPACE)
        assert 0.003 <= decoded["risk_per_trade"] <= 0.03
        assert 0.8 <= decoded["atr_multiplier_sl"] <= 3.5

    def test_encode_params(self):
        params = {"risk_per_trade": 0.015, "atr_multiplier_sl": 2.0}
        encoded = encode_params(params, BASE_SPACE)
        assert 0 <= encoded["risk_per_trade"] <= 1
        assert 0 <= encoded["atr_multiplier_sl"] <= 1

    def test_round_trip(self):
        original = {"risk_per_trade": 0.01, "atr_multiplier_sl": 1.5, "min_score_to_trade": 60}
        encoded = encode_params(original, BASE_SPACE)
        decoded = decode_chromosome(encoded, BASE_SPACE)
        assert decoded["risk_per_trade"] == pytest.approx(0.01, rel=0.01)
        assert decoded["min_score_to_trade"] == 60


# ─────────────────────────────────────────────
# GENETIC OPTIMIZER — CORE LOOP (mock fitness)
# ─────────────────────────────────────────────

class TestGeneticOptimizer:
    def test_init(self):
        opt = GeneticOptimizer(expert_mode=False)
        assert opt.cfg["population_size"] == 50
        assert len(opt.space) > 0

    def test_random_chromosome(self):
        opt = GeneticOptimizer(expert_mode=False)
        chrom = opt._random_chromosome()
        assert len(chrom) == len(opt.space)
        for v in chrom.values():
            assert 0 <= v <= 1

    def test_tournament_select(self):
        opt = GeneticOptimizer(expert_mode=False)
        pop = [Individual({"x": i / 10}, fitness=i) for i in range(10)]
        selected = opt._tournament_select(pop)
        assert selected.fitness >= 0

    def test_crossover(self):
        opt = GeneticOptimizer(expert_mode=False)
        p1 = {"a": 0.1, "b": 0.9}
        p2 = {"a": 0.8, "b": 0.2}
        c1, c2 = opt._crossover(p1, p2)
        assert len(c1) == 2
        assert len(c2) == 2

    def test_mutation(self):
        opt = GeneticOptimizer(expert_mode=False)
        chrom = {"a": 0.5, "b": 0.5}
        mutated = opt._mutate(chrom)
        assert 0 <= mutated["a"] <= 1
        assert 0 <= mutated["b"] <= 1

    def test_population_diversity(self):
        opt = GeneticOptimizer(expert_mode=False)
        pop = [Individual({"x": 0.1}, fitness=1), Individual({"x": 0.9}, fitness=2)]
        opt.population = pop
        div = opt._population_diversity()
        assert div > 0.1

    def test_evaluate_population(self, fast_fitness):
        opt = GeneticOptimizer(expert_mode=False, config={"population_size": 10, "generations": 2})
        opt.set_fitness_fn(fast_fitness)
        pop = [Individual(opt._random_chromosome()) for _ in range(5)]
        opt._evaluate_population(pop)
        for ind in pop:
            assert isinstance(ind.fitness, float)

    def test_run_end_to_end(self, fast_fitness):
        """GA loop con mock fitness O(1) — sin backtest."""
        opt = GeneticOptimizer(
            expert_mode=False,
            config={"population_size": 10, "generations": 3, "elite_count": 2},
        )
        opt.set_fitness_fn(fast_fitness)
        result = opt.run()
        assert "best_params" in result
        assert "best_fitness" in result
        assert isinstance(result["best_fitness"], float)
        assert len(result["hall_of_fame"]) > 0

    def test_hall_of_fame(self, fast_fitness):
        opt = GeneticOptimizer(
            expert_mode=False,
            config={"population_size": 8, "generations": 2, "elite_count": 2},
        )
        opt.set_fitness_fn(fast_fitness)
        result = opt.run()
        assert len(result["hall_of_fame"]) > 0
        for entry in result["hall_of_fame"]:
            assert "params" in entry
            assert "fitness" in entry

    def test_early_stop(self):
        opt = GeneticOptimizer(
            expert_mode=False,
            config={"population_size": 8, "generations": 50, "elite_count": 2, "early_stop_gens": 2},
        )
        def fit_fn(params):
            return 0.5, {"trades": 3}
        opt.set_fitness_fn(fit_fn)
        result = opt.run()
        assert result["generations"] < 10

    def test_expert_mode_init(self):
        opt = GeneticOptimizer(expert_mode=True)
        assert len(opt.space) > len(BASE_SPACE)

    def test_save_load_checkpoint(self, tmp_path):
        opt = GeneticOptimizer(expert_mode=False)
        opt.generation = 5
        opt.best_fitness = 10.0
        opt.hall_of_fame = [Individual({"x": 0.5}, fitness=10)]
        cp = tmp_path / "checkpoint.json"
        opt.save_checkpoint(cp)
        assert cp.exists()
        opt2 = GeneticOptimizer(expert_mode=False)
        opt2.load_checkpoint(cp)
        assert opt2.generation == 5
        assert opt2.best_fitness == 10.0


# ─────────────────────────────────────────────
# FITNESS FUNCTION (real backtest, data chica)
# ─────────────────────────────────────────────

class TestFitness:
    def test_evaluate_fitness_with_data(self, sample_data):
        params = {
            "risk_per_trade": 0.01,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.0,
            "min_score_to_trade": 20,
        }
        fitness, meta = evaluate_fitness(params, sample_data)
        assert isinstance(fitness, float)
        assert "trades" in meta

    def test_evaluate_fitness_insufficient_data(self):
        df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2], "volume": [100, 200]})
        fitness, meta = evaluate_fitness({}, df)
        assert fitness < 0
        assert meta.get("trades", 0) == 0 or "error" in meta

    def test_evaluate_fitness_empty_params(self, sample_data):
        fitness, meta = evaluate_fitness({}, sample_data)
        assert isinstance(fitness, float)


# ─────────────────────────────────────────────
# WALK-FORWARD OPTIMIZER
# ─────────────────────────────────────────────

class TestWalkForwardOptimizer:
    def test_insufficient_data(self):
        opt = GeneticOptimizer(expert_mode=False)
        wf = WalkForwardOptimizer(opt, n_windows=2, window_size=1000, step_size=200)
        df = pd.DataFrame({"open": [1] * 100, "high": [2] * 100, "low": [0] * 100, "close": [1] * 100, "volume": [100] * 100})
        result = wf.run(df, lambda p, d=df: (0, {}))
        assert "error" in result
