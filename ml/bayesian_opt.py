"""Bayesian hyperparameter optimization using scikit-optimize / Optuna."""
from __future__ import annotations
import numpy as np
from utils.logger import logger


class BayesianStrategyOptimizer:
    """
    Bayesian optimization for strategy parameters.

    Better than random/grid search for small evaluation budgets
    (financial backtests are expensive — every evaluation costs time).

    Optimizes: min_composite_score, stop_loss_multiplier,
               target_multiplier, min_adx, ml_prob_threshold
    """

    PARAM_SPACE = {
        "min_composite_score": (15.0, 55.0),
        "stop_loss_multiplier": (0.7, 2.5),
        "target_multiplier":    (1.5, 4.0),
        "min_adx":              (15.0, 40.0),
        "ml_prob_threshold":    (0.50, 0.70),
        "kelly_fraction":       (0.10, 0.40),
    }

    def __init__(self, n_calls: int = 25, random_state: int = 42):
        self.n_calls      = n_calls
        self.random_state = random_state
        self.best_params: dict  = {}
        self.best_score:  float = -np.inf
        self._history:    list  = []

    # ------------------------------------------------------------------
    def optimize(self, objective_fn, param_space: dict = None) -> dict:
        """
        Run Bayesian optimization.

        objective_fn: function(params_dict) → scalar score (higher = better)
        param_space:  override default param space
        """
        space = param_space or self.PARAM_SPACE

        try:
            return self._optimize_skopt(objective_fn, space)
        except ImportError:
            pass

        try:
            return self._optimize_optuna(objective_fn, space)
        except ImportError:
            pass

        return self._optimize_random(objective_fn, space)

    def optimize_backtest(self, engine_class, enriched_dfs: dict,
                          metric: str = "sharpe") -> dict:
        """
        Optimize backtest engine parameters.

        metric: "sharpe" | "calmar" | "return" | "profit_factor"
        """
        def objective(params):
            try:
                engine = engine_class(**params)
                result = engine._run_with_dataframes(enriched_dfs)
                summary = result.get("summary", {})
                ret = summary.get("return_pct", 0)
                mdd = summary.get("max_drawdown_pct", 100)
                pf  = summary.get("profit_factor", 0)

                if metric == "sharpe":
                    return ret / max(mdd, 0.1)
                elif metric == "calmar":
                    return ret / max(mdd, 0.1)
                elif metric == "profit_factor":
                    return pf
                return ret
            except Exception:
                return -100.0

        return self.optimize(objective)

    # ------------------------------------------------------------------
    def _optimize_skopt(self, objective_fn, space: dict) -> dict:
        from skopt import gp_minimize
        from skopt.space import Real

        dims   = [Real(lo, hi, name=k) for k, (lo, hi) in space.items()]
        names  = list(space.keys())
        scores = []

        def wrapped(values):
            params = dict(zip(names, values))
            score  = objective_fn(params)
            scores.append((params, score))
            return -score   # minimize negative

        result = gp_minimize(wrapped, dims, n_calls=self.n_calls,
                             random_state=self.random_state, verbose=False)

        best_params = dict(zip(names, result.x))
        best_score  = -result.fun
        self._update_best(best_params, best_score)
        logger.info(f"BayesOpt (skopt): best_score={best_score:.4f} | {best_params}")
        return best_params

    def _optimize_optuna(self, objective_fn, space: dict) -> dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def trial_fn(trial):
            params = {k: trial.suggest_float(k, lo, hi)
                      for k, (lo, hi) in space.items()}
            return objective_fn(params)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(trial_fn, n_trials=self.n_calls, show_progress_bar=False)
        best_params = study.best_params
        self._update_best(best_params, study.best_value)
        logger.info(f"BayesOpt (optuna): best={study.best_value:.4f} | {best_params}")
        return best_params

    def _optimize_random(self, objective_fn, space: dict) -> dict:
        """Fallback: random search."""
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_calls):
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}
            score  = objective_fn(params)
            self._update_best(params, score)
            self._history.append((params, score))

        logger.info(f"BayesOpt (random): best={self.best_score:.4f}")
        return self.best_params

    def _update_best(self, params: dict, score: float):
        if score > self.best_score:
            self.best_score  = score
            self.best_params = {k: round(v, 4) for k, v in params.items()}

    def get_history(self) -> list[dict]:
        return [{"params": p, "score": round(s, 4)} for p, s in self._history]

    def pareto_front(self, objectives: list[str]) -> list[dict]:
        """Return Pareto-optimal configurations (multi-objective)."""
        if not self._history:
            return []
        # Simplified: return top-5 by score
        sorted_h = sorted(self._history, key=lambda x: x[1], reverse=True)
        return [{"params": p, "score": s} for p, s in sorted_h[:5]]
