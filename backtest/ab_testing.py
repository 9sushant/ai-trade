"""A/B testing framework — statistically compare strategy variants."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger


class StrategyABTester:
    """
    Statistical A/B testing framework for strategy variants.

    Tests whether variant B is statistically significantly better than A.

    Metrics compared:
      - Returns (t-test)
      - Sharpe ratio (bootstrap)
      - Win rate (proportion test)
      - Maximum drawdown (permutation test)
    """

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence   = confidence_level
        self.n_bootstrap  = n_bootstrap
        self.alpha        = 1 - confidence_level
        self._experiments: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def register_experiment(self, name: str, description: str = ""):
        """Register a new A/B experiment."""
        self._experiments[name] = {
            "description": description,
            "control":     [],   # strategy A results
            "variant":     [],   # strategy B results
            "created_at":  datetime.now().isoformat(),
        }

    def record_control(self, experiment: str, result: dict):
        """Record a backtest result for strategy A (control)."""
        self._experiments.setdefault(experiment, {"control": [], "variant": []})
        self._experiments[experiment]["control"].append(result)

    def record_variant(self, experiment: str, result: dict):
        """Record a backtest result for strategy B (variant)."""
        self._experiments.setdefault(experiment, {"control": [], "variant": []})
        self._experiments[experiment]["variant"].append(result)

    # ------------------------------------------------------------------
    def compare(self, experiment: str) -> dict:
        """Run full statistical comparison of control vs variant."""
        exp = self._experiments.get(experiment)
        if not exp:
            return {"error": f"Experiment '{experiment}' not found"}

        controls = exp["control"]
        variants = exp["variant"]

        if not controls or not variants:
            return {"error": "Need results for both control and variant"}

        results = {
            "experiment":    experiment,
            "n_control":     len(controls),
            "n_variant":     len(variants),
            "tests":         {},
            "recommendation": "",
            "winner":        "INCONCLUSIVE",
        }

        # 1. Return comparison
        ctrl_ret  = [r.get("return_pct", 0) for r in controls]
        var_ret   = [r.get("return_pct", 0) for r in variants]
        ret_test  = self._t_test(ctrl_ret, var_ret, "return_pct")
        results["tests"]["returns"] = ret_test

        # 2. Sharpe ratio comparison
        ctrl_sharpe = [self._sharpe(r) for r in controls]
        var_sharpe  = [self._sharpe(r) for r in variants]
        sh_test     = self._bootstrap_test(ctrl_sharpe, var_sharpe, "sharpe_ratio")
        results["tests"]["sharpe"] = sh_test

        # 3. Win rate comparison
        ctrl_wr = [r.get("win_rate", 0) for r in controls]
        var_wr  = [r.get("win_rate", 0) for r in variants]
        wr_test = self._proportion_test(ctrl_wr, var_wr, "win_rate")
        results["tests"]["win_rate"] = wr_test

        # 4. Drawdown comparison (lower is better)
        ctrl_mdd = [r.get("max_drawdown_pct", 0) for r in controls]
        var_mdd  = [r.get("max_drawdown_pct", 0) for r in variants]
        mdd_test = self._t_test(var_mdd, ctrl_mdd, "max_drawdown_pct",
                                 lower_is_better_a=True)
        results["tests"]["drawdown"] = mdd_test

        # 5. Profit factor comparison
        ctrl_pf  = [r.get("profit_factor", 1) for r in controls]
        var_pf   = [r.get("profit_factor", 1) for r in variants]
        pf_test  = self._t_test(ctrl_pf, var_pf, "profit_factor")
        results["tests"]["profit_factor"] = pf_test

        # Summary
        n_significant = sum(1 for t in results["tests"].values()
                            if t.get("significant") and t.get("winner") == "VARIANT")
        n_tests = len(results["tests"])

        if n_significant >= 3:
            results["winner"]         = "VARIANT"
            results["recommendation"] = (
                f"ADOPT VARIANT — {n_significant}/{n_tests} tests significantly favor it."
            )
        elif n_significant == 0:
            results["winner"]         = "CONTROL"
            results["recommendation"] = "KEEP CONTROL — variant shows no improvement."
        else:
            results["recommendation"] = (
                f"INCONCLUSIVE — only {n_significant}/{n_tests} tests favor variant. "
                f"Run more experiments."
            )

        # Summary stats
        results["summary"] = {
            "control": {
                "mean_return":    round(np.mean(ctrl_ret),    2),
                "mean_sharpe":    round(np.mean(ctrl_sharpe), 3),
                "mean_win_rate":  round(np.mean(ctrl_wr),     1),
                "mean_drawdown":  round(np.mean(ctrl_mdd),    2),
            },
            "variant": {
                "mean_return":    round(np.mean(var_ret),     2),
                "mean_sharpe":    round(np.mean(var_sharpe),  3),
                "mean_win_rate":  round(np.mean(var_wr),      1),
                "mean_drawdown":  round(np.mean(var_mdd),     2),
            },
        }

        logger.info(f"A/B Test '{experiment}': {results['recommendation']}")
        return results

    def compare_engines(self, engine_a_results: dict, engine_b_results: dict,
                        name: str = "auto") -> dict:
        """Quick compare — pass two backtest result dicts."""
        exp = name
        self.register_experiment(exp, "Automated engine comparison")
        self.record_control(exp, engine_a_results.get("summary", engine_a_results))
        self.record_variant(exp, engine_b_results.get("summary", engine_b_results))
        return self.compare(exp)

    # ------------------------------------------------------------------
    def _t_test(self, a: list[float], b: list[float], metric: str,
                lower_is_better_a: bool = False) -> dict:
        """Welch's t-test."""
        from scipy import stats
        a, b = np.array(a), np.array(b)
        if len(a) < 2 or len(b) < 2:
            return {"metric": metric, "significant": False, "p_value": 1.0}

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        sig = p_value < self.alpha
        if lower_is_better_a:
            winner = "VARIANT" if np.mean(b) < np.mean(a) and sig else "CONTROL"
        else:
            winner = "VARIANT" if np.mean(b) > np.mean(a) and sig else "CONTROL"

        return {
            "metric":      metric,
            "mean_a":      round(float(np.mean(a)), 3),
            "mean_b":      round(float(np.mean(b)), 3),
            "t_stat":      round(float(t_stat), 3),
            "p_value":     round(float(p_value), 4),
            "significant": sig,
            "winner":      winner if sig else "INCONCLUSIVE",
        }

    def _bootstrap_test(self, a: list[float], b: list[float], metric: str) -> dict:
        """Bootstrap hypothesis test."""
        a, b  = np.array(a), np.array(b)
        obs   = np.mean(b) - np.mean(a)
        rng   = np.random.default_rng(42)
        diffs = []
        for _ in range(self.n_bootstrap):
            sa = rng.choice(a, len(a), replace=True)
            sb = rng.choice(b, len(b), replace=True)
            diffs.append(np.mean(sb) - np.mean(sa))

        p_value = float(np.mean(np.array(diffs) <= 0))
        sig     = p_value < self.alpha
        return {
            "metric":      metric,
            "mean_a":      round(float(np.mean(a)), 3),
            "mean_b":      round(float(np.mean(b)), 3),
            "observed_diff": round(float(obs), 3),
            "p_value":     round(p_value, 4),
            "significant": sig,
            "winner":      "VARIANT" if sig and obs > 0 else "CONTROL",
        }

    def _proportion_test(self, a: list[float], b: list[float], metric: str) -> dict:
        """Test difference in proportions (win rate)."""
        from scipy import stats
        a, b = np.array(a) / 100, np.array(b) / 100   # convert % to fraction
        pa, pb = np.mean(a), np.mean(b)
        na, nb = len(a), len(b)
        p_pool = (pa * na + pb * nb) / (na + nb)
        se     = np.sqrt(p_pool * (1 - p_pool) * (1/na + 1/nb))
        z      = (pb - pa) / max(se, 1e-8)
        p_val  = 2 * (1 - stats.norm.cdf(abs(z)))
        sig    = p_val < self.alpha
        return {
            "metric":      metric,
            "mean_a":      round(pa * 100, 1),
            "mean_b":      round(pb * 100, 1),
            "z_stat":      round(float(z), 3),
            "p_value":     round(float(p_val), 4),
            "significant": sig,
            "winner":      "VARIANT" if sig and pb > pa else "CONTROL",
        }

    @staticmethod
    def _sharpe(result: dict) -> float:
        ret = result.get("return_pct", 0)
        mdd = result.get("max_drawdown_pct", 1)
        return ret / max(mdd, 0.1)
