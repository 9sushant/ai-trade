"""Monte Carlo simulation of equity curves for risk-of-ruin analysis."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class MonteCarloSimulator:
    """
    Runs N Monte Carlo simulations of the equity curve by resampling trades.

    Outputs:
      - Probability of drawdown exceeding threshold
      - Risk of ruin (equity drops below min_capital)
      - 5th/50th/95th percentile equity curves
      - Expected Calmar ratio distribution
    """

    def __init__(self, n_simulations: int = 5000, seed: int = 42):
        self.n_sims = n_simulations
        self.rng    = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def run(
        self,
        trades: list[dict],
        initial_capital: float,
        min_capital: float = None,
        max_drawdown_threshold: float = 0.20,
    ) -> dict:
        """
        Simulate `n_simulations` random orderings of trades.

        Returns comprehensive risk metrics.
        """
        if not trades:
            return {"error": "No trades to simulate"}

        min_capital = min_capital or initial_capital * 0.50
        pnls = np.array([t["pnl"] for t in trades], dtype=np.float64)

        final_equities = []
        max_drawdowns  = []
        ruin_count     = 0
        all_curves     = []

        for _ in range(self.n_sims):
            sampled = self.rng.choice(pnls, size=len(pnls), replace=True)
            equity  = initial_capital
            peak    = equity
            mdd     = 0.0
            curve   = [equity]

            for pnl in sampled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > mdd:
                    mdd = dd
                curve.append(equity)
                if equity <= min_capital:
                    ruin_count += 1
                    break

            final_equities.append(equity)
            max_drawdowns.append(mdd)
            if len(all_curves) < 200:   # store first 200 for charting
                all_curves.append(curve)

        fe  = np.array(final_equities)
        mdd = np.array(max_drawdowns)

        return {
            "n_simulations":          self.n_sims,
            "initial_capital":        initial_capital,
            "final_equity": {
                "p5":    round(float(np.percentile(fe, 5)), 2),
                "p25":   round(float(np.percentile(fe, 25)), 2),
                "p50":   round(float(np.percentile(fe, 50)), 2),
                "p75":   round(float(np.percentile(fe, 75)), 2),
                "p95":   round(float(np.percentile(fe, 95)), 2),
                "mean":  round(float(np.mean(fe)), 2),
            },
            "max_drawdown": {
                "p50":   round(float(np.percentile(mdd, 50)) * 100, 2),
                "p75":   round(float(np.percentile(mdd, 75)) * 100, 2),
                "p95":   round(float(np.percentile(mdd, 95)) * 100, 2),
                "mean":  round(float(np.mean(mdd)) * 100, 2),
            },
            "prob_loss":              round(float(np.mean(fe < initial_capital)) * 100, 1),
            "prob_mdd_exceeds_threshold":
                round(float(np.mean(mdd > max_drawdown_threshold)) * 100, 1),
            "risk_of_ruin_pct":       round(ruin_count / self.n_sims * 100, 2),
            "prob_return_gt_10pct":
                round(float(np.mean(fe > initial_capital * 1.10)) * 100, 1),
            "prob_return_gt_20pct":
                round(float(np.mean(fe > initial_capital * 1.20)) * 100, 1),
            "sample_curves":          all_curves[:10],   # 10 sample paths
        }

    def run_from_returns(
        self,
        daily_returns: np.ndarray,
        initial_capital: float,
        n_days: int = 252,
        min_capital: float = None,
    ) -> dict:
        """Simulate from daily return distribution (GBM-style)."""
        mu  = float(np.mean(daily_returns))
        std = float(np.std(daily_returns))
        min_capital = min_capital or initial_capital * 0.5

        final_equities = []
        ruin_count = 0

        for _ in range(self.n_sims):
            rets   = self.rng.normal(mu, std, n_days)
            equity = initial_capital
            ruined = False
            for r in rets:
                equity *= (1 + r)
                if equity <= min_capital:
                    ruin_count += 1
                    ruined = True
                    break
            final_equities.append(equity)

        fe = np.array(final_equities)
        return {
            "n_simulations":    self.n_sims,
            "n_days":           n_days,
            "daily_mu_pct":     round(mu * 100, 4),
            "daily_std_pct":    round(std * 100, 4),
            "final_equity_p50": round(float(np.percentile(fe, 50)), 2),
            "final_equity_p95": round(float(np.percentile(fe, 95)), 2),
            "final_equity_p5":  round(float(np.percentile(fe, 5)), 2),
            "risk_of_ruin_pct": round(ruin_count / self.n_sims * 100, 2),
            "prob_profit":      round(float(np.mean(fe > initial_capital)) * 100, 1),
        }

    def optimal_f(self, pnls: list[float], initial_capital: float) -> float:
        """
        Estimate optimal fraction of capital to risk per trade (Ralph Vince method).
        Returns f in [0.01, 0.25].
        """
        if not pnls:
            return 0.01
        max_loss = abs(min(pnls)) if min(pnls) < 0 else 1
        twrs = []
        for f in np.linspace(0.01, 0.25, 50):
            twr = 1.0
            for p in pnls:
                twr *= (1 + f * p / max_loss)
                if twr <= 0:
                    twr = 0
                    break
            twrs.append((f, twr))
        best_f, _ = max(twrs, key=lambda x: x[1])
        return round(float(best_f), 3)
