"""Black-Litterman model — combine market equilibrium with trading views."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization.

    Combines:
      1. Market equilibrium returns (implied by market cap weights)
      2. Your trading views (from ML signals, sentiment, factors)

    Result: optimal portfolio weights that blend market prior with views,
            weighted by your confidence in each view.
    """

    def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
        """
        risk_aversion: lambda — higher = more conservative (typical: 2-4)
        tau: uncertainty in prior (typical: 0.01-0.10)
        """
        self.risk_aversion = risk_aversion
        self.tau           = tau

    # ------------------------------------------------------------------
    def optimize(
        self,
        returns_df: pd.DataFrame,
        market_caps: dict[str, float] | None = None,
        views: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute BL optimal weights.

        views: list of dicts, each:
          {"symbols": ["TCS", "INFY"], "weights": [1, -1],
           "expected_return": 0.05, "confidence": 0.70}
          (positive weight = long, negative = short)

        Returns: dict of symbol → portfolio weight
        """
        symbols = list(returns_df.columns)
        n       = len(symbols)

        if n < 2:
            return {s: 1/n for s in symbols}

        # Covariance matrix
        cov = returns_df.cov().values * 252   # annualized

        # Market cap weights (equal weight if not provided)
        if market_caps:
            total = sum(market_caps.get(s, 1) for s in symbols)
            w_mkt = np.array([market_caps.get(s, 1) / total for s in symbols])
        else:
            w_mkt = np.ones(n) / n

        # Equilibrium returns: π = λ Σ w_mkt
        pi = self.risk_aversion * cov @ w_mkt

        if not views:
            # No views → use market equilibrium
            w_opt = self._mean_variance(pi, cov, self.risk_aversion)
            return dict(zip(symbols, w_opt))

        # Build views matrices
        P, Q, omega = self._build_views(symbols, views, cov)

        # BL posterior
        tau_cov    = self.tau * cov
        M          = np.linalg.inv(
            np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P
        )
        mu_bl      = M @ (
            np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(omega) @ Q
        )

        # Optimal weights
        w_opt = self._mean_variance(mu_bl, cov, self.risk_aversion)
        w_opt = np.clip(w_opt, 0.02, 0.40)
        w_opt = w_opt / w_opt.sum()

        return {s: round(float(w), 4) for s, w in zip(symbols, w_opt)}

    def optimize_with_signals(
        self,
        returns_df: pd.DataFrame,
        signal_scores: dict[str, float],
        confidence: float = 0.60,
    ) -> dict[str, float]:
        """
        Convenience: convert signal scores to BL views automatically.

        signal_scores: {symbol: composite_score} (positive = bullish)
        confidence: uniform confidence for all views [0, 1]
        """
        symbols = list(returns_df.columns)
        views   = []

        for sym, score in signal_scores.items():
            if sym not in symbols:
                continue
            if abs(score) < 20:   # only use strong signals
                continue
            # Expected alpha: 5% for score=50, 2.5% for score=25
            expected_ret = score / 50 * 0.05
            views.append({
                "symbols":         [sym],
                "weights":         [1.0],
                "expected_return": expected_ret,
                "confidence":      confidence,
            })

        return self.optimize(returns_df, views=views)

    def optimal_quantities(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        prices: dict[str, float],
        capital: float,
        signal_scores: dict[str, float] = None,
    ) -> dict[str, int]:
        """Return integer share quantities from BL weights."""
        df = returns_df[symbols].dropna(axis=1)
        actual_symbols = list(df.columns)

        weights = self.optimize_with_signals(df, signal_scores or {})

        result = {}
        for sym in actual_symbols:
            w     = weights.get(sym, 0)
            price = prices.get(sym, 0)
            if price > 0 and w > 0:
                result[sym] = max(1, int(capital * w / price))
        return result

    # ------------------------------------------------------------------
    def _build_views(self, symbols, views, cov):
        n = len(symbols)
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, view in enumerate(views):
            view_syms = view["symbols"]
            wts       = view["weights"]
            for sym, w in zip(view_syms, wts):
                if sym in symbols:
                    j      = symbols.index(sym)
                    P[i, j] = w
            Q[i] = view["expected_return"]
            # Omega: uncertainty inversely proportional to confidence
            conf = view.get("confidence", 0.5)
            p_row = P[i:i+1]
            omega_diag[i] = float(
                ((1 - conf) / max(conf, 0.01)) * (p_row @ cov @ p_row.T)[0, 0]
            )

        omega = np.diag(np.maximum(omega_diag, 1e-8))
        return P, Q, omega

    @staticmethod
    def _mean_variance(mu: np.ndarray, cov: np.ndarray,
                       risk_aversion: float) -> np.ndarray:
        """Analytical mean-variance optimal weights."""
        try:
            cov_inv = np.linalg.pinv(cov)
            w       = (1 / risk_aversion) * cov_inv @ mu
            w       = np.clip(w, 0, None)
            total   = w.sum()
            return w / total if total > 0 else np.ones(len(mu)) / len(mu)
        except Exception:
            return np.ones(len(mu)) / len(mu)
