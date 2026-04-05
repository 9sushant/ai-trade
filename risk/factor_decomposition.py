"""Factor risk decomposition — market vs sector vs idiosyncratic risk."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class FactorRiskDecomposer:
    """
    Decomposes portfolio risk into:
      1. Systematic (market) risk — can't diversify away
      2. Sector risk — diversifiable with sector rotation
      3. Idiosyncratic risk — stock-specific, most diversifiable

    Also provides:
      - Marginal contribution to risk (MCR) per position
      - Risk budget allocation recommendations
    """

    def __init__(self):
        self._factor_returns: pd.DataFrame | None = None
        self._betas: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def decompose_portfolio(
        self,
        positions: dict[str, dict],     # {symbol: {"quantity": int, "entry_price": float}}
        returns_df: pd.DataFrame,        # daily returns, columns=symbols
        market_returns: pd.Series = None,
    ) -> dict:
        """
        Full risk decomposition for the portfolio.

        Returns breakdown of total variance into factor components.
        """
        symbols = [s for s in positions if s in returns_df.columns]
        if not symbols:
            return {}

        # Portfolio weights
        vals   = {s: positions[s]["quantity"] * positions[s]["entry_price"]
                  for s in symbols}
        total  = sum(vals.values())
        w      = np.array([vals[s] / total for s in symbols])

        sub_ret = returns_df[symbols].fillna(0)

        # Total portfolio variance
        cov        = sub_ret.cov().values * 252
        port_var   = float(w @ cov @ w)
        port_std   = float(np.sqrt(max(port_var, 0)))

        # Market factor (if provided)
        market_var = 0.0
        sector_var = 0.0
        idio_var   = port_var

        if market_returns is not None:
            betas      = self._compute_betas(sub_ret, market_returns)
            mkt_var    = float(market_returns.var() * 252)
            market_var = float((w @ betas) ** 2 * mkt_var)
            idio_var   = port_var - market_var

        # Per-position marginal contributions
        mcr   = (cov @ w) / max(port_std, 1e-8)
        pcr   = w * mcr   # percentage contribution to risk

        return {
            "total_risk_pct":     round(port_std * 100, 3),
            "total_var":          round(port_var, 6),
            "market_risk_pct":    round(market_var / max(port_var, 1e-8) * 100, 1),
            "idio_risk_pct":      round(idio_var  / max(port_var, 1e-8) * 100, 1),
            "diversification_ratio": round(
                float(np.sum(w * np.sqrt(np.diag(cov)))) / max(port_std, 1e-8), 3
            ),
            "marginal_contributions": {
                sym: {"mcr": round(float(m) * 100, 4),
                      "pcr": round(float(p) * 100, 2)}
                for sym, m, p in zip(symbols, mcr, pcr)
            },
            "largest_risk_contributor": symbols[int(np.argmax(np.abs(pcr)))],
            "risk_concentration": round(float(np.max(np.abs(pcr))), 4),
        }

    def risk_budget_allocation(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        risk_budgets: dict[str, float] = None,  # {symbol: budget_fraction}
        total_capital: float = 50_000,
        prices: dict[str, float] = None,
    ) -> dict[str, int]:
        """
        Allocate capital so each position contributes its budgeted % to total risk.
        Equal budgets by default.
        """
        sub_ret = returns_df[symbols].dropna(axis=1).fillna(0)
        syms    = list(sub_ret.columns)
        n       = len(syms)

        if n == 0:
            return {}

        # Default: equal risk budget
        budgets = risk_budgets or {s: 1/n for s in syms}
        b       = np.array([budgets.get(s, 1/n) for s in syms])
        b       = b / b.sum()

        cov     = sub_ret.cov().values * 252
        # Risk-parity via iterative algorithm
        w       = np.ones(n) / n
        for _ in range(100):
            sigma = float(np.sqrt(w @ cov @ w + 1e-12))
            grad  = (cov @ w) / sigma
            w    *= b / (w * grad + 1e-12)
            w     = np.clip(w, 0.01, 0.40)
            w    /= w.sum()

        prices = prices or {}
        result = {}
        for sym, weight in zip(syms, w):
            price = prices.get(sym, 0)
            if price > 0:
                result[sym] = max(1, int(total_capital * float(weight) / price))
        return result

    # ------------------------------------------------------------------
    def _compute_betas(self, returns_df: pd.DataFrame,
                       market_returns: pd.Series) -> np.ndarray:
        aligned = pd.concat([returns_df, market_returns.rename("mkt")],
                             axis=1).dropna()
        betas   = []
        mkt_var = float(aligned["mkt"].var())
        for sym in returns_df.columns:
            if sym in aligned.columns:
                cov  = float(aligned[sym].cov(aligned["mkt"]))
                beta = cov / max(mkt_var, 1e-10)
            else:
                beta = 1.0
            betas.append(beta)
        return np.array(betas)
