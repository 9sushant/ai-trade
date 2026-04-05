"""Conditional VaR (CVaR / Expected Shortfall) risk calculator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class CVaRCalculator:
    """
    Computes Conditional Value-at-Risk (Expected Shortfall).

    CVaR = average loss in the worst (1-confidence)% of scenarios.
    More conservative than VaR — captures tail risk properly.
    """

    def __init__(self, confidence: float = 0.95, lookback_days: int = 252):
        self.confidence   = confidence
        self.lookback     = lookback_days

    # ------------------------------------------------------------------
    def compute_cvar(self, df: pd.DataFrame, capital: float) -> float:
        """
        Historical simulation CVaR in rupees.
        Returns the expected loss (positive number) in the worst scenarios.
        """
        if df is None or len(df) < 30:
            return capital * 0.05   # default 5% of capital

        returns = df["close"].pct_change().dropna().iloc[-self.lookback:]
        if len(returns) < 20:
            return capital * 0.05

        # Sort returns ascending (worst first)
        sorted_ret = np.sort(returns.values)
        cutoff_idx = int(len(sorted_ret) * (1 - self.confidence))
        tail       = sorted_ret[:cutoff_idx] if cutoff_idx > 0 else sorted_ret[:1]
        cvar_pct   = float(np.mean(tail))   # negative number

        return round(abs(cvar_pct) * capital, 2)

    def cvar_adjusted_quantity(
        self, df: pd.DataFrame, capital: float,
        entry_price: float, max_loss_pct: float = 0.02,
    ) -> int:
        """
        Maximum quantity such that CVaR loss ≤ max_loss_pct of capital.
        More conservative than VaR-based sizing.
        """
        cvar_per_unit = self.compute_cvar(df, entry_price)   # CVaR on 1 share
        max_loss      = capital * max_loss_pct
        if cvar_per_unit <= 0:
            return 10  # fallback
        qty = int(max_loss / cvar_per_unit)
        return max(1, qty)

    def portfolio_cvar(
        self, positions: dict[str, dict],
        returns_df: pd.DataFrame,
        capital: float,
        confidence: float = None,
    ) -> float:
        """
        Portfolio CVaR using correlated returns.
        positions: {symbol: {"quantity": int, "entry_price": float}}
        returns_df: DataFrame of daily returns per symbol
        """
        conf = confidence or self.confidence
        symbols = [s for s in positions if s in returns_df.columns]
        if not symbols:
            return 0.0

        weights = {}
        total_val = sum(
            positions[s]["quantity"] * positions[s]["entry_price"]
            for s in symbols
        )
        for s in symbols:
            val = positions[s]["quantity"] * positions[s]["entry_price"]
            weights[s] = val / total_val if total_val > 0 else 1 / len(symbols)

        # Portfolio returns
        w_arr    = np.array([weights[s] for s in symbols])
        ret_mat  = returns_df[symbols].fillna(0).values
        port_ret = ret_mat @ w_arr

        sorted_ret = np.sort(port_ret)
        cutoff     = int(len(sorted_ret) * (1 - conf))
        tail       = sorted_ret[:max(1, cutoff)]
        cvar_pct   = float(np.mean(tail))

        return round(abs(cvar_pct) * total_val, 2)

    def expected_shortfall_report(self, df: pd.DataFrame, capital: float) -> dict:
        """Full CVaR report at multiple confidence levels."""
        rets = df["close"].pct_change().dropna().iloc[-self.lookback:].values
        if len(rets) < 20:
            return {}

        sorted_ret = np.sort(rets)
        report = {}
        for conf in (0.90, 0.95, 0.99):
            cutoff = int(len(sorted_ret) * (1 - conf))
            tail   = sorted_ret[:max(1, cutoff)]
            cvar   = abs(float(np.mean(tail))) * capital
            report[f"CVaR_{int(conf*100)}"] = round(cvar, 2)

        report["annual_vol_pct"] = round(float(np.std(rets) * np.sqrt(252) * 100), 2)
        return report
