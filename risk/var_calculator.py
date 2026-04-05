"""Value at Risk (VaR) based position sizing using historical simulation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class VaRCalculator:
    def __init__(self, confidence: float = 0.95, lookback_days: int = 252):
        self.confidence    = confidence
        self.lookback_days = lookback_days

    def compute_var(self, df: pd.DataFrame, capital: float) -> float:
        """Return 1-day VaR as a positive rupee amount at self.confidence level."""
        if df is None or len(df) < 30:
            return capital * 0.02
        try:
            returns = df["close"].pct_change().dropna().tail(self.lookback_days)
            if len(returns) < 20:
                return capital * 0.02
            var_pct = float(np.percentile(returns, (1 - self.confidence) * 100))
            return abs(var_pct) * capital
        except Exception as exc:
            logger.debug(f"VaR compute error: {exc}")
            return capital * 0.02

    def var_adjusted_quantity(self, df: pd.DataFrame, capital: float,
                               entry_price: float, stop_loss: float) -> int:
        """Max quantity such that 1-day VaR exposure ≤ (1-confidence) × capital."""
        if entry_price <= 0:
            return 0
        var_amount = self.compute_var(df, capital)
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        qty = int(var_amount / risk_per_share)
        return max(1, qty)

    def portfolio_var(self, positions: list[dict],
                      corr_matrix: pd.DataFrame | None = None) -> float:
        """Portfolio VaR considering correlations (variance-covariance method)."""
        if not positions:
            return 0.0
        vols, values = [], []
        for pos in positions:
            df  = pos.get("df")
            qty = pos.get("qty", 1)
            px  = pos.get("price", 0)
            if df is not None and len(df) > 30:
                ret = df["close"].pct_change().dropna().tail(self.lookback_days)
                vols.append(float(ret.std()) if len(ret) > 5 else 0.02)
            else:
                vols.append(0.02)
            values.append(qty * px)
        weights = np.array(values) / max(sum(values), 1)
        vols    = np.array(vols)
        if corr_matrix is not None:
            syms = [p.get("symbol","") for p in positions]
            try:
                C = corr_matrix.loc[syms, syms].values
            except Exception:
                C = np.eye(len(positions))
        else:
            C = np.eye(len(positions))
        cov   = np.outer(vols, vols) * C
        port_var_pct = float(np.sqrt(weights @ cov @ weights))
        z = abs(float(np.percentile(np.random.normal(0, 1, 100000),
                                    (1 - self.confidence) * 100)))
        return port_var_pct * z * sum(values)

    def stress_var(self, df: pd.DataFrame, capital: float,
                   shock_pct: float = 0.20) -> float:
        """Loss if stock drops shock_pct% instantly."""
        return capital * shock_pct
