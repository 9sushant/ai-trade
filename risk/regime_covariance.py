"""Regime-conditional covariance — correlations spike in crashes, standard VaR misses this."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class RegimeConditionalCovariance:
    """
    Estimates separate covariance matrices per market regime.

    In bear markets, correlations between stocks spike toward 1.0
    (all stocks fall together) — this makes standard VaR dangerously
    optimistic. This model captures that regime-switching behavior.

    Regimes: BULL (low vol, low corr) | BEAR (high vol, high corr) | CRISIS (extreme)
    """

    def __init__(self, vol_lookback: int = 20, crisis_vol_threshold: float = 2.0):
        self.vol_lookback       = vol_lookback
        self.crisis_threshold   = crisis_vol_threshold
        self._cov: dict[str, np.ndarray] = {}
        self._corr: dict[str, np.ndarray] = {}
        self._symbols: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, returns_df: pd.DataFrame):
        """Fit regime-conditional covariance matrices."""
        self._symbols = list(returns_df.columns)
        regimes = self._classify_regimes(returns_df)

        for regime in ("BULL", "BEAR", "CRISIS"):
            mask = regimes == regime
            if mask.sum() < 10:
                # Not enough data — use full-sample as fallback
                sub = returns_df
            else:
                sub = returns_df[mask]

            self._cov[regime]  = sub.cov().values * 252
            self._corr[regime] = sub.corr().values

        logger.info(f"RegimeCov: fitted. BULL={int((regimes=='BULL').sum())}d, "
                    f"BEAR={int((regimes=='BEAR').sum())}d, "
                    f"CRISIS={int((regimes=='CRISIS').sum())}d")

    def get_covariance(self, regime: str) -> np.ndarray:
        """Return annualized covariance matrix for given regime."""
        return self._cov.get(regime, np.eye(len(self._symbols)))

    def get_correlation(self, regime: str) -> np.ndarray:
        return self._corr.get(regime, np.eye(len(self._symbols)))

    def portfolio_var(self, weights: dict[str, float], regime: str,
                      capital: float) -> float:
        """Portfolio VaR using regime-conditional covariance."""
        syms = [s for s in self._symbols if s in weights]
        if not syms:
            return 0.0

        w   = np.array([weights.get(s, 0) for s in syms])
        cov = self._cov.get(regime)
        if cov is None:
            return 0.0

        # Extract sub-matrix for held symbols
        idx = [self._symbols.index(s) for s in syms if s in self._symbols]
        sub = cov[np.ix_(idx, idx)] if idx else np.eye(len(syms))

        port_var = float(w @ sub @ w)
        port_std = np.sqrt(max(port_var, 0)) / np.sqrt(252)
        return round(1.645 * port_std * capital, 2)   # 95% VaR

    def worst_case_var(self, weights: dict[str, float], capital: float) -> float:
        """Return the worst-case VaR across all regimes."""
        vars_ = [
            self.portfolio_var(weights, r, capital)
            for r in ("BULL", "BEAR", "CRISIS")
            if r in self._cov
        ]
        return max(vars_) if vars_ else 0.0

    def correlation_breakdown(self, sym_a: str, sym_b: str) -> dict:
        """Show how correlation changes across regimes."""
        result = {}
        for regime in ("BULL", "BEAR", "CRISIS"):
            corr = self._corr.get(regime)
            if corr is None or sym_a not in self._symbols or sym_b not in self._symbols:
                continue
            i = self._symbols.index(sym_a)
            j = self._symbols.index(sym_b)
            result[regime] = round(float(corr[i, j]), 4)
        return result

    # ------------------------------------------------------------------
    def _classify_regimes(self, returns_df: pd.DataFrame) -> pd.Series:
        """Classify each date into BULL/BEAR/CRISIS based on realized vol."""
        # Use market (first column or average) volatility
        port_ret = returns_df.mean(axis=1)
        vol_roll = port_ret.rolling(self.vol_lookback).std() * np.sqrt(252)
        avg_vol  = vol_roll.mean()

        regimes = pd.Series("BULL", index=returns_df.index)
        regimes[vol_roll > avg_vol * self.crisis_threshold] = "CRISIS"
        regimes[(vol_roll > avg_vol) & (vol_roll <= avg_vol * self.crisis_threshold)] = "BEAR"
        return regimes

    def current_regime(self, returns_df: pd.DataFrame) -> str:
        """Classify the current market regime."""
        regimes = self._classify_regimes(returns_df)
        return str(regimes.iloc[-1]) if len(regimes) > 0 else "BULL"
