"""Fama-French multi-factor model — decompose returns into market/size/value/momentum."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

# Factor proxies for India (using NSE indices as factor portfolios)
_FACTOR_TICKERS = {
    "market":   "^NSEI",       # Nifty 50 = market factor
    "size":     "^NSEMDCP50",  # Midcap vs large = size factor
    "value":    "^CNXPHARMA",  # proxy (low P/E sectors)
    "momentum": "^CNXIT",      # IT sector proxy for momentum
}


class FamaFrenchFactorModel:
    """
    Multi-factor model for NSE stocks.

    Computes factor exposures (betas) and generates alpha signals.

    Factors:
      - Market (Rm - Rf): excess market return
      - Size (SMB): Small Minus Big
      - Value (HML): High book-to-market Minus Low
      - Momentum (MOM): past 12-month winners vs losers
      - Quality (QMJ): profitable vs unprofitable

    Alpha = actual return - predicted factor return (unexplained = edge)
    """

    def __init__(self, lookback_days: int = 252, risk_free_rate: float = 0.065):
        self.lookback        = lookback_days
        self.risk_free       = risk_free_rate / 252   # daily
        self._factor_returns: pd.DataFrame | None = None
        self._betas:  dict[str, dict[str, float]] = {}
        self._alphas: dict[str, float] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    def load_factors(self, period: str = "1y"):
        """Download factor proxy returns."""
        try:
            import yfinance as yf
            factor_data = {}
            for name, ticker in _FACTOR_TICKERS.items():
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 20:
                    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                                   for c in raw.columns]
                    factor_data[name] = raw["close"].pct_change().dropna()

            if factor_data:
                self._factor_returns = pd.DataFrame(factor_data).dropna()
                # Market excess return
                self._factor_returns["market"] -= self.risk_free
                self._loaded = True
                logger.info(f"FamaFrench: loaded {len(self._factor_returns)} days of factors")
        except Exception as exc:
            logger.debug(f"FamaFrench load_factors: {exc}")

    def fit_symbol(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Estimate factor betas and alpha for one symbol.
        Returns dict with betas, alpha, r_squared.
        """
        if not self._loaded or self._factor_returns is None:
            return {}

        try:
            ret = df["close"].pct_change().dropna() - self.risk_free
            aligned = pd.concat([ret.rename("stock"), self._factor_returns],
                                 axis=1).dropna().iloc[-self.lookback:]

            if len(aligned) < 60:
                return {}

            Y = aligned["stock"].values
            X = aligned[list(_FACTOR_TICKERS.keys())].values
            X = np.column_stack([np.ones(len(X)), X])   # add intercept

            # OLS
            betas, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            alpha_daily  = betas[0]
            factor_betas = dict(zip(_FACTOR_TICKERS.keys(), betas[1:]))

            # R-squared
            y_pred  = X @ betas
            ss_res  = np.sum((Y - y_pred) ** 2)
            ss_tot  = np.sum((Y - Y.mean()) ** 2)
            r2      = 1 - ss_res / max(ss_tot, 1e-10)

            # Annualized alpha
            alpha_annual = alpha_daily * 252

            self._betas[symbol]  = factor_betas
            self._alphas[symbol] = alpha_annual

            return {
                "symbol":        symbol,
                "alpha_annual":  round(alpha_annual * 100, 3),   # in %
                "betas":         {k: round(v, 4) for k, v in factor_betas.items()},
                "r_squared":     round(r2, 4),
                "market_beta":   round(factor_betas.get("market", 1.0), 3),
                "has_alpha":     alpha_annual > 0.02,   # > 2% annual alpha
            }
        except Exception as exc:
            logger.debug(f"FamaFrench fit {symbol}: {exc}")
            return {}

    def get_alpha(self, symbol: str) -> float:
        """Return annualized alpha for symbol (positive = outperforms factors)."""
        return self._alphas.get(symbol, 0.0)

    def get_market_beta(self, symbol: str) -> float:
        return self._betas.get(symbol, {}).get("market", 1.0)

    def alpha_signal(self, symbol: str) -> str:
        """BUY if positive alpha, SELL if negative, NEUTRAL otherwise."""
        alpha = self.get_alpha(symbol)
        if alpha > 0.05:   return "BUY"
        if alpha < -0.05:  return "SELL"
        return "NEUTRAL"

    def size_factor_bias(self, symbol: str) -> float:
        """Positive size beta = small-cap tilt (higher risk/return)."""
        return self._betas.get(symbol, {}).get("size", 0.0)

    def momentum_beta(self, symbol: str) -> float:
        return self._betas.get(symbol, {}).get("momentum", 0.0)

    def composite_score_boost(self, symbol: str) -> float:
        """Score boost from factor alpha [-5, +5]."""
        alpha = self.get_alpha(symbol)
        return round(float(np.clip(alpha * 50, -5, 5)), 2)

    def fit_all(self, enriched_dfs: dict[str, pd.DataFrame]):
        """Batch fit for all symbols."""
        if not self._loaded:
            self.load_factors()
        results = {}
        for symbol, df in enriched_dfs.items():
            r = self.fit_symbol(symbol, df)
            if r:
                results[symbol] = r
        logger.info(f"FamaFrench: fitted {len(results)} symbols")
        return results
