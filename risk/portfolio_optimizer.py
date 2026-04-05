"""Mean-variance portfolio optimization using PyPortfolioOpt."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class PortfolioOptimizer:
    """
    Optimizes capital allocation across open positions.

    Uses PyPortfolioOpt (if installed) for:
      - Maximum Sharpe ratio portfolio
      - Minimum volatility portfolio
      - Risk parity allocation

    Falls back to equal-weight / volatility-scaled if not installed.
    """

    def __init__(self, risk_free_rate: float = 0.065):  # 6.5% India RFR
        self.risk_free_rate = risk_free_rate
        self._has_pypfopt   = self._check_pypfopt()

    @staticmethod
    def _check_pypfopt() -> bool:
        try:
            import pypfopt  # noqa
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def max_sharpe_weights(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """
        returns_df: DataFrame of daily returns, columns = symbols.
        Returns dict of symbol → weight (sums to 1).
        """
        symbols = list(returns_df.columns)

        if self._has_pypfopt and len(symbols) >= 2:
            try:
                from pypfopt import EfficientFrontier, risk_models, expected_returns
                mu  = expected_returns.mean_historical_return(returns_df + 1,
                                                               returns_data=True)
                S   = risk_models.sample_cov(returns_df + 1, returns_data=True)
                ef  = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w >= 0.05)   # min 5% per position
                ef.add_constraint(lambda w: w <= 0.40)   # max 40% per position
                raw = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                weights = ef.clean_weights()
                return dict(weights)
            except Exception as exc:
                logger.debug(f"PyPortfolioOpt max_sharpe error: {exc}")

        return self._vol_scaled_weights(returns_df)

    def min_volatility_weights(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """Minimum variance portfolio weights."""
        symbols = list(returns_df.columns)

        if self._has_pypfopt and len(symbols) >= 2:
            try:
                from pypfopt import EfficientFrontier, risk_models, expected_returns
                mu = expected_returns.mean_historical_return(returns_df + 1,
                                                              returns_data=True)
                S  = risk_models.sample_cov(returns_df + 1, returns_data=True)
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w >= 0.05)
                ef.add_constraint(lambda w: w <= 0.40)
                ef.min_volatility()
                return dict(ef.clean_weights())
            except Exception as exc:
                logger.debug(f"PyPortfolioOpt min_vol error: {exc}")

        return self._vol_scaled_weights(returns_df)

    def risk_parity_weights(self, returns_df: pd.DataFrame) -> dict[str, float]:
        """Equal risk contribution (risk parity) weights."""
        vols    = returns_df.std()
        inv_vol = 1 / (vols + 1e-8)
        weights = inv_vol / inv_vol.sum()
        return dict(weights)

    # ------------------------------------------------------------------
    def optimal_quantities(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
        prices: dict[str, float],
        total_capital: float,
        method: str = "max_sharpe",
    ) -> dict[str, int]:
        """
        Returns optimal integer share quantities for each symbol.

        method: "max_sharpe" | "min_vol" | "risk_parity" | "equal"
        """
        if method == "max_sharpe":
            weights = self.max_sharpe_weights(returns_df[symbols])
        elif method == "min_vol":
            weights = self.min_volatility_weights(returns_df[symbols])
        elif method == "risk_parity":
            weights = self.risk_parity_weights(returns_df[symbols])
        else:
            n = len(symbols)
            weights = {s: 1.0 / n for s in symbols}

        result = {}
        for sym in symbols:
            w     = weights.get(sym, 0)
            price = prices.get(sym, 0)
            if price > 0 and w > 0:
                alloc    = total_capital * w
                qty      = max(1, int(alloc / price))
                result[sym] = qty
        return result

    def portfolio_expected_return(self, weights: dict[str, float],
                                   returns_df: pd.DataFrame) -> float:
        """Annualized expected return of a weight allocation."""
        mu = returns_df.mean() * 252
        total = sum(weights.get(s, 0) * mu.get(s, 0)
                    for s in returns_df.columns)
        return round(float(total), 4)

    def portfolio_volatility(self, weights: dict[str, float],
                              returns_df: pd.DataFrame) -> float:
        """Annualized portfolio volatility."""
        w   = np.array([weights.get(s, 0) for s in returns_df.columns])
        cov = returns_df.cov().values * 252
        vol = float(np.sqrt(w @ cov @ w))
        return round(vol, 4)

    # ------------------------------------------------------------------
    @staticmethod
    def _vol_scaled_weights(returns_df: pd.DataFrame) -> dict[str, float]:
        """Inverse-volatility weighting fallback."""
        vols    = returns_df.std() + 1e-8
        inv_vol = 1 / vols
        weights = inv_vol / inv_vol.sum()
        # Clip to [5%, 40%]
        weights = weights.clip(0.05, 0.40)
        weights = weights / weights.sum()
        return dict(weights)
