"""GARCH volatility forecasting for dynamic SL/target sizing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class GARCHVolatilityForecaster:
    """
    Fits GARCH(1,1) on log-returns and forecasts next-day volatility.

    Falls back to rolling std if arch library not installed.
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self._fitted_models: dict[str, object] = {}
        self._has_arch = self._check_arch()

    @staticmethod
    def _check_arch() -> bool:
        try:
            import arch  # noqa
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def fit(self, symbol: str, df: pd.DataFrame):
        """Fit GARCH model for a symbol."""
        if len(df) < 60:
            return
        try:
            ret = np.log(df["close"] / df["close"].shift(1)).dropna() * 100
            if self._has_arch:
                from arch import arch_model
                model  = arch_model(ret, vol="Garch", p=self.p, q=self.q,
                                    dist="Normal", rescale=False)
                result = model.fit(disp="off", show_warning=False)
                self._fitted_models[symbol] = result
            else:
                # Fallback: store rolling std as "model"
                self._fitted_models[symbol] = ("rolling", float(ret.rolling(20).std().iloc[-1]))
        except Exception as exc:
            logger.debug(f"GARCH fit {symbol}: {exc}")

    def forecast_volatility(self, symbol: str, horizon: int = 1) -> float:
        """
        Return forecasted daily volatility (%) for next `horizon` days.
        Use this to scale ATR-based SL/targets.
        """
        model = self._fitted_models.get(symbol)
        if model is None:
            return 1.5   # default 1.5% daily vol

        try:
            if self._has_arch and not isinstance(model, tuple):
                fc  = model.forecast(horizon=horizon, reindex=False)
                vol = float(np.sqrt(fc.variance.values[-1, -1]))
                return round(vol, 4)
            else:
                _, rolling_std = model
                return round(float(rolling_std), 4)
        except Exception:
            return 1.5

    def volatility_regime(self, symbol: str) -> str:
        """LOW / NORMAL / HIGH / EXTREME based on forecast vs historical."""
        vol = self.forecast_volatility(symbol)
        if vol < 0.8:   return "LOW"
        if vol < 1.5:   return "NORMAL"
        if vol < 2.5:   return "HIGH"
        return "EXTREME"

    def sl_multiplier(self, symbol: str, base: float = 1.2) -> float:
        """Widen SL when vol is high, tighten when low."""
        vol = self.forecast_volatility(symbol)
        # Scale: 1% vol → ×1.0, 2% vol → ×1.5, 0.5% vol → ×0.8
        scale = np.clip(vol / 1.0, 0.7, 2.0)
        return round(float(base * scale), 3)

    def target_multiplier(self, symbol: str, base: float = 2.0) -> float:
        """Raise target when vol is high (more potential range)."""
        vol = self.forecast_volatility(symbol)
        scale = np.clip(vol / 1.0, 0.8, 1.8)
        return round(float(base * scale), 3)

    def fit_all(self, enriched_dfs: dict[str, pd.DataFrame]):
        """Batch fit for all symbols."""
        for symbol, df in enriched_dfs.items():
            self.fit(symbol, df)
        logger.info(f"GARCH: fitted {len(self._fitted_models)} models")
