"""Dynamic risk adjustment based on drawdown and volatility regime."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class DynamicRiskManager:
    def __init__(self, base_risk_pct: float = 0.01, max_risk_pct: float = 0.02,
                 min_risk_pct: float = 0.005):
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct  = max_risk_pct
        self.min_risk_pct  = min_risk_pct
        self._current_equity = 0.0
        self._peak_equity    = 0.0
        self._initial_equity = 0.0

    def set_initial(self, capital: float):
        self._current_equity = capital
        self._peak_equity    = capital
        self._initial_equity = capital

    def update_equity(self, new_equity: float):
        self._current_equity = new_equity
        if new_equity > self._peak_equity:
            self._peak_equity = new_equity

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity * 100

    def get_risk_multiplier(self) -> float:
        dd = self.current_drawdown_pct
        if dd <= 3:   return 1.00
        if dd <= 5:   return 0.75
        if dd <= 8:   return 0.50
        if dd <= 12:  return 0.25
        return 0.10

    def volatility_adjustment(self, df: pd.DataFrame) -> float:
        """Returns extra multiplier based on recent vs baseline volatility."""
        if df is None or len(df) < 25:
            return 1.0
        try:
            returns  = df["close"].pct_change().dropna()
            vol_20d  = float(returns.tail(20).std())
            vol_60d  = float(returns.tail(60).std()) if len(returns) >= 60 else vol_20d
            if vol_60d <= 0:
                return 1.0
            ratio = vol_20d / vol_60d
            if ratio > 2.0:   return 0.60   # Very high short-term vol
            if ratio > 1.5:   return 0.80
            if ratio < 0.5:   return 1.20   # Unusually calm — can size up slightly
            return 1.0
        except Exception:
            return 1.0

    def get_position_risk(self, capital: float, df: pd.DataFrame = None) -> float:
        multiplier = self.get_risk_multiplier()
        if df is not None:
            multiplier *= self.volatility_adjustment(df)
        risk = capital * self.base_risk_pct * multiplier
        return float(np.clip(risk, capital * self.min_risk_pct,
                                    capital * self.max_risk_pct))

    def get_quantity(self, capital: float, entry_price: float,
                     stop_loss: float, df: pd.DataFrame = None) -> int:
        risk_amount    = self.get_position_risk(capital, df)
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        return max(1, int(risk_amount / risk_per_share))

    def should_halt_trading(self) -> bool:
        return self.current_drawdown_pct > 15.0

    def get_status(self) -> dict:
        return {
            "current_equity":  round(self._current_equity, 2),
            "peak_equity":     round(self._peak_equity, 2),
            "drawdown_pct":    round(self.current_drawdown_pct, 2),
            "multiplier":      round(self.get_risk_multiplier(), 2),
            "should_halt":     self.should_halt_trading(),
            "risk_amount":     round(self.get_position_risk(self._current_equity), 2),
        }
