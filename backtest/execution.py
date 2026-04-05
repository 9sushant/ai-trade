"""TWAP/VWAP execution model — realistic large-order fill simulation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

# Market impact model parameters
_IMPACT_COEFF = 0.1    # Kyle's lambda approximation
_SPREAD_BPS   = 5      # average bid-ask spread in basis points for Nifty 50


class ExecutionModel:
    """
    Simulates realistic order execution for large positions.

    Models:
      - TWAP (Time-Weighted Average Price) — split order over N bars
      - VWAP (Volume-Weighted Average Price) — slice proportional to volume
      - Market impact — price moves against you as you buy/sell
      - Bid-ask spread cost
    """

    def __init__(
        self,
        use_market_impact: bool = True,
        spread_bps: float = _SPREAD_BPS,
        impact_coeff: float = _IMPACT_COEFF,
    ):
        self.use_market_impact = use_market_impact
        self.spread_bps        = spread_bps
        self.impact_coeff      = impact_coeff

    # ------------------------------------------------------------------
    def twap_fill(
        self,
        df: pd.DataFrame,
        start_idx: int,
        quantity: int,
        direction: str,
        n_slices: int = 5,
    ) -> dict:
        """
        Simulate TWAP execution: split `quantity` into `n_slices` equal parts
        across consecutive bars.

        Returns average fill price and total slippage cost.
        """
        end_idx     = min(start_idx + n_slices, len(df))
        slice_qty   = quantity / n_slices
        fill_prices = []
        costs       = []

        for i in range(start_idx, end_idx):
            bar   = df.iloc[i]
            vwap  = self._bar_vwap(bar)
            price = self._apply_impact(vwap, slice_qty, bar, direction)
            fill_prices.append(price)
            costs.append(self._spread_cost(price, slice_qty))

        avg_fill = float(np.mean(fill_prices)) if fill_prices else float(df.iloc[start_idx]["close"])
        total_cost = sum(costs)

        return {
            "avg_fill_price":  round(avg_fill, 2),
            "slippage_cost":   round(total_cost, 2),
            "n_slices":        len(fill_prices),
            "execution_type":  "TWAP",
        }

    def vwap_fill(
        self,
        df: pd.DataFrame,
        start_idx: int,
        quantity: int,
        direction: str,
        n_slices: int = 5,
    ) -> dict:
        """
        VWAP execution: allocate order slices proportional to historical volume profile.
        """
        end_idx = min(start_idx + n_slices, len(df))
        bars    = [df.iloc[i] for i in range(start_idx, end_idx)]

        volumes = np.array([float(b.get("volume", 1) or 1) for b in bars])
        total_v = volumes.sum()
        if total_v == 0:
            return self.twap_fill(df, start_idx, quantity, direction, n_slices)

        fill_prices = []
        costs       = []
        for bar, vol in zip(bars, volumes):
            slice_qty = quantity * (vol / total_v)
            vwap      = self._bar_vwap(bar)
            price     = self._apply_impact(vwap, slice_qty, bar, direction)
            fill_prices.append(price)
            costs.append(self._spread_cost(price, slice_qty))

        avg_fill = float(np.average(fill_prices, weights=volumes))
        return {
            "avg_fill_price": round(avg_fill, 2),
            "slippage_cost":  round(sum(costs), 2),
            "n_slices":       len(fill_prices),
            "execution_type": "VWAP",
        }

    def market_order_fill(
        self, bar: pd.Series, quantity: int, direction: str
    ) -> dict:
        """Single bar market order — most slippage."""
        close  = float(bar.get("close", 0) or 0)
        spread = close * self.spread_bps / 10_000
        if direction == "BUY":
            fill = close + spread / 2   # pay the ask
        else:
            fill = close - spread / 2   # hit the bid

        impact = self._market_impact(close, quantity, bar)
        fill  += impact if direction == "BUY" else -impact

        return {
            "avg_fill_price": round(fill, 2),
            "slippage_cost":  round(self._spread_cost(close, quantity) + abs(impact * quantity), 2),
            "execution_type": "MARKET",
        }

    def effective_slippage_pct(
        self, df: pd.DataFrame, start_idx: int,
        quantity: int, direction: str, method: str = "vwap"
    ) -> float:
        """Return total slippage as % of notional (for use in backtest cost model)."""
        if method == "vwap":
            result = self.vwap_fill(df, start_idx, quantity, direction)
        elif method == "twap":
            result = self.twap_fill(df, start_idx, quantity, direction)
        else:
            bar    = df.iloc[start_idx]
            result = self.market_order_fill(bar, quantity, direction)

        close    = float(df.iloc[start_idx].get("close", 1) or 1)
        notional = close * quantity
        slippage = result["slippage_cost"]
        return round(slippage / notional * 100 if notional > 0 else 0, 4)

    # ------------------------------------------------------------------
    def _bar_vwap(self, bar: pd.Series) -> float:
        """Approximate VWAP for a single bar as (H+L+C)/3."""
        h = float(bar.get("high",  bar.get("close", 0)) or 0)
        l = float(bar.get("low",   bar.get("close", 0)) or 0)
        c = float(bar.get("close", 0) or 0)
        return (h + l + c) / 3 if c > 0 else c

    def _market_impact(self, price: float, quantity: float, bar: pd.Series) -> float:
        """Kyle's lambda price impact: Δp = λ × quantity / ADV."""
        if not self.use_market_impact:
            return 0.0
        adv = float(bar.get("volume", 1_000_000) or 1_000_000) * price
        if adv <= 0:
            return 0.0
        notional = quantity * price
        return self.impact_coeff * price * (notional / adv) ** 0.5

    def _apply_impact(self, price: float, qty: float,
                      bar: pd.Series, direction: str) -> float:
        impact = self._market_impact(price, qty, bar)
        return price + impact if direction == "BUY" else price - impact

    def _spread_cost(self, price: float, quantity: float) -> float:
        """Half-spread cost (one side)."""
        return price * quantity * (self.spread_bps / 10_000) / 2
