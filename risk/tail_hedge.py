"""Tail risk hedging — auto-suggest Nifty Put hedge when drawdown risk is high."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from utils.logger import logger


class TailRiskHedger:
    """
    Automatically suggests Nifty Put option hedges when portfolio risk is elevated.

    Strategy:
      - Monitor portfolio drawdown + VIX + cross-asset signals
      - When risk score > threshold: suggest buying OTM Nifty Puts
      - Size hedge proportional to portfolio value at risk
      - Unwind hedge when risk score normalizes

    Hedge instrument: Nifty 50 Weekly/Monthly Puts (1-3% OTM)
    """

    def __init__(
        self,
        hedge_trigger_score:  float = 6.0,   # risk score [0-10] to trigger hedge
        unwind_score:         float = 3.0,   # score to unwind
        hedge_ratio:          float = 0.50,  # hedge 50% of portfolio delta
        otm_pct:              float = 0.02,  # 2% OTM put
    ):
        self.trigger_score = hedge_trigger_score
        self.unwind_score  = unwind_score
        self.hedge_ratio   = hedge_ratio
        self.otm_pct       = otm_pct
        self._hedge_on     = False
        self._hedge_details: dict = {}

    # ------------------------------------------------------------------
    def assess(
        self,
        portfolio_value: float,
        drawdown_pct: float,
        vix_level: float,
        cross_asset_score: float,   # from CrossAssetSignals (negative = bearish)
        portfolio_var_pct: float,   # 1-day VaR as % of portfolio
    ) -> dict:
        """Compute risk score and recommend hedge action."""
        # Risk score components [0-10 each]
        drawdown_score = min(drawdown_pct / 0.15 * 10, 10)           # 15% DD = max
        vix_score      = min((vix_level - 10) / 20 * 10, 10)         # VIX 30 = max
        ca_score       = min(abs(min(cross_asset_score, 0)) / 5 * 10, 10)  # -5 = max
        var_score      = min(portfolio_var_pct / 0.03 * 10, 10)       # 3% VaR = max

        risk_score = (0.30 * drawdown_score + 0.30 * vix_score
                      + 0.20 * ca_score + 0.20 * var_score)

        should_hedge  = risk_score >= self.trigger_score
        should_unwind = self._hedge_on and risk_score <= self.unwind_score

        if should_hedge and not self._hedge_on:
            hedge_rec  = self._recommend_hedge(portfolio_value)
            self._hedge_on      = True
            self._hedge_details = hedge_rec
            action = "INITIATE_HEDGE"
        elif should_unwind:
            self._hedge_on = False
            hedge_rec      = {}
            action = "UNWIND_HEDGE"
        elif self._hedge_on:
            hedge_rec = self._hedge_details
            action    = "MAINTAIN_HEDGE"
        else:
            hedge_rec = {}
            action    = "NO_HEDGE"

        return {
            "risk_score":      round(risk_score, 2),
            "action":          action,
            "hedge_on":        self._hedge_on,
            "components": {
                "drawdown_score": round(drawdown_score, 2),
                "vix_score":      round(vix_score, 2),
                "ca_score":       round(ca_score, 2),
                "var_score":      round(var_score, 2),
            },
            "hedge": hedge_rec,
            "cost_estimate_pct": hedge_rec.get("cost_pct", 0),
        }

    def _recommend_hedge(self, portfolio_value: float) -> dict:
        """Recommend a specific Nifty Put hedge."""
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI")
            spot  = getattr(nifty.fast_info, "last_price", 22000)
        except Exception:
            spot = 22000

        strike  = round(spot * (1 - self.otm_pct) / 50) * 50  # round to 50
        delta   = -0.25   # approximate delta for 2% OTM put
        lots    = max(1, int(portfolio_value * self.hedge_ratio / (spot * 50 * abs(delta))))

        # Approximate put price (Black-Scholes ballpark at 30 DTE, 15% IV)
        T       = 30 / 365
        approx_premium = spot * 0.015 * np.sqrt(T / (30/365)) * (self.otm_pct / 0.02)
        total_cost      = approx_premium * lots * 50   # lot size = 50
        cost_pct        = total_cost / portfolio_value * 100

        # Expiry: next month's last Thursday
        expiry = self._next_expiry()

        return {
            "instrument":    "NIFTY_PUT",
            "strike":         strike,
            "spot":           round(spot, 2),
            "lots":           lots,
            "lot_size":       50,
            "expiry":         expiry.isoformat(),
            "approx_premium": round(approx_premium, 2),
            "total_cost":     round(total_cost, 2),
            "cost_pct":       round(cost_pct, 3),
            "delta_hedge":    round(delta * lots * 50, 1),
        }

    @staticmethod
    def _next_expiry() -> date:
        """Next Nifty monthly expiry (last Thursday)."""
        today = date.today()
        if today.month == 12:
            d = date(today.year + 1, 1, 31)
        else:
            d = date(today.year, today.month + 1, 28)
        while d.weekday() != 3:
            d -= timedelta(days=1)
        return d

    def get_hedge_status(self) -> dict:
        return {"hedge_on": self._hedge_on, "details": self._hedge_details}
