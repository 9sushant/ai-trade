"""F&O expiry pattern signals — Thursday expiry effects and rollover signals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
from utils.logger import logger


def _last_thursday(d: date) -> date:
    """Return the last Thursday of the month containing date d."""
    # Find last day of month
    if d.month == 12:
        last_day = date(d.year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(d.year, d.month + 1, 1) - timedelta(days=1)
    # Walk back to Thursday (weekday 3)
    offset = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=offset)


class FNOExpiryAnalyzer:
    """
    Captures recurring F&O expiry patterns:
      - Expiry week: increased volatility, short-covering rallies
      - Rollover period (last 3 days before expiry): directional bias
      - Post-expiry (day after): mean-reversion tendency
      - Max pain pull: price gravitates toward max pain strike
    """

    def __init__(self):
        self._expiry_dates: dict[str, date] = {}   # month_key → expiry date

    # ------------------------------------------------------------------
    def get_expiry_date(self, ref_date: date = None) -> date:
        """Return current month's expiry Thursday."""
        ref_date = ref_date or date.today()
        key = f"{ref_date.year}-{ref_date.month:02d}"
        if key not in self._expiry_dates:
            self._expiry_dates[key] = _last_thursday(ref_date)
        return self._expiry_dates[key]

    def days_to_expiry(self, ref_date: date = None) -> int:
        ref_date = ref_date or date.today()
        expiry   = self.get_expiry_date(ref_date)
        delta    = (expiry - ref_date).days
        # If past expiry, get next month
        if delta < 0:
            next_month = date(ref_date.year + (ref_date.month // 12),
                              ref_date.month % 12 + 1, 1)
            expiry = self.get_expiry_date(next_month)
            delta  = (expiry - ref_date).days
        return delta

    def get_expiry_signal(self, ref_date: date = None) -> dict:
        """Return expiry-based trading signal and size adjustment."""
        ref_date = ref_date or date.today()
        dte      = self.days_to_expiry(ref_date)
        expiry   = self.get_expiry_date(ref_date)
        is_expiry_day    = ref_date == expiry
        is_rollover_week = 0 <= dte <= 3
        is_post_expiry   = dte == 20 or (ref_date - expiry).days == 1

        # In rollover week — increase volatility expectation, widen SL
        if is_expiry_day:
            bias          = "VOLATILE"
            size_factor   = 0.75    # reduce size on expiry day
            sl_multiplier = 1.3     # widen SL
            avoid_trade   = True    # skip expiry day entirely
        elif is_rollover_week:
            bias          = "SHORT_COVERING"   # shorts tend to cover before expiry
            size_factor   = 0.9
            sl_multiplier = 1.2
            avoid_trade   = False
        elif is_post_expiry:
            bias          = "MEAN_REVERT"
            size_factor   = 1.0
            sl_multiplier = 1.0
            avoid_trade   = False
        elif dte <= 7:
            bias          = "EXPIRY_WEEK"
            size_factor   = 0.9
            sl_multiplier = 1.1
            avoid_trade   = False
        else:
            bias          = "NORMAL"
            size_factor   = 1.0
            sl_multiplier = 1.0
            avoid_trade   = False

        return {
            "days_to_expiry":  dte,
            "expiry_date":     expiry.isoformat(),
            "is_expiry_day":   is_expiry_day,
            "is_rollover_week": is_rollover_week,
            "bias":            bias,
            "size_factor":     size_factor,
            "sl_multiplier":   sl_multiplier,
            "avoid_trade":     avoid_trade,
        }

    def size_factor(self, ref_date: date = None) -> float:
        return self.get_expiry_signal(ref_date)["size_factor"]

    def should_avoid(self, ref_date: date = None) -> bool:
        return self.get_expiry_signal(ref_date)["avoid_trade"]

    def rollover_direction(self, symbol: str, ref_date: date = None) -> str:
        """
        Estimate rollover direction from OI data (simplified).
        HIGH OI with price up → long rollover → BULLISH
        HIGH OI with price down → short rollover → BEARISH
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            info   = ticker.fast_info
            price  = getattr(info, "last_price", 0)
            prev   = getattr(info, "previous_close", price)
            if price > prev * 1.005:
                return "BULLISH"
            if price < prev * 0.995:
                return "BEARISH"
        except Exception:
            pass
        return "NEUTRAL"
