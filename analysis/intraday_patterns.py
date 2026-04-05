"""Intraday time-of-day patterns — power hour, lunch drift, close auction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time as dtime
from utils.logger import logger

# NSE trading hours (IST)
_OPEN    = dtime(9, 15)
_CLOSE   = dtime(15, 30)
_LUNCH_S = dtime(12, 0)
_LUNCH_E = dtime(13, 0)
_POWER_S = dtime(14, 30)   # power hour start
_PRE_CLOSE = dtime(15, 0)


class IntradayPatternAnalyzer:
    """
    Identifies time-of-day biases based on historical intraday data.

    Patterns detected:
      - Opening range drift (9:15–9:45): volatile, momentum-driven
      - Lunch lull (12:00–13:00): low volume, avoid entries
      - Power hour (14:30–15:30): high volume, trend acceleration
      - Pre-close auction (15:00–15:15): reversal prone
    """

    def __init__(self):
        self._stats: dict[str, dict] = {}   # symbol → hourly return stats

    def fit(self, symbol: str, intraday_df: pd.DataFrame):
        """Compute time-of-day return statistics from intraday OHLCV."""
        if intraday_df is None or len(intraday_df) < 50:
            return
        try:
            df    = intraday_df.copy()
            df.index = pd.to_datetime(df.index)
            df["hour"]    = df.index.hour
            df["minute"]  = df.index.minute
            df["ret"]     = df["close"].pct_change()

            hourly = df.groupby("hour")["ret"].agg(["mean", "std", "count"])
            self._stats[symbol] = hourly.to_dict()
        except Exception as exc:
            logger.debug(f"IntradayPattern fit {symbol}: {exc}")

    def get_time_signal(self, current_time: dtime = None) -> dict:
        """Return signal for the current time of day."""
        if current_time is None:
            from datetime import datetime
            current_time = datetime.now().time()

        t = current_time

        if _OPEN <= t < dtime(9, 45):
            period = "OPENING"
            bias   = "MOMENTUM"
            size_f = 0.8    # volatile, smaller size
            quality = "MEDIUM"

        elif dtime(9, 45) <= t < _LUNCH_S:
            period  = "MORNING"
            bias    = "TRENDING"
            size_f  = 1.0
            quality = "HIGH"

        elif _LUNCH_S <= t < _LUNCH_E:
            period  = "LUNCH"
            bias    = "AVOID"
            size_f  = 0.5   # low volume, spreads widen
            quality = "LOW"

        elif _LUNCH_E <= t < _POWER_S:
            period  = "AFTERNOON"
            bias    = "MIXED"
            size_f  = 0.9
            quality = "MEDIUM"

        elif _POWER_S <= t < _PRE_CLOSE:
            period  = "POWER_HOUR"
            bias    = "TRENDING"
            size_f  = 1.1   # highest volume, best fills
            quality = "HIGH"

        elif _PRE_CLOSE <= t <= _CLOSE:
            period  = "PRE_CLOSE"
            bias    = "REVERSAL"
            size_f  = 0.7
            quality = "LOW"

        else:
            period  = "CLOSED"
            bias    = "NONE"
            size_f  = 0.0
            quality = "NONE"

        return {
            "period":          period,
            "bias":            bias,
            "size_factor":     size_f,
            "quality":         quality,
            "avoid":           bias in ("AVOID", "NONE"),
            "is_power_hour":   period == "POWER_HOUR",
            "is_lunch":        period == "LUNCH",
        }

    def get_best_entry_windows(self) -> list[dict]:
        """Return the two best intraday windows for entries."""
        return [
            {"window": "09:45–12:00", "bias": "TRENDING",  "quality": "HIGH"},
            {"window": "14:30–15:00", "bias": "TRENDING",  "quality": "HIGH"},
        ]

    def historical_hourly_returns(self, symbol: str) -> dict:
        """Return mean returns by hour for a symbol."""
        return self._stats.get(symbol, {})

    def size_factor(self, current_time: dtime = None) -> float:
        return self.get_time_signal(current_time)["size_factor"]

    def should_avoid(self, current_time: dtime = None) -> bool:
        return self.get_time_signal(current_time)["avoid"]
