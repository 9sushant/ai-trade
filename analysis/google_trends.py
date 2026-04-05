"""Google Trends signal — retail attention precedes price moves by 1-2 days."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

_CACHE_TTL = 7200   # 2 hours


class GoogleTrendsSignal:
    """
    Fetches Google Trends search volume for NSE stocks.

    Research shows: spike in search volume → price move 1-2 days later.
    Uses pytrends library; falls back to zero signal if not installed.

    Score: [-5, +5]
      +5 = massive surge in searches (retail FOMO incoming)
      -5 = search volume crash (retail losing interest)
    """

    def __init__(self):
        self._cache: dict[str, tuple[dict, datetime]] = {}
        self._has_pytrends = self._check_pytrends()

    @staticmethod
    def _check_pytrends() -> bool:
        try:
            from pytrends.request import TrendReq  # noqa
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def get_signal(self, symbol: str) -> dict:
        cached = self._cache.get(symbol)
        if cached:
            data, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return data

        result = self._fetch_trends(symbol)
        self._cache[symbol] = (result, datetime.now())
        return result

    def get_score(self, symbol: str) -> float:
        return self.get_signal(symbol).get("score", 0.0)

    def is_attention_spike(self, symbol: str) -> bool:
        sig = self.get_signal(symbol)
        return sig.get("trend", "FLAT") in ("SURGING", "RISING")

    def composite_score_boost(self, symbol: str) -> float:
        """Return score adjustment [-5, +5] for composite signal."""
        return self.get_score(symbol)

    # ------------------------------------------------------------------
    def _fetch_trends(self, symbol: str) -> dict:
        if not self._has_pytrends:
            return self._empty(symbol)

        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-IN", tz=330, timeout=(5, 15))

            # Search terms: stock name + NSE suffix
            kw_list = [f"{symbol} share", f"{symbol} stock"]
            pt.build_payload(kw_list, timeframe="today 3-m", geo="IN")
            df = pt.interest_over_time()

            if df is None or df.empty:
                return self._empty(symbol)

            # Use primary keyword
            col  = kw_list[0] if kw_list[0] in df.columns else df.columns[0]
            vals = df[col].values.astype(float)

            if len(vals) < 4:
                return self._empty(symbol)

            current  = float(vals[-1])
            avg_4w   = float(np.mean(vals[-4:]))
            avg_12w  = float(np.mean(vals[-12:]) if len(vals) >= 12 else np.mean(vals))
            momentum = (current - avg_4w) / max(avg_4w, 1)
            z_score  = (current - avg_12w) / max(np.std(vals), 1)

            # Score: z_score clipped to [-5, +5]
            score = float(np.clip(z_score, -5, 5))

            if z_score > 2:      trend = "SURGING"
            elif z_score > 0.5:  trend = "RISING"
            elif z_score < -2:   trend = "COLLAPSING"
            elif z_score < -0.5: trend = "FALLING"
            else:                trend = "FLAT"

            return {
                "symbol":       symbol,
                "current":      round(current, 1),
                "avg_4w":       round(avg_4w,  1),
                "z_score":      round(z_score, 3),
                "momentum":     round(momentum, 3),
                "score":        round(score, 2),
                "trend":        trend,
                "lead_signal":  "BUY" if score > 2 else "SELL" if score < -2 else "NEUTRAL",
            }

        except Exception as exc:
            logger.debug(f"GoogleTrends {symbol}: {exc}")
            return self._empty(symbol)

    @staticmethod
    def _empty(symbol: str) -> dict:
        return {
            "symbol": symbol, "current": 0, "avg_4w": 0,
            "z_score": 0, "momentum": 0, "score": 0.0,
            "trend": "UNKNOWN", "lead_signal": "NEUTRAL",
        }
