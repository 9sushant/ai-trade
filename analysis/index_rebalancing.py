"""Index rebalancing signals — Nifty 50 additions/deletions create predictable flows."""
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime, date
from utils.logger import logger

# NSE index composition API
_NSE_INDEX_URL   = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
_NSE_HEADERS     = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer":    "https://www.nseindia.com/",
    "Accept":     "application/json",
}
_CACHE_TTL = 86400   # 24 hours


class IndexRebalancingTracker:
    """
    Detects and trades around Nifty 50 / Nifty Next 50 index rebalancing events.

    Patterns:
      - Addition candidates: price rises 3-5% in week before announcement
      - Deletion candidates: price falls on announcement day
      - Passive fund forced buying: +1-3% on effective rebalancing date
      - Index arbitrage: simultaneous long addition / short deletion

    Data sources:
      - NSE index composition (current)
      - IISL (India Index Services) announcements (simulated)
    """

    def __init__(self):
        self._current_nifty50: list[str] = []
        self._prev_nifty50:    list[str] = []
        self._cache: tuple[list, datetime] | None = None
        self._additions:  list[str] = []
        self._deletions:  list[str] = []

    # ------------------------------------------------------------------
    def refresh(self):
        """Fetch current Nifty 50 composition."""
        self._prev_nifty50    = list(self._current_nifty50)
        self._current_nifty50 = self._fetch_nifty50()
        self._detect_changes()

    def get_addition_candidates(self) -> list[str]:
        """Stocks likely to be added to Nifty 50 (Nifty Next 50 top performers)."""
        return self._additions

    def get_deletion_candidates(self) -> list[str]:
        """Stocks at risk of deletion (lowest market cap in Nifty 50)."""
        return self._deletions

    def get_rebalancing_signal(self, symbol: str) -> dict:
        """Return trading signal for a symbol based on rebalancing flow."""
        in_current = symbol in self._current_nifty50
        was_added  = symbol in self._additions
        was_deleted = symbol in self._deletions

        if was_added:
            return {
                "symbol":       symbol,
                "event":        "ADDITION",
                "signal":       "BUY",
                "reason":       "Passive fund forced buying on addition",
                "size_factor":  1.2,
                "hold_days":    5,
            }
        if was_deleted:
            return {
                "symbol":       symbol,
                "event":        "DELETION",
                "signal":       "SELL",
                "reason":       "Passive fund forced selling on deletion",
                "size_factor":  1.1,
                "hold_days":    3,
            }
        if in_current:
            return {
                "symbol":   symbol,
                "event":    "IN_INDEX",
                "signal":   "NEUTRAL",
                "reason":   "Regular index constituent",
                "size_factor": 1.0,
            }
        return {
            "symbol": symbol, "event": "NOT_IN_INDEX",
            "signal": "NEUTRAL", "size_factor": 1.0,
        }

    def upcoming_rebalancing_dates(self) -> list[date]:
        """
        IISL typically rebalances Nifty 50 in March and September.
        Returns next two rebalancing dates.
        """
        today = date.today()
        dates = []
        for year in [today.year, today.year + 1]:
            for month in [3, 9]:
                d = date(year, month, 31 if month == 3 else 30)
                if d > today:
                    dates.append(d)
        return sorted(dates)[:2]

    def days_to_rebalancing(self) -> int:
        """Days until the next Nifty 50 rebalancing."""
        upcoming = self.upcoming_rebalancing_dates()
        if not upcoming:
            return 180
        return (upcoming[0] - date.today()).days

    def is_rebalancing_week(self) -> bool:
        """True if within 5 days of a rebalancing date."""
        return self.days_to_rebalancing() <= 5

    def nifty50_weight(self, symbol: str) -> float:
        """Approximate Nifty 50 weight (proportional to market cap)."""
        # Simplified: top-10 stocks get higher weight
        top10 = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
                  "BHARTIARTL", "SBIN", "HDFC", "LT", "KOTAKBANK"]
        if symbol in top10:
            return 0.05 + top10.index(symbol) * 0.002
        if symbol in self._current_nifty50:
            return 0.01
        return 0.0

    # ------------------------------------------------------------------
    def _fetch_nifty50(self) -> list[str]:
        cached = self._cache
        if cached:
            data, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return data

        try:
            sess = requests.Session()
            sess.get("https://www.nseindia.com/", headers=_NSE_HEADERS, timeout=8)
            resp = sess.get(_NSE_INDEX_URL, headers=_NSE_HEADERS, timeout=8)
            if resp.ok:
                raw  = resp.json()
                syms = [r.get("symbol", "") for r in raw.get("data", [])]
                syms = [s for s in syms if s]
                self._cache = (syms, datetime.now())
                return syms
        except Exception as exc:
            logger.debug(f"IndexRebalancing fetch: {exc}")

        # Fallback: known Nifty 50 symbols
        from config.settings import MarketConfig
        return MarketConfig.NIFTY_50_SYMBOLS

    def _detect_changes(self):
        if not self._prev_nifty50:
            return
        prev = set(self._prev_nifty50)
        curr = set(self._current_nifty50)
        self._additions = list(curr - prev)
        self._deletions = list(prev - curr)
        if self._additions:
            logger.info(f"IndexRebalancing: ADDITIONS={self._additions}")
        if self._deletions:
            logger.info(f"IndexRebalancing: DELETIONS={self._deletions}")
