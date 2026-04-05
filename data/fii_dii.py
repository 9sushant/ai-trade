"""FII/DII institutional flow tracker using NSE data."""
from __future__ import annotations
import requests
import numpy as np
from datetime import datetime
from utils.logger import logger

_NSE_URL     = "https://www.nseindia.com/api/fiidiiTradeReact"
_NSE_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer":         "https://www.nseindia.com/",
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class FIIDIITracker:
    _CACHE_TTL = 4 * 3600   # 4 hours

    def __init__(self):
        self._cache: tuple[list, datetime] | None = None

    def get_flow(self, date: str = None) -> dict:
        """Return latest FII/DII flow dict."""
        data = self._fetch()
        if not data:
            return {}
        # NSE returns latest day first
        row = data[0] if data else {}
        return {
            "date":      row.get("date", ""),
            "fii_buy":   self._to_float(row.get("fiiBuying")),
            "fii_sell":  self._to_float(row.get("fiiSelling")),
            "fii_net":   self._to_float(row.get("fiiNet")),
            "dii_buy":   self._to_float(row.get("diiBuying")),
            "dii_sell":  self._to_float(row.get("diiSelling")),
            "dii_net":   self._to_float(row.get("diiNet")),
        }

    def get_trend(self, lookback_days: int = 5) -> str:
        data = self._fetch()
        if not data:
            return "NEUTRAL"
        rows     = data[:lookback_days]
        fii_nets = [self._to_float(r.get("fiiNet", 0)) for r in rows]
        avg      = float(np.mean(fii_nets)) if fii_nets else 0
        if avg > 500:    return "BULLISH"
        if avg < -500:   return "BEARISH"
        return "NEUTRAL"

    def get_score(self) -> float:
        """Return score [-10, +10] based on 5-day net FII flow."""
        data = self._fetch()
        if not data:
            return 0.0
        rows     = data[:5]
        fii_nets = [self._to_float(r.get("fiiNet", 0)) for r in rows]
        avg      = float(np.mean(fii_nets)) if fii_nets else 0
        # Scale: ±5000 Cr → ±10 score
        score    = np.clip(avg / 500, -10, 10)
        return round(float(score), 2)

    def _fetch(self) -> list:
        if self._cache:
            data, ts = self._cache
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return data
        try:
            sess = requests.Session()
            # NSE requires a session cookie first
            sess.get("https://www.nseindia.com/", headers=_NSE_HEADERS, timeout=8)
            resp = sess.get(_NSE_URL, headers=_NSE_HEADERS, timeout=8)
            if resp.ok:
                data = resp.json()
                if isinstance(data, list):
                    self._cache = (data, datetime.now())
                    return data
        except Exception as exc:
            logger.debug(f"FII/DII fetch failed: {exc}")
        return []

    @staticmethod
    def _to_float(val) -> float:
        try:
            return float(str(val).replace(",", ""))
        except Exception:
            return 0.0
