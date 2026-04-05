"""Sector rotation tracker — follow money into leading NSE sectors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

# NSE sector ETFs / indices on Yahoo Finance
_SECTORS = {
    "IT":         "^CNXIT",
    "BANK":       "^NSEBANK",
    "PHARMA":     "^CNXPHARMA",
    "AUTO":       "^CNXAUTO",
    "FMCG":       "^CNXFMCG",
    "METAL":      "^CNXMETAL",
    "ENERGY":     "^CNXENERGY",
    "REALTY":     "^CNXREALTY",
    "INFRA":      "^CNXINFRA",
    "MIDCAP":     "^NSEMDCP50",
}

# Which NIFTY 50 symbols belong to which sector
_SYMBOL_SECTOR = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "PERSISTENTSYS": "IT",
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "KOTAKBANK": "BANK",
    "AXISBANK": "BANK", "SBIN": "BANK", "BAJFINANCE": "BANK",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "BAJAJ-AUTO": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL", "HINDALCO": "METAL",
    "COALINDIA": "ENERGY", "ONGC": "ENERGY", "RELIANCE": "ENERGY",
    "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    "DLF": "REALTY", "GODREJPROP": "REALTY",
    "ASIANPAINT": "FMCG",
}

_CACHE_TTL = 7200   # 2 hours


class SectorRotationTracker:
    """
    Ranks sectors by recent momentum and identifies rotating money flows.
    Score = momentum + relative strength vs Nifty.
    """

    def __init__(self):
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._sector_scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    def refresh(self, period: str = "3mo"):
        """Fetch and score all sectors."""
        scores = {}
        for sector, ticker in _SECTORS.items():
            df = self._fetch(ticker, period)
            if df is not None and len(df) >= 20:
                scores[sector] = self._momentum_score(df)
        self._sector_scores = scores
        logger.info(f"SectorRotation: {scores}")

    def get_leading_sectors(self, top_n: int = 3) -> list[str]:
        """Return top N leading sectors by momentum score."""
        return sorted(self._sector_scores, key=self._sector_scores.get, reverse=True)[:top_n]

    def get_lagging_sectors(self, bottom_n: int = 3) -> list[str]:
        return sorted(self._sector_scores, key=self._sector_scores.get)[:bottom_n]

    def get_sector_score(self, sector: str) -> float:
        return self._sector_scores.get(sector, 0.0)

    def get_symbol_sector_score(self, symbol: str) -> float:
        """Return sector momentum score for a given symbol."""
        sector = _SYMBOL_SECTOR.get(symbol, "")
        return self._sector_scores.get(sector, 0.0)

    def is_in_leading_sector(self, symbol: str, top_n: int = 4) -> bool:
        """True if the symbol's sector is in the top N leading sectors."""
        if not self._sector_scores:
            return True  # no data → don't block
        symbol_sector = _SYMBOL_SECTOR.get(symbol, "")
        if not symbol_sector:
            return True
        return symbol_sector in self.get_leading_sectors(top_n)

    def sector_bias(self, symbol: str) -> float:
        """Return position size adjustment: +0.2 for leading, -0.2 for lagging."""
        score = self.get_symbol_sector_score(symbol)
        if not self._sector_scores:
            return 0.0
        max_s = max(self._sector_scores.values()) if self._sector_scores else 1
        min_s = min(self._sector_scores.values()) if self._sector_scores else -1
        rng   = max_s - min_s
        if rng == 0:
            return 0.0
        normalized = (score - min_s) / rng   # [0, 1]
        return round((normalized - 0.5) * 0.4, 3)   # [-0.2, +0.2]

    def get_rotation_report(self) -> dict:
        return {
            "leading":  self.get_leading_sectors(3),
            "lagging":  self.get_lagging_sectors(3),
            "scores":   {k: round(v, 3) for k, v in sorted(
                self._sector_scores.items(), key=lambda x: x[1], reverse=True)},
        }

    # ------------------------------------------------------------------
    def _fetch(self, ticker: str, period: str) -> pd.DataFrame | None:
        cached = self._cache.get(ticker)
        if cached:
            df, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return df
        try:
            import yfinance as yf
            raw = yf.download(ticker, period=period, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is None or len(raw) < 10:
                return None
            raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                           for c in raw.columns]
            self._cache[ticker] = (raw, datetime.now())
            return raw
        except Exception:
            return None

    @staticmethod
    def _momentum_score(df: pd.DataFrame) -> float:
        """Composite score: ret_1m (40%) + ret_3m (30%) + RS (30%)."""
        c      = df["close"].dropna()
        ret_1m = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) >= 22 else 0
        ret_3m = float(c.iloc[-1] / c.iloc[-63] - 1) if len(c) >= 63 else 0
        trend  = 1 if c.iloc[-1] > c.ewm(span=20).mean().iloc[-1] else -1
        return round(ret_1m * 40 + ret_3m * 30 + trend * 0.3, 4)
