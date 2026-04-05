"""Bond yield signals — India 10Y G-Sec drives sector rotation in NSE."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

_CACHE_TTL = 3600

# Yahoo Finance tickers for Indian bond proxies
_BOND_TICKERS = {
    "india_10y":  "^IRX",        # closest proxy (US 13w); use India ETF below
    "india_gsec": "0P0001BRTM.BO",  # SBI 10Y G-Sec ETF
    "us_10y":     "^TNX",        # US 10Y (global risk driver)
    "india_vix":  "^INDIAVIX",
}

# Sector sensitivity to rising yields
_SECTOR_YIELD_SENSITIVITY = {
    "BANK":   +1.5,   # Banks benefit from rising rates (NIM expansion)
    "FMCG":   -1.5,   # High-PE defensives hurt by rising rates
    "IT":     -1.2,   # Tech/growth hurt by rising rates
    "REALTY": -2.0,   # Most sensitive — high leverage
    "ENERGY": +0.5,   # Moderate positive
    "AUTO":   -0.8,   # Consumer credit cost rises
    "PHARMA": -0.5,   # Mild negative
    "METAL":  +0.3,   # Commodities slight positive
}

_SYMBOL_SECTOR = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT",
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "KOTAKBANK": "BANK", "AXISBANK": "BANK",
    "SBIN": "BANK", "BAJFINANCE": "BANK",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY",
    "TATASTEEL": "METAL", "JSWSTEEL": "METAL",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO",
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA",
    "DLF": "REALTY",
}


class BondYieldSignals:
    """
    Uses Indian G-Sec yield movements to generate sector rotation signals.

    Rising yields → rotate OUT of IT/FMCG/Realty INTO Banks/Metals
    Falling yields → rotate INTO IT/FMCG, OUT of Banks
    """

    def __init__(self):
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._yield_trend: str = "STABLE"
        self._yield_level: float = 7.0   # default 7% for India 10Y

    # ------------------------------------------------------------------
    def refresh(self, period: str = "3mo"):
        """Fetch yield data and compute trend."""
        df = self._fetch("us_10y", period)   # US 10Y as proxy for global rates
        if df is not None and len(df) >= 20:
            c = df["close"].dropna()
            ret_1m = float(c.iloc[-1] / c.iloc[max(0, len(c)-22)] - 1)
            if ret_1m > 0.02:      self._yield_trend = "RISING"
            elif ret_1m < -0.02:   self._yield_trend = "FALLING"
            else:                  self._yield_trend = "STABLE"
            self._yield_level = float(c.iloc[-1])
            logger.info(f"BondYield: trend={self._yield_trend}, level={self._yield_level:.2f}")

    def get_yield_signal(self) -> dict:
        return {
            "trend":       self._yield_trend,
            "level":       round(self._yield_level, 3),
            "nse_impact":  "NEGATIVE" if self._yield_trend == "RISING" else
                           "POSITIVE" if self._yield_trend == "FALLING" else "NEUTRAL",
            "beneficiary_sectors": self._get_beneficiaries(),
            "hurt_sectors":        self._get_hurt_sectors(),
        }

    def get_symbol_signal(self, symbol: str) -> dict:
        """Return yield-based signal for a specific symbol."""
        sector  = _SYMBOL_SECTOR.get(symbol, "")
        sens    = _SECTOR_YIELD_SENSITIVITY.get(sector, 0.0)

        if self._yield_trend == "RISING":
            bias = "BUY" if sens > 0 else "SELL" if sens < -0.5 else "NEUTRAL"
        elif self._yield_trend == "FALLING":
            bias = "BUY" if sens < 0 else "SELL" if sens > 0.5 else "NEUTRAL"
        else:
            bias = "NEUTRAL"

        return {
            "symbol":    symbol,
            "sector":    sector,
            "yield_bias": bias,
            "sensitivity": sens,
            "score_boost": round(
                -sens * (1 if self._yield_trend == "RISING" else
                         -1 if self._yield_trend == "FALLING" else 0), 2
            ),
        }

    def composite_score_boost(self, symbol: str) -> float:
        """Return score adjustment [-3, +3] based on yield sensitivity."""
        sig = self.get_symbol_signal(symbol)
        return round(float(np.clip(sig["score_boost"], -3, 3)), 2)

    def _get_beneficiaries(self) -> list[str]:
        if self._yield_trend == "RISING":
            return [s for s, v in _SECTOR_YIELD_SENSITIVITY.items() if v > 0]
        if self._yield_trend == "FALLING":
            return [s for s, v in _SECTOR_YIELD_SENSITIVITY.items() if v < 0]
        return []

    def _get_hurt_sectors(self) -> list[str]:
        if self._yield_trend == "RISING":
            return [s for s, v in _SECTOR_YIELD_SENSITIVITY.items() if v < -1]
        if self._yield_trend == "FALLING":
            return [s for s, v in _SECTOR_YIELD_SENSITIVITY.items() if v > 1]
        return []

    def _fetch(self, key: str, period: str) -> pd.DataFrame | None:
        cached = self._cache.get(key)
        if cached:
            df, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return df
        try:
            import yfinance as yf
            ticker = _BOND_TICKERS.get(key, "^TNX")
            raw    = yf.download(ticker, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
            if raw is None or len(raw) < 10:
                return None
            raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                           for c in raw.columns]
            self._cache[key] = (raw, datetime.now())
            return raw
        except Exception:
            return None
