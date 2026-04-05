"""Cross-asset signal generator — USD/INR, crude oil, gold, SGX Nifty."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

# Yahoo Finance tickers
_TICKERS = {
    "usdinr":   "USDINR=X",
    "crude":    "CL=F",
    "gold":     "GC=F",
    "sgx_nifty":"^NSEI",     # SGX Nifty not on yfinance; use Nifty as proxy
    "vix":      "^INDIAVIX",
    "dow":      "^DJI",
}

_CACHE_TTL = 3600   # 1 hour


class CrossAssetSignals:
    """
    Generates composite cross-asset signal score for NSE market direction.

    Score: -10 (very bearish) to +10 (very bullish)
    """

    def __init__(self):
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}

    # ------------------------------------------------------------------
    def get_market_score(self) -> float:
        """Overall cross-asset score for the day."""
        signals = self.get_all_signals()
        weights = {
            "usdinr_signal": -1.5,   # strong USD/INR up = bearish for NSE
            "crude_signal":  -1.0,   # crude up = cost pressure (bearish)
            "gold_signal":   -0.5,   # gold up = risk-off (bearish)
            "vix_signal":    -2.0,   # VIX up = fear (bearish)
            "dow_signal":     2.0,   # Dow up = positive sentiment
        }
        score = 0.0
        for key, weight in weights.items():
            score += signals.get(key, 0) * weight
        return round(np.clip(score, -10, 10), 2)

    def get_all_signals(self) -> dict:
        """Return individual directional signals [-1, 0, +1] per asset."""
        result = {}
        for name, ticker in _TICKERS.items():
            try:
                df = self._fetch(ticker)
                if df is not None and len(df) >= 5:
                    result[f"{name}_signal"] = self._trend_signal(df)
                    result[f"{name}_ret_1d"] = self._ret_1d(df)
                    result[f"{name}_ret_5d"] = self._ret_5d(df)
            except Exception as exc:
                logger.debug(f"CrossAsset {name}: {exc}")
        return result

    def get_usdinr_trend(self) -> str:
        """INR strengthening (USD/INR falling) = BULLISH for NSE."""
        df = self._fetch(_TICKERS["usdinr"])
        if df is None:
            return "NEUTRAL"
        sig = self._trend_signal(df)
        if sig > 0:  return "BEARISH"   # USD/INR up = INR weak = bad for NSE
        if sig < 0:  return "BULLISH"
        return "NEUTRAL"

    def get_crude_signal(self) -> str:
        df = self._fetch(_TICKERS["crude"])
        if df is None:
            return "NEUTRAL"
        ret = self._ret_5d(df)
        if ret > 0.03:   return "BEARISH"
        if ret < -0.03:  return "BULLISH"
        return "NEUTRAL"

    def get_vix_level(self) -> float:
        """Return latest India VIX level."""
        df = self._fetch(_TICKERS["vix"])
        if df is None or df.empty:
            return 15.0
        return float(df["close"].iloc[-1])

    def should_reduce_risk(self) -> bool:
        """True when cross-asset picture is risk-off."""
        score = self.get_market_score()
        vix   = self.get_vix_level()
        return score < -4 or vix > 20

    # ------------------------------------------------------------------
    def _fetch(self, ticker: str) -> pd.DataFrame | None:
        cached = self._cache.get(ticker)
        if cached:
            df, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return df
        try:
            import yfinance as yf
            raw = yf.download(ticker, period="30d", interval="1d",
                              auto_adjust=True, progress=False)
            if raw is None or len(raw) < 5:
                return None
            raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                           for c in raw.columns]
            self._cache[ticker] = (raw, datetime.now())
            return raw
        except Exception:
            return None

    @staticmethod
    def _trend_signal(df: pd.DataFrame) -> int:
        """Return +1 if trending up, -1 down, 0 flat (based on 5d vs 20d EMA)."""
        c    = df["close"].dropna()
        if len(c) < 10:
            return 0
        ema5  = c.ewm(span=5).mean().iloc[-1]
        ema20 = c.ewm(span=20).mean().iloc[-1]
        if ema5 > ema20 * 1.002:  return 1
        if ema5 < ema20 * 0.998:  return -1
        return 0

    @staticmethod
    def _ret_1d(df: pd.DataFrame) -> float:
        c = df["close"].dropna()
        if len(c) < 2:
            return 0.0
        return float(c.iloc[-1] / c.iloc[-2] - 1)

    @staticmethod
    def _ret_5d(df: pd.DataFrame) -> float:
        c = df["close"].dropna()
        if len(c) < 6:
            return 0.0
        return float(c.iloc[-1] / c.iloc[-6] - 1)
