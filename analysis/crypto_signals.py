"""Crypto correlation signals — BTC/ETH as risk-on/off leading indicators for NSE."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger

_CACHE_TTL = 1800   # 30 min
_CRYPTO_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


class CryptoSignals:
    """
    BTC/ETH as leading risk-on/off indicators for NSE.

    Research: BTC drops >3% often precede NSE selloffs by 12-24 hours.
    BTC surges signal risk appetite → positive for NSE equities.

    Score: -5 (risk-off / sell NSE) to +5 (risk-on / buy NSE)
    """

    def __init__(self):
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}

    # ------------------------------------------------------------------
    def get_risk_signal(self) -> dict:
        """Get overall crypto risk signal for NSE."""
        btc = self._get_data("BTC")
        eth = self._get_data("ETH")

        btc_score = self._score(btc) if btc is not None else 0.0
        eth_score = self._score(eth) if eth is not None else 0.0

        # BTC has 70% weight, ETH 30%
        combined = 0.70 * btc_score + 0.30 * eth_score

        if combined > 2:     regime = "RISK_ON"
        elif combined > 0.5: regime = "MILDLY_BULLISH"
        elif combined < -2:  regime = "RISK_OFF"
        elif combined < -0.5:regime = "MILDLY_BEARISH"
        else:                regime = "NEUTRAL"

        return {
            "btc_score":    round(btc_score,  2),
            "eth_score":    round(eth_score,  2),
            "combined":     round(combined,   2),
            "regime":       regime,
            "nse_bias":     "BULLISH" if combined > 1 else "BEARISH" if combined < -1 else "NEUTRAL",
            "reduce_risk":  combined < -2,
            "btc_ret_24h":  self._ret_24h(btc),
            "btc_ret_1w":   self._ret_1w(btc),
        }

    def get_btc_trend(self) -> str:
        btc = self._get_data("BTC")
        if btc is None:
            return "NEUTRAL"
        score = self._score(btc)
        if score > 1:    return "BULLISH"
        if score < -1:   return "BEARISH"
        return "NEUTRAL"

    def is_risk_off(self) -> bool:
        return self.get_risk_signal()["reduce_risk"]

    def composite_score_boost(self) -> float:
        """Score boost [-3, +3] based on crypto risk signal."""
        combined = self.get_risk_signal()["combined"]
        return round(float(np.clip(combined * 0.6, -3, 3)), 2)

    def get_correlation(self, nifty_df: pd.DataFrame) -> dict:
        """Compute rolling correlation between BTC and Nifty 50."""
        btc = self._get_data("BTC")
        if btc is None or nifty_df is None:
            return {"correlation": 0, "period_days": 0}

        btc_ret   = btc["close"].pct_change().dropna()
        nifty_ret = nifty_df["close"].pct_change().dropna()

        aligned = pd.concat([btc_ret.rename("btc"),
                              nifty_ret.rename("nifty")], axis=1).dropna()
        if len(aligned) < 20:
            return {"correlation": 0, "period_days": 0}

        corr = float(aligned["btc"].corr(aligned["nifty"]))
        return {
            "correlation":  round(corr, 4),
            "period_days":  len(aligned),
            "relationship": "positive" if corr > 0.2 else "negative" if corr < -0.2 else "uncorrelated",
        }

    # ------------------------------------------------------------------
    def _get_data(self, ticker_key: str) -> pd.DataFrame | None:
        ticker = _CRYPTO_TICKERS.get(ticker_key)
        if not ticker:
            return None

        cached = self._cache.get(ticker_key)
        if cached:
            df, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return df

        try:
            import yfinance as yf
            raw = yf.download(ticker, period="30d", interval="1h",
                              auto_adjust=True, progress=False)
            if raw is None or len(raw) < 24:
                return None
            raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                           for c in raw.columns]
            self._cache[ticker_key] = (raw, datetime.now())
            return raw
        except Exception as exc:
            logger.debug(f"CryptoSignals {ticker_key}: {exc}")
            return None

    def _score(self, df: pd.DataFrame) -> float:
        """Score from -5 to +5 based on momentum + trend."""
        c      = df["close"].dropna()
        ret_1d = float(c.iloc[-1] / c.iloc[max(0, len(c)-24)] - 1) if len(c) >= 24 else 0
        ret_7d = float(c.iloc[-1] / c.iloc[max(0, len(c)-168)] - 1) if len(c) >= 168 else 0
        ema_s  = c.ewm(span=12).mean().iloc[-1]
        ema_l  = c.ewm(span=48).mean().iloc[-1]
        trend  = 1 if ema_s > ema_l else -1

        # Scale: ±5% daily → ±3 score, trend adds ±1
        score  = np.clip(ret_1d / 0.05, -3, 3) + np.clip(ret_7d / 0.10, -1, 1) + trend
        return float(np.clip(score, -5, 5))

    def _ret_24h(self, df: pd.DataFrame | None) -> float:
        if df is None or len(df) < 24:
            return 0.0
        c = df["close"].dropna()
        return round(float(c.iloc[-1] / c.iloc[max(0, len(c)-24)] - 1) * 100, 2)

    def _ret_1w(self, df: pd.DataFrame | None) -> float:
        if df is None or len(df) < 168:
            return 0.0
        c = df["close"].dropna()
        return round(float(c.iloc[-1] / c.iloc[max(0, len(c)-168)] - 1) * 100, 2)
