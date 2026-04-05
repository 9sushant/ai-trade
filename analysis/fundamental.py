"""
Fundamental analysis using yfinance data.
Scores stocks based on valuation, profitability, and financial health.
"""
import numpy as np
import yfinance as yf
from datetime import datetime
from utils.logger import logger


class FundamentalAnalyzer:
    """
    Scores stocks on fundamentals using yfinance Ticker.info.

    Scoring breakdown (total: -20 to +20):
      Valuation (P/E):     -6 to +6
      Profitability (ROE): -5 to +5
      Financial health:    -4 to +4  (debt/equity)
      Earnings growth:     -3 to +3
      Net margin:          -2 to +2

    Positive score → fundamentally strong (favours BUY).
    Negative score → fundamentally weak (favours SELL).
    """

    # Cache TTL: 24 h (fundamentals change slowly)
    _CACHE_TTL = 86_400

    def __init__(self):
        self._cache: dict[str, tuple[dict, datetime]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fundamentals(self, symbol: str) -> dict:
        """Return a flat dict of fundamental metrics for *symbol*."""
        cached = self._cache.get(symbol)
        if cached:
            data, ts = cached
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return data

        try:
            info = yf.Ticker(f"{symbol}.NS").info
            data = {
                # Valuation
                "pe_ratio":          info.get("trailingPE"),
                "pb_ratio":          info.get("priceToBook"),
                "ev_ebitda":         info.get("enterpriseToEbitda"),
                "peg_ratio":         info.get("pegRatio"),
                "forward_pe":        info.get("forwardPE"),
                # Profitability
                "roe":               info.get("returnOnEquity"),
                "roa":               info.get("returnOnAssets"),
                "net_margin":        info.get("profitMargins"),
                "operating_margin":  info.get("operatingMargins"),
                "gross_margin":      info.get("grossMargins"),
                # Financial health
                "debt_to_equity":    info.get("debtToEquity"),   # in % in yfinance
                "current_ratio":     info.get("currentRatio"),
                "quick_ratio":       info.get("quickRatio"),
                # Growth
                "revenue_growth":    info.get("revenueGrowth"),
                "earnings_growth":   info.get("earningsGrowth"),
                "eps_trailing":      info.get("trailingEps"),
                "eps_forward":       info.get("forwardEps"),
                # Meta
                "market_cap":        info.get("marketCap"),
                "sector":            info.get("sector", ""),
                "industry":          info.get("industry", ""),
                "beta":              info.get("beta"),
                # Institutional
                "institutional_pct": info.get("institutionPercentHeld"),
                "insider_pct":       info.get("insidersPercentHeld"),
            }
            self._cache[symbol] = (data, datetime.now())
            return data
        except Exception as exc:
            logger.debug(f"Fundamental fetch failed for {symbol}: {exc}")
            return {}

    def score(self, symbol: str) -> float:
        """Return a fundamental score in [-20, +20]."""
        data = self.get_fundamentals(symbol)
        if not data:
            return 0.0

        s = 0.0

        # ── Valuation: P/E ratio ──────────────────────────────── max ±6
        pe = data.get("pe_ratio")
        if pe is not None and pe > 0:
            if pe < 12:
                s += 6
            elif pe < 20:
                s += 3
            elif pe < 30:
                s += 1
            elif pe < 45:
                s -= 3
            else:
                s -= 6

        # ── Profitability: ROE ───────────────────────────────── max ±5
        roe = data.get("roe")
        if roe is not None:
            if roe > 0.25:
                s += 5
            elif roe > 0.18:
                s += 3
            elif roe > 0.12:
                s += 1
            elif roe > 0:
                s -= 1
            else:
                s -= 5

        # ── Financial health: Debt/Equity ───────────────────── max ±4
        # yfinance reports D/E as a percentage (e.g. 50 = 50%)
        de = data.get("debt_to_equity")
        if de is not None:
            if de < 20:
                s += 4
            elif de < 60:
                s += 2
            elif de < 120:
                s -= 1
            elif de < 200:
                s -= 3
            else:
                s -= 4

        # ── Earnings growth ──────────────────────────────────── max ±3
        eg = data.get("earnings_growth")
        if eg is not None:
            if eg > 0.25:
                s += 3
            elif eg > 0.10:
                s += 2
            elif eg > 0:
                s += 1
            elif eg > -0.10:
                s -= 1
            else:
                s -= 3

        # ── Net margin ───────────────────────────────────────── max ±2
        nm = data.get("net_margin")
        if nm is not None:
            if nm > 0.20:
                s += 2
            elif nm > 0.08:
                s += 1
            elif nm < 0:
                s -= 2

        return float(np.clip(s, -20, 20))

    def label(self, symbol: str) -> str:
        """Human-readable quality label for a stock."""
        s = self.score(symbol)
        if s >= 8:
            return "STRONG"
        if s >= 3:
            return "GOOD"
        if s >= -3:
            return "FAIR"
        if s >= -8:
            return "WEAK"
        return "POOR"

    def blocks_buy(self, symbol: str) -> bool:
        """True when fundamentals are too poor to support a long trade."""
        return self.score(symbol) < -8

    def blocks_sell(self, symbol: str) -> bool:
        """True when fundamentals are too strong to support a short trade."""
        return self.score(symbol) > 8
