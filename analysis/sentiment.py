"""
Sentiment analysis using Yahoo Finance news headlines and keyword scoring.
For live trading — not used in backtesting (no historical news archive).
"""
import re
import numpy as np
import yfinance as yf
from datetime import datetime
from utils.logger import logger


class SentimentAnalyzer:
    """
    Scores recent news sentiment for a stock on a scale of -15 to +15.

    Method:
    - Fetches up to 10 recent news items from Yahoo Finance.
    - Scores each headline + summary with a financial keyword lexicon.
    - Applies a time-decay weight (news older than 72 h → low weight).
    - Returns a weighted average mapped to [-15, +15].

    Positive score → bullish news flow.
    Negative score → bearish news flow.
    Zero          → neutral / no news available.
    """

    # Cache TTL: 1 h (news changes frequently)
    _CACHE_TTL = 3_600

    # Financial keyword lexicons
    _BULLISH = [
        "profit", "record profit", "beat", "earnings beat", "upgraded", "outperform",
        "surge", "rally", "breakout", "strong growth", "expansion", "order win",
        "contract", "acquisition", "partnership", "dividend", "buyback", "guidance raised",
        "positive", "robust", "recovery", "high", "peak", "rebound", "upgrade",
        "buy", "overweight", "target raised", "revenue growth", "margin expansion",
        "debt reduction", "promoter buying", "stake buy", "fii buying",
    ]
    _BEARISH = [
        "loss", "miss", "earnings miss", "downgrade", "underperform", "sell",
        "decline", "fall", "weak", "slowdown", "guidance cut", "profit warning",
        "fraud", "scam", "raid", "investigation", "penalty", "fine", "lawsuit",
        "default", "debt concern", "leverage", "layoff", "job cut", "plant shutdown",
        "resignation", "exit", "stake sale", "promoter sell", "fii selling",
        "below estimate", "disappoints", "crash", "plunge", "concern", "risk",
    ]

    def __init__(self):
        self._cache: dict[str, tuple[list, datetime]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_news(self, symbol: str) -> list[dict]:
        cached = self._cache.get(symbol)
        if cached:
            items, ts = cached
            if (datetime.now() - ts).seconds < self._CACHE_TTL:
                return items
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            news = ticker.news or []
            self._cache[symbol] = (news, datetime.now())
            return news
        except Exception as exc:
            logger.debug(f"News fetch failed for {symbol}: {exc}")
            return []

    def _score_text(self, text: str) -> float:
        """Return raw sentiment score in [-1, +1] for a piece of text."""
        text = text.lower()
        bullish = sum(1 for kw in self._BULLISH if kw in text)
        bearish = sum(1 for kw in self._BEARISH if kw in text)
        total = bullish + bearish
        if total == 0:
            return 0.0
        return (bullish - bearish) / total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, symbol: str) -> float:
        """Return sentiment score in [-15, +15]."""
        news = self._fetch_news(symbol)
        if not news:
            return 0.0

        now_ts = datetime.now().timestamp()
        weighted_scores: list[float] = []

        for item in news[:10]:
            title   = item.get("title", "")
            summary = item.get("summary", item.get("description", ""))
            text    = f"{title} {summary}"

            raw = self._score_text(text)

            # Time-decay: full weight within 24 h, decays linearly to 0.1 at 72 h
            pub_ts    = item.get("providerPublishTime", now_ts)
            age_hours = (now_ts - pub_ts) / 3_600
            weight    = max(0.1, 1.0 - age_hours / 72.0)

            weighted_scores.append(raw * weight)

        if not weighted_scores:
            return 0.0

        avg = float(np.mean(weighted_scores))
        return float(np.clip(avg * 15, -15, 15))

    def label(self, symbol: str) -> str:
        s = self.score(symbol)
        if s >= 6:
            return "VERY BULLISH"
        if s >= 2:
            return "BULLISH"
        if s >= -2:
            return "NEUTRAL"
        if s >= -6:
            return "BEARISH"
        return "VERY BEARISH"

    def get_headlines(self, symbol: str, n: int = 5) -> list[str]:
        """Return the n most recent news headlines."""
        news = self._fetch_news(symbol)
        return [item.get("title", "") for item in news[:n]]
