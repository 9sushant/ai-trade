"""FinBERT-based news sentiment for NSE stocks."""
from __future__ import annotations
import re
import time
import hashlib
from datetime import datetime, timedelta
from utils.logger import logger

_CACHE_TTL = 3600   # 1 hour


class NewsSentimentAnalyzer:
    """
    Fetches recent news headlines for a stock and scores them.

    Priority:
      1. FinBERT (transformers) — financial domain BERT
      2. VADER — fast lexicon-based fallback
      3. Keyword heuristic — always available
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, datetime]] = {}
        self._pipeline = None
        self._vader    = None
        self._init_models()

    def _init_models(self):
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,          # CPU
                top_k=None,
            )
            logger.info("NewsSentiment: FinBERT loaded")
        except Exception:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
                logger.info("NewsSentiment: VADER fallback loaded")
            except Exception:
                logger.info("NewsSentiment: keyword heuristic only")

    # ------------------------------------------------------------------
    def get_score(self, symbol: str) -> float:
        """Return sentiment score in [-1, +1]. Positive = bullish."""
        cached = self._cache.get(symbol)
        if cached:
            score, ts = cached
            if (datetime.now() - ts).seconds < _CACHE_TTL:
                return score

        headlines = self._fetch_headlines(symbol)
        if not headlines:
            return 0.0

        score = self._score_headlines(headlines)
        self._cache[symbol] = (score, datetime.now())
        return score

    def get_signal(self, symbol: str) -> dict:
        """Return detailed sentiment dict."""
        score = self.get_score(symbol)
        return {
            "symbol":          symbol,
            "sentiment_score": round(score, 3),
            "sentiment":       "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL",
            "should_buy":      score > 0.15,
            "should_avoid":    score < -0.20,
        }

    def composite_score_boost(self, symbol: str) -> float:
        """Return a score adjustment in [-10, +10] for composite signal."""
        return round(self.get_score(symbol) * 10, 2)

    # ------------------------------------------------------------------
    def _fetch_headlines(self, symbol: str) -> list[str]:
        headlines = []
        # Try yfinance news
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            news   = ticker.news or []
            for item in news[:10]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
        except Exception:
            pass

        # Try RSS / Google News as fallback
        if not headlines:
            try:
                headlines = self._fetch_google_news(symbol)
            except Exception:
                pass

        return headlines[:15]

    def _fetch_google_news(self, symbol: str) -> list[str]:
        import requests
        query   = f"{symbol} NSE stock"
        url     = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        resp    = requests.get(url, timeout=8,
                               headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
        return [t for t in titles if symbol.lower() not in t.lower()[:5]][:10]

    # ------------------------------------------------------------------
    def _score_headlines(self, headlines: list[str]) -> float:
        scores = []
        for h in headlines:
            scores.append(self._score_one(h))
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _score_one(self, text: str) -> float:
        if self._pipeline:
            try:
                results = self._pipeline(text[:512])[0]
                label_score = {r["label"]: r["score"] for r in results}
                return label_score.get("positive", 0) - label_score.get("negative", 0)
            except Exception:
                pass

        if self._vader:
            try:
                return self._vader.polarity_scores(text)["compound"]
            except Exception:
                pass

        return self._keyword_score(text)

    @staticmethod
    def _keyword_score(text: str) -> float:
        text = text.lower()
        bull = ["profit", "growth", "beat", "strong", "upgrade", "buy",
                "positive", "gain", "record", "surge", "rally", "outperform",
                "revenue up", "raise", "expansion", "acquisition"]
        bear = ["loss", "decline", "miss", "weak", "downgrade", "sell",
                "negative", "fall", "warning", "crash", "fraud", "default",
                "lawsuit", "cut", "layoff", "probe", "penalty"]
        b = sum(1 for w in bull if w in text)
        s = sum(1 for w in bear if w in text)
        total = b + s
        if total == 0:
            return 0.0
        return (b - s) / total
