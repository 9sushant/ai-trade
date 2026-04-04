from agents.base import BaseAgent
from data.fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from analysis.patterns import PatternDetector
from utils.logger import logger


class AnalysisAgent(BaseAgent):
    """Performs deep technical analysis on stocks and generates trade signals."""

    def __init__(self):
        super().__init__("AnalysisAgent")
        self.fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.pattern_detector = PatternDetector()
        self.signals: list[dict] = []

    def analyze_stock(self, symbol: str, trade_type: str = "intraday") -> dict | None:
        try:
            period = "3mo" if trade_type == "intraday" else "6mo"
            df = self.fetcher.get_historical_data(symbol, period=period)
            if df is None or len(df) < 30:
                return None

            df = self.analyzer.compute_all_indicators(df)
            signals = self.analyzer.get_latest_signals(df)
            patterns = self.pattern_detector.detect_all(df)

            last = df.iloc[-1]
            atr = signals.get("atr", last["close"] * 0.015)

            direction = "BUY" if signals["composite_score"] >= 20 else (
                "SELL" if signals["composite_score"] <= -20 else "NEUTRAL"
            )

            if direction == "NEUTRAL":
                return None

            entry = round(last["close"], 2)
            if direction == "BUY":
                stop_loss = round(entry - 1.5 * atr, 2)
                target_1 = round(entry + 2 * atr, 2)
                target_2 = round(entry + 3 * atr, 2)
            else:
                stop_loss = round(entry + 1.5 * atr, 2)
                target_1 = round(entry - 2 * atr, 2)
                target_2 = round(entry - 3 * atr, 2)

            risk = abs(entry - stop_loss)
            reward = abs(target_1 - entry)
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0

            return {
                "symbol": symbol,
                "trade_type": trade_type,
                "direction": direction,
                "entry": entry,
                "stop_loss": stop_loss,
                "target_1": target_1,
                "target_2": target_2,
                "rr_ratio": rr_ratio,
                "composite_score": signals["composite_score"],
                "rsi": signals["rsi"],
                "adx": signals["adx"],
                "macd_hist": signals["macd_hist"],
                "trend": signals["ema_trend"],
                "supertrend": signals["supertrend_dir"],
                "volume_ratio": signals["volume_ratio"],
                "patterns": [p["name"] for p in patterns["patterns"]],
                "support": patterns["support"],
                "resistance": patterns["resistance"],
                "recommendation": signals["recommendation"],
                "confidence": self._calc_confidence(signals, patterns),
            }
        except Exception as e:
            logger.error(f"AnalysisAgent: Error analyzing {symbol}: {e}")
            return None

    def analyze_many(self, symbols: list[str], trade_type: str = "intraday") -> list[dict]:
        results = []
        for symbol in symbols:
            result = self.analyze_stock(symbol, trade_type)
            if result and result["rr_ratio"] >= 1.5:  # Only good R:R trades
                results.append(result)
        results.sort(key=lambda x: (x["confidence"], abs(x["composite_score"])), reverse=True)
        self.signals = results
        return results

    def run_cycle(self) -> dict:
        return {"signals": self.signals}

    def _calc_confidence(self, signals: dict, patterns: dict) -> float:
        """Calculate signal confidence (0-100%)."""
        score = 0.0

        # ADX trend strength
        adx = signals.get("adx", 0)
        if adx > 40:
            score += 25
        elif adx > 25:
            score += 15
        elif adx > 20:
            score += 8

        # RSI not extreme
        rsi = signals.get("rsi", 50)
        if 35 <= rsi <= 65:
            score += 20
        elif 30 <= rsi <= 70:
            score += 10

        # Volume confirmation
        vol_ratio = signals.get("volume_ratio", 0)
        if vol_ratio > 2.0:
            score += 20
        elif vol_ratio > 1.5:
            score += 12
        elif vol_ratio > 1.2:
            score += 6

        # Trend alignment
        if signals["ema_trend"] in ("STRONG_UPTREND", "STRONG_DOWNTREND"):
            score += 20
        elif signals["ema_trend"] in ("UPTREND", "DOWNTREND"):
            score += 10

        # SuperTrend confirmation
        if signals["supertrend_dir"] != "SIDEWAYS":
            score += 15

        # Bullish/bearish patterns
        if patterns["patterns"]:
            max_strength = max(p["strength"] for p in patterns["patterns"])
            score += max_strength * 0.1

        return min(round(score, 1), 100.0)
