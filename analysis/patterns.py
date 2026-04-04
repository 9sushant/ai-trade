import pandas as pd
import numpy as np
from utils.logger import logger


class PatternDetector:
    """Detects candlestick and chart patterns for trade signals."""

    def detect_all(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 5:
            return {"patterns": [], "support": None, "resistance": None}

        patterns = []
        patterns.extend(self._candlestick_patterns(df))
        patterns.extend(self._chart_patterns(df))

        support, resistance = self._support_resistance(df)

        return {
            "patterns": patterns,
            "support": round(support, 2) if support else None,
            "resistance": round(resistance, 2) if resistance else None,
        }

    def _candlestick_patterns(self, df: pd.DataFrame) -> list[dict]:
        patterns = []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        body = abs(last["close"] - last["open"])
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        candle_range = last["high"] - last["low"]

        if candle_range == 0:
            return patterns

        # Doji
        if body / candle_range < 0.1:
            patterns.append({"name": "Doji", "type": "reversal", "strength": 60})

        # Hammer (bullish reversal)
        if lower_wick > 2 * body and upper_wick < body * 0.5 and last["close"] < prev["close"]:
            patterns.append({"name": "Hammer", "type": "bullish", "strength": 75})

        # Shooting Star (bearish reversal)
        if upper_wick > 2 * body and lower_wick < body * 0.5 and last["close"] > prev["close"]:
            patterns.append({"name": "Shooting Star", "type": "bearish", "strength": 75})

        # Bullish Engulfing
        if (prev["close"] < prev["open"] and last["close"] > last["open"]
                and last["open"] <= prev["close"] and last["close"] >= prev["open"]):
            patterns.append({"name": "Bullish Engulfing", "type": "bullish", "strength": 80})

        # Bearish Engulfing
        if (prev["close"] > prev["open"] and last["close"] < last["open"]
                and last["open"] >= prev["close"] and last["close"] <= prev["open"]):
            patterns.append({"name": "Bearish Engulfing", "type": "bearish", "strength": 80})

        # Morning Star (3-candle)
        if len(df) >= 3:
            three_back = df.iloc[-3]
            if (three_back["close"] < three_back["open"]  # First: bearish
                    and abs(prev["close"] - prev["open"]) / (prev["high"] - prev["low"] + 0.001) < 0.3  # Second: small body
                    and last["close"] > last["open"]  # Third: bullish
                    and last["close"] > (three_back["open"] + three_back["close"]) / 2):
                patterns.append({"name": "Morning Star", "type": "bullish", "strength": 85})

        return patterns

    def _chart_patterns(self, df: pd.DataFrame) -> list[dict]:
        patterns = []
        if len(df) < 20:
            return patterns

        closes = df["close"].values[-20:]
        highs = df["high"].values[-20:]
        lows = df["low"].values[-20:]

        # Double bottom detection (simplified)
        min_idx = np.argmin(lows)
        if 5 < min_idx < 15:
            left_min = np.min(lows[:min_idx])
            right_min = np.min(lows[min_idx + 3:]) if min_idx + 3 < len(lows) else float("inf")
            if abs(left_min - right_min) / left_min < 0.02:
                patterns.append({"name": "Double Bottom", "type": "bullish", "strength": 80})

        # Higher highs and higher lows (uptrend confirmation)
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        if all(recent_highs[i] >= recent_highs[i - 1] for i in range(1, len(recent_highs))):
            if all(recent_lows[i] >= recent_lows[i - 1] for i in range(1, len(recent_lows))):
                patterns.append({"name": "Higher Highs & Higher Lows", "type": "bullish", "strength": 70})

        # Lower highs and lower lows (downtrend confirmation)
        if all(recent_highs[i] <= recent_highs[i - 1] for i in range(1, len(recent_highs))):
            if all(recent_lows[i] <= recent_lows[i - 1] for i in range(1, len(recent_lows))):
                patterns.append({"name": "Lower Highs & Lower Lows", "type": "bearish", "strength": 70})

        return patterns

    def _support_resistance(self, df: pd.DataFrame) -> tuple:
        if len(df) < 20:
            return None, None

        recent = df.tail(50) if len(df) >= 50 else df
        pivots_high = []
        pivots_low = []

        for i in range(2, len(recent) - 2):
            if recent["high"].iloc[i] > recent["high"].iloc[i - 1] and recent["high"].iloc[i] > recent["high"].iloc[i + 1]:
                if recent["high"].iloc[i] > recent["high"].iloc[i - 2] and recent["high"].iloc[i] > recent["high"].iloc[i + 2]:
                    pivots_high.append(recent["high"].iloc[i])

            if recent["low"].iloc[i] < recent["low"].iloc[i - 1] and recent["low"].iloc[i] < recent["low"].iloc[i + 1]:
                if recent["low"].iloc[i] < recent["low"].iloc[i - 2] and recent["low"].iloc[i] < recent["low"].iloc[i + 2]:
                    pivots_low.append(recent["low"].iloc[i])

        current_price = df["close"].iloc[-1]
        resistance = min((p for p in pivots_high if p > current_price), default=None)
        support = max((p for p in pivots_low if p < current_price), default=None)

        return support, resistance
