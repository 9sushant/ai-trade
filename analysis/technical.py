import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from utils.logger import logger


class TechnicalAnalyzer:
    """Comprehensive technical analysis engine with multiple indicators."""

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 30:
            return df

        df = df.copy()

        # --- Trend Indicators ---
        # EMA (9, 21, 50, 200)
        for period in [9, 21, 50, 200]:
            if len(df) >= period:
                df[f"ema_{period}"] = EMAIndicator(df["close"], window=period).ema_indicator()

        # SMA 20
        df["sma_20"] = SMAIndicator(df["close"], window=20).sma_indicator()

        # MACD
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # ADX (trend strength)
        if len(df) >= 14:
            adx = ADXIndicator(df["high"], df["low"], df["close"])
            df["adx"] = adx.adx()
            df["adx_pos"] = adx.adx_pos()
            df["adx_neg"] = adx.adx_neg()

        # SuperTrend
        df = self._supertrend(df, period=10, multiplier=3)

        # --- Momentum Indicators ---
        # RSI
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

        # Stochastic
        stoch = StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # --- Volatility Indicators ---
        # Bollinger Bands
        bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR
        atr = AverageTrueRange(df["high"], df["low"], df["close"])
        df["atr"] = atr.average_true_range()

        # --- Volume Indicators ---
        # VWAP (only for intraday with datetime index)
        try:
            vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"])
            df["vwap"] = vwap.volume_weighted_average_price()
        except Exception:
            df["vwap"] = np.nan

        # OBV
        df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        # Volume SMA
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # --- Derived Signals ---
        df["ema_crossover"] = self._ema_crossover_signal(df)
        df["macd_signal_line"] = self._macd_signal(df)
        df["rsi_signal"] = self._rsi_signal(df)
        df["bb_signal"] = self._bb_signal(df)
        df["volume_signal"] = self._volume_signal(df)

        # Composite score (-100 to +100)
        df["composite_score"] = self._composite_score(df)

        return df

    def _supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.DataFrame:
        hl2 = (df["high"] + df["low"]) / 2
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]

        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction  # 1 = bullish, -1 = bearish
        return df

    def _ema_crossover_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if "ema_9" in df.columns and "ema_21" in df.columns:
            signal = np.where(
                (df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1)),
                1,  # Bullish crossover
                np.where(
                    (df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1)),
                    -1,  # Bearish crossover
                    0,
                ),
            )
        return pd.Series(signal, index=df.index)

    def _macd_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if "macd" in df.columns and "macd_signal" in df.columns:
            signal = np.where(
                (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)),
                1,
                np.where(
                    (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)),
                    -1,
                    0,
                ),
            )
        return pd.Series(signal, index=df.index)

    def _rsi_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if "rsi" in df.columns:
            signal = np.where(
                df["rsi"] < 30, 1,  # Oversold = Buy
                np.where(df["rsi"] > 70, -1, 0),  # Overbought = Sell
            )
        return pd.Series(signal, index=df.index)

    def _bb_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if "bb_lower" in df.columns:
            signal = np.where(
                df["close"] <= df["bb_lower"], 1,  # Near lower band = Buy
                np.where(df["close"] >= df["bb_upper"], -1, 0),  # Near upper band = Sell
            )
        return pd.Series(signal, index=df.index)

    def _volume_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if "volume_ratio" in df.columns:
            price_change = df["close"].pct_change()
            signal = np.where(
                (df["volume_ratio"] > 1.5) & (price_change > 0), 1,  # High volume + price up
                np.where(
                    (df["volume_ratio"] > 1.5) & (price_change < 0), -1, 0,
                ),
            )
        return pd.Series(signal, index=df.index)

    def _composite_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)

        weights = {
            "ema_crossover": 15,
            "macd_signal_line": 15,
            "rsi_signal": 15,
            "bb_signal": 10,
            "volume_signal": 10,
        }

        for col, weight in weights.items():
            if col in df.columns:
                score += df[col] * weight

        # SuperTrend direction
        if "supertrend_dir" in df.columns:
            score += df["supertrend_dir"] * 20

        # Trend alignment (EMA stack)
        if all(f"ema_{p}" in df.columns for p in [9, 21, 50]):
            bullish_stack = (df["ema_9"] > df["ema_21"]) & (df["ema_21"] > df["ema_50"])
            bearish_stack = (df["ema_9"] < df["ema_21"]) & (df["ema_21"] < df["ema_50"])
            score += np.where(bullish_stack, 15, np.where(bearish_stack, -15, 0))

        return score.clip(-100, 100)

    def get_latest_signals(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        signals = {
            "close": round(last["close"], 2),
            "rsi": round(last.get("rsi", 0), 2),
            "macd": round(last.get("macd", 0), 4),
            "macd_hist": round(last.get("macd_histogram", 0), 4),
            "adx": round(last.get("adx", 0), 2),
            "supertrend_dir": "BULLISH" if last.get("supertrend_dir", 0) == 1 else "BEARISH",
            "bb_position": self._bb_position(last),
            "volume_ratio": round(last.get("volume_ratio", 0), 2),
            "composite_score": round(last.get("composite_score", 0), 1),
            "ema_trend": self._ema_trend(last),
            "atr": round(last.get("atr", 0), 2),
        }

        # Generate recommendation
        score = signals["composite_score"]
        if score >= 40:
            signals["recommendation"] = "STRONG BUY"
        elif score >= 20:
            signals["recommendation"] = "BUY"
        elif score <= -40:
            signals["recommendation"] = "STRONG SELL"
        elif score <= -20:
            signals["recommendation"] = "SELL"
        else:
            signals["recommendation"] = "NEUTRAL"

        return signals

    def _bb_position(self, row) -> str:
        close = row.get("close", 0)
        upper = row.get("bb_upper", 0)
        lower = row.get("bb_lower", 0)
        middle = row.get("bb_middle", 0)
        if close >= upper:
            return "ABOVE_UPPER"
        elif close <= lower:
            return "BELOW_LOWER"
        elif close > middle:
            return "UPPER_HALF"
        else:
            return "LOWER_HALF"

    def _ema_trend(self, row) -> str:
        ema9 = row.get("ema_9", 0)
        ema21 = row.get("ema_21", 0)
        ema50 = row.get("ema_50", 0)
        if ema9 > ema21 > ema50:
            return "STRONG_UPTREND"
        elif ema9 < ema21 < ema50:
            return "STRONG_DOWNTREND"
        elif ema9 > ema21:
            return "UPTREND"
        elif ema9 < ema21:
            return "DOWNTREND"
        return "SIDEWAYS"
