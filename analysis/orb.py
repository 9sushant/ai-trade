"""Opening Range Breakout (ORB) strategy — first 15 minutes high/low breakout."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger


class ORBStrategy:
    def __init__(self, orb_minutes: int = 15, atr_multiplier_sl: float = 1.0,
                 atr_multiplier_target: float = 2.0, min_volume_ratio: float = 1.5):
        self.orb_minutes         = orb_minutes
        self.atr_multiplier_sl   = atr_multiplier_sl
        self.atr_multiplier_target = atr_multiplier_target
        self.min_volume_ratio    = min_volume_ratio

    def compute_orb(self, intraday_df: pd.DataFrame) -> dict | None:
        """Find ORB high/low from first orb_minutes of trading."""
        if intraday_df is None or intraday_df.empty:
            return None
        try:
            idx = pd.DatetimeIndex(intraday_df.index)
            if idx.tz is not None:
                idx = idx.tz_convert("Asia/Kolkata")
            market_open = idx[0].replace(hour=9, minute=15, second=0)
            cutoff      = market_open + pd.Timedelta(minutes=self.orb_minutes)
            orb_bars    = intraday_df[idx <= cutoff]
            if len(orb_bars) < 1:
                return None
            orb_high = float(orb_bars["high"].max())
            orb_low  = float(orb_bars["low"].min())
            return {
                "orb_high":    round(orb_high, 2),
                "orb_low":     round(orb_low, 2),
                "orb_range":   round(orb_high - orb_low, 2),
                "orb_midpoint": round((orb_high + orb_low) / 2, 2),
            }
        except Exception as exc:
            logger.debug(f"ORB compute error: {exc}")
            return None

    def get_signal(self, intraday_df: pd.DataFrame,
                   daily_df: pd.DataFrame = None) -> dict | None:
        """Return ORB signal dict if a valid breakout is detected."""
        orb = self.compute_orb(intraday_df)
        if orb is None:
            return None
        atr = 0.0
        if daily_df is not None and len(daily_df) > 14 and "atr" in daily_df.columns:
            atr = float(daily_df["atr"].iloc[-1])
        if atr <= 0 and orb["orb_range"] > 0:
            atr = orb["orb_range"]   # Use ORB range as ATR proxy

        # Look at bars after the ORB period
        try:
            idx = pd.DatetimeIndex(intraday_df.index)
            if idx.tz is not None:
                idx = idx.tz_convert("Asia/Kolkata")
            market_open = idx[0].replace(hour=9, minute=15, second=0)
            cutoff      = market_open + pd.Timedelta(minutes=self.orb_minutes)
            post_orb    = intraday_df[idx > cutoff]
        except Exception:
            return None

        if post_orb.empty:
            return None

        for ts, bar in post_orb.iterrows():
            high   = float(bar["high"])
            low    = float(bar["low"])
            volume = float(bar.get("volume", 0))
            vol_sma= float(bar.get("volume_sma_20", volume)) or volume
            vol_r  = volume / vol_sma if vol_sma > 0 else 1.0

            if not self.is_valid_orb(orb["orb_range"], atr):
                return None

            if high > orb["orb_high"] and vol_r >= self.min_volume_ratio:
                entry = orb["orb_high"]
                sl    = entry - self.atr_multiplier_sl * atr
                return {
                    "direction":   "BUY",
                    "entry":       round(entry, 2),
                    "stop_loss":   round(sl, 2),
                    "target_1":    round(entry + self.atr_multiplier_target * atr, 2),
                    "target_2":    round(entry + self.atr_multiplier_target * 2 * atr, 2),
                    "orb_high":    orb["orb_high"],
                    "orb_low":     orb["orb_low"],
                    "signal_time": str(ts),
                    "volume_ratio": round(vol_r, 2),
                    "strength":    round(vol_r * (high - orb["orb_high"]) / (atr or 1), 2),
                }
            if low < orb["orb_low"] and vol_r >= self.min_volume_ratio:
                entry = orb["orb_low"]
                sl    = entry + self.atr_multiplier_sl * atr
                return {
                    "direction":   "SELL",
                    "entry":       round(entry, 2),
                    "stop_loss":   round(sl, 2),
                    "target_1":    round(entry - self.atr_multiplier_target * atr, 2),
                    "target_2":    round(entry - self.atr_multiplier_target * 2 * atr, 2),
                    "orb_high":    orb["orb_high"],
                    "orb_low":     orb["orb_low"],
                    "signal_time": str(ts),
                    "volume_ratio": round(vol_r, 2),
                    "strength":    round(vol_r * (orb["orb_low"] - low) / (atr or 1), 2),
                }
        return None

    def is_valid_orb(self, orb_range: float, atr: float) -> bool:
        if atr <= 0:
            return True
        ratio = orb_range / atr
        return 0.2 <= ratio <= 2.5

    def scan_orb_signals(self, symbols: list[str], fetcher) -> list[dict]:
        results = []
        for symbol in symbols:
            try:
                intraday = fetcher.get_intraday_data(symbol)
                daily    = fetcher.get_historical_data(symbol, period="3mo", interval="1d")
                if intraday is None or intraday.empty:
                    continue
                from analysis.technical import TechnicalAnalyzer
                if daily is not None:
                    daily = TechnicalAnalyzer().compute_all_indicators(daily)
                sig = self.get_signal(intraday, daily)
                if sig:
                    sig["symbol"] = symbol
                    results.append(sig)
            except Exception as exc:
                logger.debug(f"ORB scan {symbol}: {exc}")
        results.sort(key=lambda x: x.get("strength", 0), reverse=True)
        return results
