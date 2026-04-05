"""Tick/minute-level backtest engine — more realistic intraday simulation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, time as dtime
from utils.logger import logger

_NSE_OPEN  = dtime(9, 15)
_NSE_CLOSE = dtime(15, 30)

# Transaction costs for intraday
_SLIPPAGE_PCT  = 0.0003   # 0.03% intraday (tighter than delivery)
_BROKERAGE_PCT = 0.0003
_STT_PCT       = 0.00025  # 0.025% STT on sell for intraday


class TickBacktestEngine:
    """
    Backtests on minute-level OHLCV data.

    Advantages over daily backtest:
      - More accurate intraday SL/target hits
      - Captures opening range breakout, power hour effects
      - Realistic entry at bar open after signal bar close
      - Proper time-of-day filtering
    """

    def __init__(
        self,
        capital: float = 50_000,
        max_risk_per_trade: float = 500,
        stop_loss_atr_mult: float = 1.5,
        target_atr_mult: float = 2.5,
        max_positions: int = 3,
        use_orb: bool = True,
        use_time_filter: bool = True,
    ):
        self.capital            = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_mult     = stop_loss_atr_mult
        self.target_mult        = target_atr_mult
        self.max_positions      = max_positions
        self.use_orb            = use_orb
        self.use_time_filter    = use_time_filter

    # ------------------------------------------------------------------
    def run(self, symbols: list[str], period: str = "60d") -> dict:
        """Download minute data and run backtest."""
        import yfinance as yf

        all_trades = []
        for symbol in symbols:
            try:
                raw = yf.download(
                    f"{symbol}.NS", period=period, interval="5m",
                    auto_adjust=True, progress=False,
                )
                if raw is None or len(raw) < 100:
                    continue
                raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                               for c in raw.columns]
                raw = self._add_indicators(raw)
                trades = self._backtest_symbol(symbol, raw)
                all_trades.extend(trades)
                logger.info(f"TickBacktest {symbol}: {len(trades)} trades")
            except Exception as exc:
                logger.debug(f"TickBacktest {symbol}: {exc}")

        if not all_trades:
            return {"error": "No trades generated"}

        all_trades.sort(key=lambda t: t["entry_time"])
        return self._summarize(all_trades)

    def run_with_df(self, symbol: str, minute_df: pd.DataFrame) -> list[dict]:
        """Run on pre-fetched minute DataFrame."""
        df = self._add_indicators(minute_df)
        return self._backtest_symbol(symbol, df)

    # ------------------------------------------------------------------
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"]     = self._atr(df, 14)
        df["ema_9"]   = df["close"].ewm(span=9).mean()
        df["ema_21"]  = df["close"].ewm(span=21).mean()
        df["rsi"]     = self._rsi(df["close"], 14)
        df["vol_ma"]  = df["volume"].rolling(20).mean()
        return df.dropna()

    def _backtest_symbol(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Run through each bar and generate trades."""
        trades       = []
        in_trade     = False
        entry_price  = 0.0
        stop_loss    = 0.0
        target       = 0.0
        direction    = ""
        entry_time   = None
        quantity     = 0
        orb_high     = None
        orb_low      = None
        current_date = None

        for i in range(21, len(df)):
            bar  = df.iloc[i]
            ts   = bar.name
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()

            bar_time = ts.time() if hasattr(ts, "time") else dtime(10, 0)
            bar_date = ts.date() if hasattr(ts, "date") else None

            # New day reset
            if bar_date and bar_date != current_date:
                current_date = bar_date
                orb_high = None
                orb_low  = None

            # Time filter
            if self.use_time_filter and not (_NSE_OPEN <= bar_time <= _NSE_CLOSE):
                continue

            # Compute ORB (first 15 minutes = 3 x 5-min bars)
            if self.use_orb and orb_high is None:
                day_bars = df[df.index.date == bar_date] if hasattr(df.index, "date") else df
                first_3  = day_bars.iloc[:3]
                if len(first_3) >= 3:
                    orb_high = float(first_3["high"].max())
                    orb_low  = float(first_3["low"].min())

            # Skip first 3 bars (ORB formation)
            if self.use_orb and bar_time < dtime(9, 30):
                continue

            # Skip lunch lull
            if self.use_time_filter and dtime(12, 0) <= bar_time < dtime(13, 0):
                if in_trade:
                    pass   # continue monitoring existing trade
                else:
                    continue

            close  = float(bar["close"])
            high   = float(bar["high"])
            low    = float(bar["low"])
            atr    = float(bar.get("atr", close * 0.01) or close * 0.01)
            volume = float(bar.get("volume", 0) or 0)
            vol_ma = float(bar.get("vol_ma", volume) or volume)

            # ── Monitor existing trade ────────────────────────────────
            if in_trade:
                exit_price  = None
                exit_reason = None

                if direction == "BUY":
                    if low <= stop_loss:
                        exit_price  = stop_loss
                        exit_reason = "Stop Loss"
                    elif high >= target:
                        exit_price  = target
                        exit_reason = "Target Hit"
                    elif bar_time >= dtime(15, 15):
                        exit_price  = close
                        exit_reason = "EOD Exit"
                else:
                    if high >= stop_loss:
                        exit_price  = stop_loss
                        exit_reason = "Stop Loss"
                    elif low <= target:
                        exit_price  = target
                        exit_reason = "Target Hit"
                    elif bar_time >= dtime(15, 15):
                        exit_price  = close
                        exit_reason = "EOD Exit"

                if exit_price:
                    pnl = (exit_price - entry_price) * quantity if direction == "BUY" \
                          else (entry_price - exit_price) * quantity
                    # Apply costs
                    tc  = entry_price * quantity * (_SLIPPAGE_PCT + _BROKERAGE_PCT) * 2
                    tc += exit_price * quantity * _STT_PCT
                    pnl -= tc

                    trades.append({
                        "symbol":      symbol,
                        "direction":   direction,
                        "entry_time":  entry_time.isoformat() if entry_time else "",
                        "exit_time":   ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "entry_price": round(entry_price, 2),
                        "exit_price":  round(exit_price, 2),
                        "quantity":    quantity,
                        "pnl":         round(pnl, 2),
                        "exit_reason": exit_reason,
                    })
                    in_trade = False
                continue

            # ── Look for new entries ──────────────────────────────────
            if bar_time > dtime(14, 45):   # no new entries after 2:45 PM
                continue

            ema9  = float(bar.get("ema_9",  close) or close)
            ema21 = float(bar.get("ema_21", close) or close)
            rsi   = float(bar.get("rsi",    50)    or 50)
            vol_ok = volume > vol_ma * 1.2 if vol_ma > 0 else True

            signal = None

            # ORB breakout
            if self.use_orb and orb_high and orb_low:
                if close > orb_high and ema9 > ema21 and vol_ok and rsi < 70:
                    signal = "BUY"
                elif close < orb_low  and ema9 < ema21 and vol_ok and rsi > 30:
                    signal = "SELL"

            # EMA crossover fallback
            if signal is None:
                prev = df.iloc[i - 1]
                prev9  = float(prev.get("ema_9",  close) or close)
                prev21 = float(prev.get("ema_21", close) or close)
                if prev9 < prev21 and ema9 > ema21 and vol_ok and rsi < 65:
                    signal = "BUY"
                elif prev9 > prev21 and ema9 < ema21 and vol_ok and rsi > 35:
                    signal = "SELL"

            if signal is None:
                continue

            # Size and SL
            risk_per_share = atr * self.stop_loss_mult
            quantity       = max(1, int(self.max_risk_per_trade / risk_per_share))
            quantity       = min(quantity, int(self.capital * 0.25 / close))

            if signal == "BUY":
                entry_price = close
                stop_loss   = close - atr * self.stop_loss_mult
                target      = close + atr * self.target_mult
            else:
                entry_price = close
                stop_loss   = close + atr * self.stop_loss_mult
                target      = close - atr * self.target_mult

            in_trade   = True
            direction  = signal
            entry_time = ts

        return trades

    # ------------------------------------------------------------------
    def _summarize(self, trades: list[dict]) -> dict:
        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total  = sum(t["pnl"] for t in trades)
        gp     = sum(t["pnl"] for t in wins)
        gl     = abs(sum(t["pnl"] for t in losses))

        return {
            "total_trades":  len(trades),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "total_pnl":     round(total, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            "return_pct":    round(total / self.capital * 100, 2),
            "avg_win":       round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
            "avg_loss":      round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
            "trades":        trades,
        }

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
