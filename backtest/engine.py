import pandas as pd
import numpy as np
from datetime import datetime
from data.fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from analysis.patterns import PatternDetector
from config.settings import TradingConfig, MarketConfig
from utils.logger import logger
from utils.helpers import format_currency


class BacktestEngine:
    """Backtests the trading strategy against historical data."""

    def __init__(
        self,
        capital: float = None,
        max_risk_per_trade: float = None,
        stop_loss_multiplier: float = 1.5,
        target_multiplier: float = 2.0,
        max_positions: int = None,
        daily_profit_target: float = None,
        daily_loss_limit: float = None,
        min_composite_score: float = 20,
        min_rr_ratio: float = 1.5,
    ):
        self.initial_capital = capital or TradingConfig.MAX_CAPITAL
        self.max_risk_per_trade = max_risk_per_trade or TradingConfig.MAX_RISK_PER_TRADE
        self.stop_loss_multiplier = stop_loss_multiplier
        self.target_multiplier = target_multiplier
        self.max_positions = max_positions or TradingConfig.MAX_POSITIONS
        self.daily_profit_target = daily_profit_target or TradingConfig.DAILY_PROFIT_TARGET
        self.daily_loss_limit = daily_loss_limit or TradingConfig.MAX_DAILY_LOSS
        self.min_composite_score = min_composite_score
        self.min_rr_ratio = min_rr_ratio

        self.fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.pattern_detector = PatternDetector()

    def run(
        self,
        symbols: list[str] = None,
        period: str = "1y",
        trade_type: str = "intraday",
    ) -> dict:
        """Run backtest across symbols for the given period."""
        if symbols is None:
            symbols = MarketConfig.NIFTY_50_SYMBOLS[:20]  # Top 20 for speed

        logger.info(
            f"Backtest: {len(symbols)} symbols, period={period}, "
            f"type={trade_type}, capital={format_currency(self.initial_capital)}"
        )

        all_trades: list[dict] = []
        symbol_results: dict[str, dict] = {}

        for symbol in symbols:
            trades = self._backtest_symbol(symbol, period, trade_type)
            if trades:
                all_trades.extend(trades)
                symbol_results[symbol] = self._summarize_trades(trades)

        if not all_trades:
            logger.warning("Backtest: No trades generated")
            return {"error": "No trades generated. Try more symbols or a longer period."}

        # Sort all trades by date
        all_trades.sort(key=lambda t: t["entry_date"])

        # Simulate portfolio with daily limits
        portfolio_result = self._simulate_portfolio(all_trades)

        # Overall summary
        summary = self._generate_summary(portfolio_result, symbol_results)
        return summary

    def _backtest_symbol(self, symbol: str, period: str, trade_type: str) -> list[dict]:
        """Backtest strategy on a single symbol."""
        df = self.fetcher.get_historical_data(symbol, period=period, interval="1d")
        if df is None or len(df) < 60:
            return []

        df = self.analyzer.compute_all_indicators(df)
        trades = []

        # We need at least 50 bars of history before we start trading
        lookback = 50

        i = lookback
        while i < len(df):
            row = df.iloc[i]
            score = row.get("composite_score", 0)
            atr = row.get("atr", 0)

            if atr <= 0 or pd.isna(atr):
                i += 1
                continue

            # Check entry signal
            direction = None
            if score >= self.min_composite_score:
                direction = "BUY"
            elif score <= -self.min_composite_score:
                direction = "SELL"

            if direction is None:
                i += 1
                continue

            # Additional filters
            adx = row.get("adx", 0)
            volume_ratio = row.get("volume_ratio", 0)
            if pd.isna(adx) or adx < 20:
                i += 1
                continue
            if pd.isna(volume_ratio) or volume_ratio < 1.0:
                i += 1
                continue

            entry_price = row["close"]
            if direction == "BUY":
                stop_loss = entry_price - self.stop_loss_multiplier * atr
                target = entry_price + self.target_multiplier * atr
            else:
                stop_loss = entry_price + self.stop_loss_multiplier * atr
                target = entry_price - self.target_multiplier * atr

            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio < self.min_rr_ratio:
                i += 1
                continue

            # Position sizing
            risk_per_share = abs(entry_price - stop_loss)
            quantity = int(self.max_risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            if quantity < 1:
                i += 1
                continue

            # Cap at 30% of capital
            max_qty = int((self.initial_capital * 0.3) / entry_price)
            quantity = min(quantity, max_qty)
            if quantity < 1:
                i += 1
                continue

            # Simulate trade outcome using future bars
            trade = self._simulate_trade(
                df, i, symbol, direction, entry_price, stop_loss, target,
                quantity, score, atr, rr_ratio,
            )
            if trade:
                trades.append(trade)
                # Skip bars while in trade
                i = trade["_exit_bar_idx"] + 1
            else:
                i += 1

        return trades

    def _simulate_trade(
        self, df, entry_idx, symbol, direction, entry_price,
        stop_loss, target, quantity, score, atr, rr_ratio,
    ) -> dict | None:
        """Simulate a trade from entry_idx forward."""
        entry_date = df.index[entry_idx]
        max_hold = 10 if "swing" in str(score) else 5  # Max hold days

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            bar = df.iloc[j]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            if direction == "BUY":
                # Check stop loss hit (low touches SL)
                if low <= stop_loss:
                    pnl = (stop_loss - entry_price) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, stop_loss, quantity, pnl,
                        "Stop Loss", score, atr, rr_ratio, j,
                    )
                # Check target hit (high touches target)
                if high >= target:
                    pnl = (target - entry_price) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, target, quantity, pnl,
                        "Target Hit", score, atr, rr_ratio, j,
                    )
            else:
                if high >= stop_loss:
                    pnl = (entry_price - stop_loss) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, stop_loss, quantity, pnl,
                        "Stop Loss", score, atr, rr_ratio, j,
                    )
                if low <= target:
                    pnl = (entry_price - target) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, target, quantity, pnl,
                        "Target Hit", score, atr, rr_ratio, j,
                    )

        # Time-based exit at last bar
        last_idx = min(entry_idx + max_hold, len(df) - 1)
        exit_price = df.iloc[last_idx]["close"]
        if direction == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        return self._make_trade_record(
            symbol, direction, entry_date, df.index[last_idx],
            entry_price, exit_price, quantity, pnl,
            "Time Exit", score, atr, rr_ratio, last_idx,
        )

    def _make_trade_record(
        self, symbol, direction, entry_date, exit_date,
        entry_price, exit_price, quantity, pnl,
        exit_reason, score, atr, rr_ratio, exit_bar_idx,
    ) -> dict:
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "quantity": quantity,
            "pnl": round(pnl, 2),
            "pnl_pct": round((pnl / (entry_price * quantity)) * 100, 2),
            "exit_reason": exit_reason,
            "composite_score": round(score, 1),
            "atr": round(atr, 2),
            "rr_ratio": round(rr_ratio, 2),
            "hold_days": (exit_date - entry_date).days,
            "_exit_bar_idx": exit_bar_idx,
        }

    def _simulate_portfolio(self, trades: list[dict]) -> dict:
        """Simulate trades with daily P&L limits and capital tracking."""
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        equity_curve = [{"date": trades[0]["entry_date"], "equity": capital}]

        daily_pnl: dict[str, float] = {}
        accepted_trades = []
        rejected_trades = []
        open_count = 0

        for trade in trades:
            date_key = trade["entry_date"].strftime("%Y-%m-%d")

            # Check daily limits
            day_pnl = daily_pnl.get(date_key, 0.0)
            if day_pnl >= self.daily_profit_target:
                rejected_trades.append({**trade, "reject_reason": "Daily target reached"})
                continue
            if day_pnl <= -self.daily_loss_limit:
                rejected_trades.append({**trade, "reject_reason": "Daily loss limit"})
                continue

            # Check capital
            trade_cost = trade["entry_price"] * trade["quantity"]
            if trade_cost > capital * 0.5:
                rejected_trades.append({**trade, "reject_reason": "Insufficient capital"})
                continue

            # Execute trade
            capital += trade["pnl"]
            daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + trade["pnl"]
            accepted_trades.append(trade)

            # Track equity curve
            equity_curve.append({
                "date": trade["exit_date"],
                "equity": round(capital, 2),
            })

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "trades": accepted_trades,
            "rejected_trades": rejected_trades,
            "final_capital": round(capital, 2),
            "equity_curve": equity_curve,
            "max_drawdown": round(max_drawdown, 2),
            "daily_pnl": daily_pnl,
        }

    def _summarize_trades(self, trades: list[dict]) -> dict:
        """Summarize trades for a single symbol."""
        if not trades:
            return {}
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        }

    def _generate_summary(self, portfolio: dict, symbol_results: dict) -> dict:
        trades = portfolio["trades"]
        if not trades:
            return {"error": "No accepted trades after portfolio simulation"}

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Daily stats
        daily_pnl = portfolio["daily_pnl"]
        profitable_days = sum(1 for v in daily_pnl.values() if v > 0)
        losing_days = sum(1 for v in daily_pnl.values() if v <= 0)
        avg_daily_pnl = np.mean(list(daily_pnl.values())) if daily_pnl else 0
        best_day = max(daily_pnl.values()) if daily_pnl else 0
        worst_day = min(daily_pnl.values()) if daily_pnl else 0

        # By exit reason
        target_hits = len([t for t in trades if t["exit_reason"] == "Target Hit"])
        sl_hits = len([t for t in trades if t["exit_reason"] == "Stop Loss"])
        time_exits = len([t for t in trades if t["exit_reason"] == "Time Exit"])

        # Streaks
        streak, max_win_streak, max_lose_streak = 0, 0, 0
        current_streak_type = None
        for t in trades:
            if t["pnl"] > 0:
                if current_streak_type == "win":
                    streak += 1
                else:
                    streak = 1
                    current_streak_type = "win"
                max_win_streak = max(max_win_streak, streak)
            else:
                if current_streak_type == "lose":
                    streak += 1
                else:
                    streak = 1
                    current_streak_type = "lose"
                max_lose_streak = max(max_lose_streak, streak)

        # Top and bottom performers
        top_symbols = sorted(
            symbol_results.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True,
        )

        return {
            "summary": {
                "initial_capital": self.initial_capital,
                "final_capital": portfolio["final_capital"],
                "net_pnl": round(total_pnl, 2),
                "return_pct": round((total_pnl / self.initial_capital) * 100, 2),
                "total_trades": len(trades),
                "rejected_trades": len(portfolio["rejected_trades"]),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(trades) * 100, 1),
                "profit_factor": round(profit_factor, 2),
                "max_drawdown_pct": portfolio["max_drawdown"],
                "avg_win": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
                "avg_loss": round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
                "largest_win": round(max(t["pnl"] for t in trades), 2),
                "largest_loss": round(min(t["pnl"] for t in trades), 2),
                "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
            },
            "exit_reasons": {
                "target_hit": target_hits,
                "stop_loss": sl_hits,
                "time_exit": time_exits,
            },
            "daily_stats": {
                "trading_days": len(daily_pnl),
                "profitable_days": profitable_days,
                "losing_days": losing_days,
                "avg_daily_pnl": round(avg_daily_pnl, 2),
                "best_day": round(best_day, 2),
                "worst_day": round(worst_day, 2),
            },
            "streaks": {
                "max_win_streak": max_win_streak,
                "max_lose_streak": max_lose_streak,
            },
            "top_symbols": [
                {"symbol": s, **r} for s, r in top_symbols[:5]
            ],
            "bottom_symbols": [
                {"symbol": s, **r} for s, r in top_symbols[-5:]
            ],
            "trades": trades,
            "equity_curve": portfolio["equity_curve"],
        }
