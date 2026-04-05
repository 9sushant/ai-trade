"""Live paper trading loop — real-time signal → order → track P&L."""
from __future__ import annotations
import time
import threading
from datetime import datetime, date, time as dtime
from utils.logger import logger


class LiveTrader:
    """
    End-to-end live paper trading loop.

    Flow:
      1. StreamingDataHandler polls real-time prices (every 5s)
      2. On each tick: compute indicators → generate signal → apply filters
      3. If signal passes all filters → place paper order via KiteTrader
      4. Monitor open positions for SL/target/trailing stop
      5. Send Telegram alerts for every action
      6. Save all trades to SQLite database
      7. Update ML online learner after each trade closes
    """

    def __init__(
        self,
        symbols: list[str] = None,
        capital: float = 50_000,
        max_positions: int = 5,
        poll_interval: int = 30,    # seconds between signal scans
        paper_trade: bool = True,
    ):
        from config.settings import MarketConfig, TradingConfig
        self.symbols       = symbols or MarketConfig.NIFTY_50_SYMBOLS[:20]
        self.capital       = capital
        self.max_positions = max_positions
        self.poll_interval = poll_interval
        self.paper_trade   = paper_trade
        self._running      = False

        # State
        self._open_positions: dict[str, dict] = {}
        self._daily_pnl    = 0.0
        self._peak_capital = capital
        self._trade_count  = 0

        # Components (lazy init)
        self._stream     = None
        self._broker     = None
        self._db         = None
        self._notifier   = None
        self._screener   = None
        self._risk       = None
        self._online_ml  = None

    # ------------------------------------------------------------------
    def start(self):
        """Start the live trading loop."""
        logger.info("LiveTrader: initializing components...")
        self._init_components()
        self._running = True

        # Start streaming
        self._stream.start_polling_stream(interval_seconds=5)

        # Start position monitor in background
        t = threading.Thread(target=self._position_monitor_loop, daemon=True)
        t.start()

        logger.info(f"LiveTrader: started — {len(self.symbols)} symbols, "
                    f"capital=₹{self.capital:,.0f}, paper={self.paper_trade}")

        if self._notifier:
            self._notifier.send_message(
                f"LiveTrader started\nSymbols: {len(self.symbols)}\n"
                f"Capital: ₹{self.capital:,.0f}\nMode: {'PAPER' if self.paper_trade else 'LIVE'}"
            )

        # Main signal scan loop
        self._signal_scan_loop()

    def stop(self):
        """Gracefully stop trading."""
        self._running = False
        if self._stream:
            self._stream.stop()
        logger.info("LiveTrader: stopped")

    # ------------------------------------------------------------------
    def _signal_scan_loop(self):
        """Main loop: scan for signals every poll_interval seconds."""
        while self._running:
            try:
                if not self._is_market_open():
                    time.sleep(60)
                    continue

                if self._check_daily_limits():
                    time.sleep(300)
                    continue

                self._scan_and_trade()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error(f"LiveTrader signal loop error: {exc}")

            time.sleep(self.poll_interval)

    def _scan_and_trade(self):
        """Scan all symbols for signals and place orders."""
        if len(self._open_positions) >= self.max_positions:
            return

        try:
            from screener.stock_screener import StockScreener
            signals = self._screener.scan_signals(self.symbols)
            for signal in signals:
                if len(self._open_positions) >= self.max_positions:
                    break
                sym = signal.get("symbol", "")
                if sym in self._open_positions:
                    continue
                self._evaluate_and_enter(signal)
        except Exception as exc:
            logger.error(f"LiveTrader scan error: {exc}")

    def _evaluate_and_enter(self, signal: dict):
        """Run all filters and enter trade if passes."""
        symbol    = signal.get("symbol", "")
        direction = signal.get("direction", "")
        score     = signal.get("composite_score", 0)

        if not symbol or not direction:
            return

        # Risk checks
        if self._risk and self._risk.should_halt_trading():
            logger.warning(f"LiveTrader: trading halted (drawdown limit)")
            return

        entry_price = signal.get("entry_price", 0)
        stop_loss   = signal.get("stop_loss", 0)
        target      = signal.get("target_1", 0)

        if entry_price <= 0 or stop_loss <= 0:
            return

        # Position sizing
        risk_per_share = abs(entry_price - stop_loss)
        risk_capital   = self.capital * 0.01   # 1% risk per trade
        if self._risk:
            risk_capital *= self._risk.get_risk_multiplier()

        quantity = max(1, int(risk_capital / risk_per_share)) if risk_per_share > 0 else 1
        quantity = min(quantity, int(self.capital * 0.25 / entry_price))  # max 25% of capital

        # Place order
        try:
            order_id = self._broker.place_order(
                symbol=symbol, direction=direction,
                quantity=quantity, order_type="MARKET",
                stop_loss=stop_loss, target=target,
            )
            position = {
                "symbol":      symbol,
                "direction":   direction,
                "entry_price": entry_price,
                "quantity":    quantity,
                "stop_loss":   stop_loss,
                "target":      target,
                "order_id":    order_id,
                "entry_time":  datetime.now().isoformat(),
                "score":       score,
            }
            self._open_positions[symbol] = position

            if self._notifier:
                self._notifier.send_trade_opened(
                    symbol, direction, entry_price, stop_loss, target, quantity
                )
            if self._db:
                self._db.save_signal({**signal, "action": "ENTRY"})

            logger.info(f"LiveTrader: ENTERED {direction} {symbol} @ ₹{entry_price} "
                        f"qty={quantity} SL={stop_loss:.2f}")
        except Exception as exc:
            logger.error(f"LiveTrader: order failed {symbol}: {exc}")

    def _position_monitor_loop(self):
        """Background: monitor open positions for exits."""
        while self._running:
            try:
                for symbol, pos in list(self._open_positions.items()):
                    tick = self._stream.get_latest(symbol) if self._stream else None
                    if tick is None:
                        continue
                    self._check_exit(symbol, pos, tick)
            except Exception as exc:
                logger.debug(f"LiveTrader monitor error: {exc}")
            time.sleep(5)

    def _check_exit(self, symbol: str, pos: dict, tick: dict):
        """Check if position should be exited."""
        price     = tick.get("last_price", 0)
        direction = pos["direction"]
        sl        = pos["stop_loss"]
        target    = pos["target"]

        exit_reason = None
        if direction == "BUY":
            if price <= sl:       exit_reason = "Stop Loss"
            elif price >= target: exit_reason = "Target Hit"
        else:
            if price >= sl:       exit_reason = "Stop Loss"
            elif price <= target: exit_reason = "Target Hit"

        if exit_reason:
            self._close_position(symbol, pos, price, exit_reason)

    def _close_position(self, symbol: str, pos: dict, exit_price: float, reason: str):
        """Close a position and record the trade."""
        qty   = pos["quantity"]
        entry = pos["entry_price"]
        dir_  = pos["direction"]

        if dir_ == "BUY":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        self._daily_pnl += pnl
        self.capital    += pnl
        self._trade_count += 1
        if self.capital > self._peak_capital:
            self._peak_capital = self.capital
        if self._risk:
            self._risk.update_equity(self.capital)

        if self._broker:
            self._broker.close_position(symbol, qty, dir_)

        if self._notifier:
            self._notifier.send_trade_closed(
                symbol, dir_, entry, exit_price, qty, pnl, reason
            )

        if self._db:
            self._db.save_trade({
                "symbol": symbol, "direction": dir_,
                "entry_price": entry, "exit_price": exit_price,
                "quantity": qty, "pnl": pnl, "exit_reason": reason,
                "entry_date": pos["entry_time"],
                "exit_date": datetime.now().isoformat(),
            })

        del self._open_positions[symbol]
        logger.info(f"LiveTrader: CLOSED {dir_} {symbol} @ ₹{exit_price:.2f} "
                    f"PnL=₹{pnl:.2f} ({reason})")

    # ------------------------------------------------------------------
    def _check_daily_limits(self) -> bool:
        """Return True if daily limits reached (stop trading)."""
        from config.settings import TradingConfig
        if self._daily_pnl >= TradingConfig.DAILY_PROFIT_TARGET:
            logger.info("LiveTrader: daily profit target reached")
            return True
        if self._daily_pnl <= -TradingConfig.MAX_DAILY_LOSS:
            logger.warning("LiveTrader: daily loss limit reached")
            return True
        return False

    @staticmethod
    def _is_market_open() -> bool:
        now = datetime.now().time()
        return dtime(9, 15) <= now <= dtime(15, 30) and date.today().weekday() < 5

    def _init_components(self):
        from data.streaming import StreamingDataHandler
        from broker.kite_trader import KiteTrader
        from risk.dynamic_risk import DynamicRiskManager
        from screener.stock_screener import StockScreener

        self._stream   = StreamingDataHandler(self.symbols)
        self._broker   = KiteTrader(paper_trade=self.paper_trade)
        self._risk     = DynamicRiskManager()
        self._risk.set_initial(self.capital)
        self._screener = StockScreener()

        try:
            from data.database import TradeDatabase
            self._db = TradeDatabase()
        except Exception:
            pass

        try:
            from notifications.telegram_bot import TelegramNotifier
            self._notifier = TelegramNotifier()
        except Exception:
            pass

    def get_status(self) -> dict:
        """Return current trading status."""
        return {
            "running":         self._running,
            "open_positions":  len(self._open_positions),
            "daily_pnl":       round(self._daily_pnl, 2),
            "capital":         round(self.capital, 2),
            "trade_count":     self._trade_count,
            "positions":       list(self._open_positions.keys()),
        }
