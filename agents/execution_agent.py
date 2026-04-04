from agents.risk_agent import RiskAgent
from broker.angel_one import AngelOneBroker
from utils.logger import logger
from utils.helpers import format_currency
import datetime


class ExecutionAgent:
    """Executes trades via Angel One broker with risk checks."""

    def __init__(self, broker: AngelOneBroker | None = None, paper_mode: bool = True):
        self.broker = broker
        self.paper_mode = paper_mode
        self.risk = RiskAgent()
        self.symbol_token_cache: dict[str, dict] = {}
        self.executed_trades: list[dict] = []

    def execute_signal(self, signal: dict) -> dict:
        """Execute a trade signal after risk approval."""
        symbol = signal["symbol"]
        direction = signal["direction"]

        # Risk check
        approval = self.risk.approve_trade(signal)
        if not approval["approved"]:
            logger.warning(f"Trade REJECTED for {symbol}: {approval['reason']}")
            return {"status": "rejected", "reason": approval["reason"]}

        quantity = approval["quantity"]
        entry = signal["entry"]
        stop_loss = signal["stop_loss"]
        target = signal["target_1"]

        if self.paper_mode:
            return self._paper_execute(signal, quantity, entry, stop_loss, target, approval)
        else:
            return self._live_execute(signal, quantity, entry, stop_loss, target, approval)

    def _paper_execute(self, signal: dict, quantity: int, entry: float,
                        stop_loss: float, target: float, approval: dict) -> dict:
        """Simulate trade execution for paper trading."""
        trade = {
            "status": "filled",
            "mode": "PAPER",
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "quantity": quantity,
            "entry_price": entry,
            "stop_loss": stop_loss,
            "target": target,
            "trade_value": approval["trade_value"],
            "max_loss": approval["max_loss"],
            "max_profit": approval["max_profit"],
            "order_id": f"PAPER-{datetime.datetime.now().strftime('%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
        }

        self.risk.record_trade_open(
            signal["symbol"], signal["direction"], quantity, entry, stop_loss, target
        )
        self.executed_trades.append(trade)

        logger.info(
            f"TRADE [PAPER]: {signal['direction']} {quantity}x {signal['symbol']} "
            f"@ {entry} | SL:{stop_loss} TGT:{target} | "
            f"Value:{format_currency(approval['trade_value'])}"
        )
        return trade

    def _live_execute(self, signal: dict, quantity: int, entry: float,
                      stop_loss: float, target: float, approval: dict) -> dict:
        """Execute live order via Angel One SmartAPI."""
        if not self.broker:
            return {"status": "error", "reason": "Broker not initialized"}

        symbol = signal["symbol"]
        exchange = "NSE"
        transaction_type = signal["direction"]  # BUY or SELL

        # Get symbol token
        token_info = self._get_symbol_token(symbol, exchange)
        if not token_info:
            return {"status": "error", "reason": f"Symbol token not found for {symbol}"}

        token = token_info.get("symboltoken", token_info.get("token", ""))

        # Place bracket order (entry + SL + target)
        result = self.broker.place_bracket_order(
            symbol=f"{symbol}-EQ",
            token=token,
            exchange=exchange,
            transaction_type=transaction_type,
            quantity=quantity,
            price=entry,
            stop_loss=stop_loss,
            target=target,
        )

        if result:
            self.risk.record_trade_open(symbol, transaction_type, quantity, entry, stop_loss, target)
            self.executed_trades.append({**result, "signal": signal})
            logger.info(f"TRADE [LIVE]: Order placed for {symbol} | ID: {result.get('order_id')}")
            return result
        else:
            return {"status": "error", "reason": "Order placement failed"}

    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> dict | None:
        if symbol in self.symbol_token_cache:
            return self.symbol_token_cache[symbol]
        if self.broker:
            info = self.broker.search_symbol(symbol, exchange)
            if info:
                self.symbol_token_cache[symbol] = info
                return info
        return None

    def monitor_positions(self):
        """Check open positions against live prices and exit if needed."""
        if not self.risk.open_positions:
            return

        for symbol, pos in list(self.risk.open_positions.items()):
            try:
                if self.paper_mode:
                    # In paper mode, fetch live price from yfinance
                    from data.fetcher import DataFetcher
                    fetcher = DataFetcher()
                    price_data = fetcher.get_live_price(symbol)
                    current_price = price_data["price"] if price_data else None
                else:
                    if self.broker:
                        current_price = self.broker.get_ltp("NSE", f"{symbol}-EQ", "")
                    else:
                        current_price = None

                if current_price:
                    exit_check = self.risk.check_exit_conditions(symbol, current_price)
                    if exit_check["exit"]:
                        self.close_position(symbol, current_price, exit_check["reason"])
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")

    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual"):
        """Close an open position."""
        if symbol not in self.risk.open_positions:
            return

        pos = self.risk.open_positions[symbol]

        if not self.paper_mode and self.broker:
            # Place market sell/buy to close
            close_direction = "SELL" if pos["direction"] == "BUY" else "BUY"
            token_info = self._get_symbol_token(symbol)
            if token_info:
                self.broker.place_order(
                    symbol=f"{symbol}-EQ",
                    token=token_info.get("symboltoken", ""),
                    exchange="NSE",
                    transaction_type=close_direction,
                    quantity=pos["quantity"],
                    order_type="MARKET",
                )

        self.risk.record_trade_close(symbol, exit_price, reason)

    def squareoff_all(self):
        """Square off all intraday positions (call before 3:15 PM)."""
        logger.info("ExecutionAgent: Squaring off all open positions...")
        for symbol in list(self.risk.open_positions.keys()):
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            price_data = fetcher.get_live_price(symbol)
            if price_data:
                self.close_position(symbol, price_data["price"], "EOD Square-off")

    def get_trade_summary(self) -> dict:
        return {
            **self.risk.get_summary(),
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "recent_trades": self.executed_trades[-10:],
        }
