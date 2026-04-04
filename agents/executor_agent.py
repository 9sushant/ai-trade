from broker.angel_one import AngelOneBroker
from agents.risk_agent import RiskAgent
from utils.logger import logger
from utils.helpers import round_to_tick


class ExecutorAgent:
    """Agent that executes validated trades via Angel One."""

    def __init__(self, broker: AngelOneBroker, risk_agent: RiskAgent):
        self.broker = broker
        self.risk_agent = risk_agent

    def execute_trade(self, trade: dict, auto_execute: bool = False) -> dict | None:
        """Execute a single trade after risk validation."""
        # Validate through risk agent
        valid, reason, adjusted = self.risk_agent.validate_trade(trade)
        if not valid:
            logger.warning(f"ExecutorAgent: Trade rejected - {reason}")
            return {"status": "rejected", "reason": reason, "trade": trade}

        symbol = adjusted["symbol"]

        if not auto_execute:
            logger.info(f"ExecutorAgent: Trade ready for {symbol} (manual mode - not auto-executing)")
            return {"status": "ready", "trade": adjusted, "message": "Awaiting manual confirmation"}

        # Look up symbol token
        sym_info = self.broker.search_symbol(symbol)
        if not sym_info:
            return {"status": "error", "reason": f"Symbol {symbol} not found on Angel One"}

        token = sym_info.get("symboltoken", "")
        exchange = sym_info.get("exchange", "NSE")
        trading_symbol = sym_info.get("tradingsymbol", symbol)

        direction = adjusted.get("direction", "BUY")
        quantity = adjusted["quantity"]
        entry_price = adjusted["entry"]
        stop_loss = adjusted["stop_loss"]
        target = adjusted["target"]

        # Try bracket order first (has built-in SL and target)
        result = self.broker.place_bracket_order(
            symbol=trading_symbol,
            token=token,
            exchange=exchange,
            transaction_type=direction,
            quantity=quantity,
            price=entry_price,
            stop_loss=stop_loss,
            target=target,
        )

        if result:
            logger.info(f"ExecutorAgent: Bracket order placed for {symbol}")
            return {"status": "executed", "order": result, "trade": adjusted}

        # Fallback to regular order + separate SL order
        logger.info(f"ExecutorAgent: Bracket order failed, trying regular + SL order")
        main_order = self.broker.place_order(
            symbol=trading_symbol,
            token=token,
            exchange=exchange,
            transaction_type=direction,
            quantity=quantity,
            price=entry_price,
            order_type="LIMIT",
            product_type="INTRADAY" if adjusted.get("trade_type") == "INTRADAY" else "DELIVERY",
        )

        if main_order:
            # Place stop loss order
            sl_direction = "SELL" if direction == "BUY" else "BUY"
            sl_order = self.broker.place_order(
                symbol=trading_symbol,
                token=token,
                exchange=exchange,
                transaction_type=sl_direction,
                quantity=quantity,
                price=round_to_tick(stop_loss),
                order_type="STOPLOSS_LIMIT",
                product_type="INTRADAY" if adjusted.get("trade_type") == "INTRADAY" else "DELIVERY",
                trigger_price=round_to_tick(stop_loss + 0.05 if sl_direction == "SELL" else stop_loss - 0.05),
            )

            return {
                "status": "executed",
                "main_order": main_order,
                "sl_order": sl_order,
                "trade": adjusted,
            }

        return {"status": "error", "reason": "Order placement failed"}

    def execute_batch(self, trades: list[dict], auto_execute: bool = False) -> list[dict]:
        """Execute multiple trades."""
        results = []
        for trade in trades:
            can_trade, reason = self.risk_agent.can_take_trade()
            if not can_trade:
                logger.info(f"ExecutorAgent: Stopping batch - {reason}")
                break

            result = self.execute_trade(trade, auto_execute=auto_execute)
            if result:
                results.append(result)

        return results

    def execute_exit(self, exit_signal: dict) -> dict | None:
        """Execute an exit for an existing position."""
        symbol = exit_signal["symbol"]
        sym_info = self.broker.search_symbol(symbol)
        if not sym_info:
            return None

        positions = self.broker.get_positions()
        for pos in positions:
            if pos.get("tradingsymbol") == symbol and int(pos.get("netqty", 0)) != 0:
                net_qty = int(pos["netqty"])
                direction = "SELL" if net_qty > 0 else "BUY"
                quantity = abs(net_qty)

                result = self.broker.place_order(
                    symbol=sym_info.get("tradingsymbol", symbol),
                    token=sym_info.get("symboltoken", ""),
                    exchange=sym_info.get("exchange", "NSE"),
                    transaction_type=direction,
                    quantity=quantity,
                    order_type="MARKET",
                    product_type=pos.get("producttype", "INTRADAY"),
                )

                if result:
                    logger.info(
                        f"ExecutorAgent: Exited {symbol} - {exit_signal.get('reason', 'Manual exit')}"
                    )
                    return {"status": "exited", "order": result, "reason": exit_signal.get("reason")}

        return None
