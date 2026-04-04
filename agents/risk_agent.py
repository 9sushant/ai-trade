from config.settings import TradingConfig
from broker.angel_one import AngelOneBroker
from utils.logger import logger
from utils.helpers import calculate_quantity, round_to_tick


class RiskAgent:
    """Agent that manages risk - position sizing, stop losses, daily limits."""

    def __init__(self, broker: AngelOneBroker):
        self.broker = broker
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.open_positions = 0

    def can_take_trade(self) -> tuple[bool, str]:
        """Check if we can take a new trade based on risk rules."""
        # Check daily loss limit
        pnl = self.broker.get_pnl()
        self.daily_pnl = pnl["total_pnl"]

        if self.daily_pnl <= -TradingConfig.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached: ₹{self.daily_pnl}"

        # Check daily profit target (stop trading if target met)
        if self.daily_pnl >= TradingConfig.DAILY_PROFIT_TARGET:
            return False, f"Daily profit target reached: ₹{self.daily_pnl}"

        # Check max positions
        positions = self.broker.get_positions()
        self.open_positions = sum(1 for p in positions if int(p.get("netqty", 0)) != 0)
        if self.open_positions >= TradingConfig.MAX_POSITIONS:
            return False, f"Max positions ({TradingConfig.MAX_POSITIONS}) reached"

        return True, "OK"

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk per trade."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0

        # Risk-based sizing
        qty_by_risk = int(TradingConfig.MAX_RISK_PER_TRADE / risk_per_share)

        # Capital-based sizing (don't use more than allocated per position)
        capital_per_position = TradingConfig.MAX_CAPITAL / TradingConfig.MAX_POSITIONS
        qty_by_capital = int(capital_per_position / entry_price) if entry_price > 0 else 0

        quantity = min(qty_by_risk, qty_by_capital)
        return max(quantity, 1) if quantity > 0 else 0

    def validate_trade(self, trade: dict) -> tuple[bool, str, dict]:
        """Validate and adjust a proposed trade."""
        can_trade, reason = self.can_take_trade()
        if not can_trade:
            return False, reason, {}

        entry = trade.get("entry", 0)
        stop_loss = trade.get("stop_loss", 0)
        target = trade.get("target", trade.get("target_1", 0))

        if entry <= 0 or stop_loss <= 0:
            return False, "Invalid entry or stop loss price", {}

        # Check risk:reward ratio (minimum 1:1.5)
        risk = abs(entry - stop_loss)
        reward = abs(target - entry) if target else risk * 2
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < 1.2:
            return False, f"Poor risk:reward ratio ({rr_ratio:.1f}:1)", {}

        quantity = self.calculate_position_size(entry, stop_loss)
        if quantity == 0:
            return False, "Position size too small", {}

        potential_loss = quantity * risk
        potential_profit = quantity * reward

        adjusted_trade = {
            **trade,
            "quantity": quantity,
            "entry": round_to_tick(entry),
            "stop_loss": round_to_tick(stop_loss),
            "target": round_to_tick(target) if target else round_to_tick(entry + reward),
            "risk_amount": round(potential_loss, 2),
            "reward_amount": round(potential_profit, 2),
            "rr_ratio": round(rr_ratio, 2),
        }

        logger.info(
            f"RiskAgent: Validated {trade.get('symbol', '?')} - "
            f"Qty: {quantity}, Risk: ₹{potential_loss:.0f}, "
            f"Reward: ₹{potential_profit:.0f}, RR: {rr_ratio:.1f}"
        )

        return True, "Trade validated", adjusted_trade

    def check_exit_conditions(self) -> list[dict]:
        """Check all open positions for exit conditions."""
        exits = []
        positions = self.broker.get_positions()

        for pos in positions:
            net_qty = int(pos.get("netqty", 0))
            if net_qty == 0:
                continue

            pnl = float(pos.get("pnl", 0))
            symbol = pos.get("tradingsymbol", "")

            # Trailing stop loss check
            if pnl > 0:
                # If we're in profit, tighten the stop
                entry = float(pos.get("buyavgprice", 0)) if net_qty > 0 else float(pos.get("sellavgprice", 0))
                if entry > 0:
                    pnl_pct = (pnl / (entry * abs(net_qty))) * 100
                    if pnl_pct >= TradingConfig.TARGET_PCT:
                        exits.append({
                            "symbol": symbol,
                            "action": "EXIT",
                            "reason": f"Target reached ({pnl_pct:.1f}%)",
                            "pnl": pnl,
                        })

            # Stop loss hit
            if pnl < 0:
                entry = float(pos.get("buyavgprice", 0)) if net_qty > 0 else float(pos.get("sellavgprice", 0))
                if entry > 0:
                    loss_pct = abs(pnl / (entry * abs(net_qty))) * 100
                    if loss_pct >= TradingConfig.STOP_LOSS_PCT:
                        exits.append({
                            "symbol": symbol,
                            "action": "EXIT",
                            "reason": f"Stop loss hit ({loss_pct:.1f}%)",
                            "pnl": pnl,
                        })

        return exits

    def get_risk_summary(self) -> dict:
        pnl = self.broker.get_pnl()
        positions = self.broker.get_positions()
        open_count = sum(1 for p in positions if int(p.get("netqty", 0)) != 0)

        return {
            "daily_pnl": pnl["total_pnl"],
            "realized_pnl": pnl["realized_pnl"],
            "unrealized_pnl": pnl["unrealized_pnl"],
            "open_positions": open_count,
            "max_positions": TradingConfig.MAX_POSITIONS,
            "daily_target": TradingConfig.DAILY_PROFIT_TARGET,
            "daily_loss_limit": TradingConfig.MAX_DAILY_LOSS,
            "target_reached": pnl["total_pnl"] >= TradingConfig.DAILY_PROFIT_TARGET,
            "loss_limit_hit": pnl["total_pnl"] <= -TradingConfig.MAX_DAILY_LOSS,
        }
