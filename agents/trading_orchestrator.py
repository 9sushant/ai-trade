import time
from datetime import datetime
from broker.angel_one import AngelOneBroker
from agents.scanner_agent import ScannerAgent
from agents.risk_agent import RiskAgent
from agents.executor_agent import ExecutorAgent
from config.settings import TradingConfig
from utils.logger import logger
from utils.helpers import is_market_open, is_pre_market, format_currency


class TradingOrchestrator:
    """Main orchestrator that coordinates all trading agents."""

    def __init__(self, auto_trade: bool = False):
        self.broker = AngelOneBroker()
        self.scanner = ScannerAgent()
        self.risk_agent = RiskAgent(self.broker)
        self.executor = ExecutorAgent(self.broker, self.risk_agent)
        self.auto_trade = auto_trade
        self.morning_scan_done = False
        self.scan_results = {}

    def initialize(self) -> bool:
        """Login to broker and prepare for trading."""
        logger.info("Initializing Trading Orchestrator...")
        if not self.broker.login():
            logger.error("Failed to login to Angel One. Check credentials.")
            return False
        logger.info("Trading Orchestrator initialized successfully!")
        return True

    def run_morning_routine(self) -> dict:
        """Run the morning scan and prepare the day's trade plan."""
        logger.info("=" * 60)
        logger.info("MORNING ROUTINE - Scanning markets...")
        logger.info("=" * 60)

        self.scan_results = self.scanner.run_morning_scan()
        self.morning_scan_done = True

        summary = {
            "intraday_picks": len(self.scan_results.get("intraday", [])),
            "swing_picks": len(self.scan_results.get("swing", [])),
            "option_trades": len(self.scan_results.get("options", [])),
            "top_intraday": self.scan_results.get("intraday", [])[:5],
            "top_swing": self.scan_results.get("swing", [])[:5],
        }

        logger.info(f"Morning scan complete: {summary['intraday_picks']} intraday, "
                     f"{summary['swing_picks']} swing, {summary['option_trades']} options")

        return summary

    def run_trading_loop(self, interval_seconds: int = 300):
        """Main trading loop - runs during market hours."""
        logger.info("Starting trading loop...")

        while True:
            try:
                if not is_market_open():
                    if datetime.now().hour >= 16:
                        logger.info("Market closed. Running end-of-day routine.")
                        self.run_eod_routine()
                        break
                    logger.info("Waiting for market to open...")
                    time.sleep(60)
                    continue

                # Check risk limits
                risk = self.risk_agent.get_risk_summary()
                if risk["target_reached"]:
                    logger.info(f"Daily target reached! P&L: {format_currency(risk['daily_pnl'])}")
                    break
                if risk["loss_limit_hit"]:
                    logger.info(f"Daily loss limit hit! P&L: {format_currency(risk['daily_pnl'])}")
                    break

                # Check exits for existing positions
                exit_signals = self.risk_agent.check_exit_conditions()
                for signal in exit_signals:
                    logger.info(f"Exit signal: {signal['symbol']} - {signal['reason']}")
                    if self.auto_trade:
                        self.executor.execute_exit(signal)

                # Look for new opportunities if we have capacity
                can_trade, reason = self.risk_agent.can_take_trade()
                if can_trade:
                    live_scan = self.scanner.run_live_scan()
                    intraday = live_scan.get("intraday", [])
                    if intraday and self.auto_trade:
                        self.executor.execute_batch(intraday[:2], auto_execute=True)

                logger.info(f"Loop tick - P&L: {format_currency(risk['daily_pnl'])} | "
                           f"Positions: {risk['open_positions']}/{risk['max_positions']}")

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Trading loop interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)

    def run_eod_routine(self):
        """End of day summary and cleanup."""
        logger.info("=" * 60)
        logger.info("END OF DAY ROUTINE")
        logger.info("=" * 60)

        risk = self.risk_agent.get_risk_summary()
        logger.info(f"Daily P&L: {format_currency(risk['daily_pnl'])}")
        logger.info(f"  Realized: {format_currency(risk['realized_pnl'])}")
        logger.info(f"  Unrealized: {format_currency(risk['unrealized_pnl'])}")
        logger.info(f"  Open Positions: {risk['open_positions']}")

        return risk

    def get_recommendations(self) -> dict:
        """Get current trading recommendations without executing."""
        scan = self.scanner.run_morning_scan()
        top_10 = self.scanner.get_top_10_daily()

        recommendations = {
            "top_10_picks": top_10,
            "intraday_opportunities": scan.get("intraday", []),
            "swing_opportunities": scan.get("swing", []),
            "option_trades": scan.get("options", []),
        }

        # Validate each through risk agent (without broker connection)
        for pick in top_10:
            entry = pick.get("entry", pick.get("close", 0))
            sl = pick.get("stop_loss", entry * 0.985)
            target = pick.get("target", pick.get("target_1", entry * 1.025))
            risk = abs(entry - sl)
            reward = abs(target - entry)
            pick["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0
            pick["suggested_qty"] = self.risk_agent.calculate_position_size(entry, sl) if entry > 0 and sl > 0 else 0
            pick["potential_profit"] = round(pick["suggested_qty"] * reward, 2)
            pick["potential_loss"] = round(pick["suggested_qty"] * risk, 2)

        return recommendations

    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down Trading Orchestrator...")
        try:
            self.broker.logout()
        except Exception:
            pass
        logger.info("Shutdown complete.")
