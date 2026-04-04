from screener.stock_screener import StockScreener
from options.analyzer import OptionsAnalyzer
from utils.logger import logger


class ScannerAgent:
    """Agent that scans markets and identifies trading opportunities."""

    def __init__(self):
        self.screener = StockScreener()
        self.options_analyzer = OptionsAnalyzer()

    def run_morning_scan(self) -> dict:
        """Run pre-market/morning scan for the day's opportunities."""
        logger.info("ScannerAgent: Running morning scan...")

        intraday_picks = self.screener.get_intraday_picks(count=10)
        swing_picks = self.screener.get_swing_trade_picks(count=10)
        option_trades = self.options_analyzer.find_best_option_trades(count=5)

        logger.info(
            f"ScannerAgent: Found {len(intraday_picks)} intraday, "
            f"{len(swing_picks)} swing, {len(option_trades)} option opportunities"
        )

        return {
            "intraday": intraday_picks,
            "swing": swing_picks,
            "options": option_trades,
        }

    def run_live_scan(self) -> dict:
        """Run a quick scan during market hours."""
        logger.info("ScannerAgent: Running live scan...")
        intraday = self.screener.get_intraday_picks(count=5)
        return {"intraday": intraday}

    def get_top_10_daily(self) -> list[dict]:
        """Get top 10 stocks for today's trading."""
        logger.info("ScannerAgent: Getting top 10 daily picks...")
        intraday = self.screener.get_intraday_picks(count=5)
        swing = self.screener.get_swing_trade_picks(count=5)

        combined = []
        for pick in intraday:
            pick["trade_type"] = "INTRADAY"
            combined.append(pick)
        for pick in swing:
            pick["trade_type"] = "SWING"
            combined.append(pick)

        return combined[:10]
