import pandas as pd
import numpy as np
from data.fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from config.settings import TradingConfig
from utils.logger import logger


class OptionsAnalyzer:
    """Analyzes options chains and identifies high-probability option trades."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()

    def analyze_options(self, symbol: str) -> dict | None:
        chain = self.fetcher.get_options_chain(symbol)
        if chain is None:
            logger.warning(f"No options chain available for {symbol}")
            return None

        # Get stock technical signals
        daily = self.fetcher.get_historical_data(symbol, period="3mo")
        if daily is None:
            return None
        daily = self.analyzer.compute_all_indicators(daily)
        signals = self.analyzer.get_latest_signals(daily)

        current_price = signals["close"]
        direction = "BULLISH" if signals["composite_score"] > 20 else "BEARISH" if signals["composite_score"] < -20 else "NEUTRAL"

        calls = chain["calls"]
        puts = chain["puts"]

        recommendations = []

        if direction == "BULLISH":
            # Buy ATM/slightly OTM Call
            call_rec = self._find_best_call(calls, current_price, "BUY")
            if call_rec:
                recommendations.append(call_rec)

            # Sell OTM Put (cash-secured)
            put_rec = self._find_best_put(puts, current_price, "SELL")
            if put_rec:
                recommendations.append(put_rec)

        elif direction == "BEARISH":
            # Buy ATM/slightly OTM Put
            put_rec = self._find_best_put(puts, current_price, "BUY")
            if put_rec:
                recommendations.append(put_rec)

            # Sell OTM Call (covered)
            call_rec = self._find_best_call(calls, current_price, "SELL")
            if call_rec:
                recommendations.append(call_rec)

        else:
            # Iron Condor or Straddle for neutral
            straddle = self._straddle_recommendation(calls, puts, current_price)
            if straddle:
                recommendations.append(straddle)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "direction": direction,
            "expiry": chain["expiry"],
            "composite_score": signals["composite_score"],
            "recommendations": recommendations,
        }

    def _find_best_call(self, calls: pd.DataFrame, current_price: float, action: str) -> dict | None:
        try:
            if calls.empty:
                return None

            if action == "BUY":
                # Find ATM or slightly OTM call
                calls = calls.copy()
                calls["distance"] = abs(calls["strike"] - current_price)
                best = calls.nsmallest(3, "distance")
                # Prefer slight OTM
                otm = best[best["strike"] >= current_price]
                if not otm.empty:
                    pick = otm.iloc[0]
                else:
                    pick = best.iloc[0]

                return {
                    "action": "BUY CALL",
                    "strike": float(pick["strike"]),
                    "premium": float(pick.get("lastPrice", pick.get("ask", 0))),
                    "iv": float(pick.get("impliedVolatility", 0)) * 100,
                    "oi": int(pick.get("openInterest", 0)),
                    "volume": int(pick.get("volume", 0)),
                    "reason": "Bullish signal - Buy ATM Call for directional play",
                }
            else:
                # Sell OTM call for premium
                otm = calls[calls["strike"] > current_price * 1.03]
                if otm.empty:
                    return None
                # Pick highest OI for liquidity
                pick = otm.nlargest(1, "openInterest").iloc[0] if "openInterest" in otm.columns else otm.iloc[0]
                return {
                    "action": "SELL CALL",
                    "strike": float(pick["strike"]),
                    "premium": float(pick.get("lastPrice", pick.get("bid", 0))),
                    "iv": float(pick.get("impliedVolatility", 0)) * 100,
                    "oi": int(pick.get("openInterest", 0)),
                    "reason": "Bearish signal - Sell OTM Call for premium collection",
                }
        except Exception as e:
            logger.error(f"Error finding best call: {e}")
            return None

    def _find_best_put(self, puts: pd.DataFrame, current_price: float, action: str) -> dict | None:
        try:
            if puts.empty:
                return None

            if action == "BUY":
                puts = puts.copy()
                puts["distance"] = abs(puts["strike"] - current_price)
                best = puts.nsmallest(3, "distance")
                otm = best[best["strike"] <= current_price]
                if not otm.empty:
                    pick = otm.iloc[0]
                else:
                    pick = best.iloc[0]

                return {
                    "action": "BUY PUT",
                    "strike": float(pick["strike"]),
                    "premium": float(pick.get("lastPrice", pick.get("ask", 0))),
                    "iv": float(pick.get("impliedVolatility", 0)) * 100,
                    "oi": int(pick.get("openInterest", 0)),
                    "reason": "Bearish signal - Buy ATM Put for directional play",
                }
            else:
                otm = puts[puts["strike"] < current_price * 0.97]
                if otm.empty:
                    return None
                pick = otm.nlargest(1, "openInterest").iloc[0] if "openInterest" in otm.columns else otm.iloc[0]
                return {
                    "action": "SELL PUT",
                    "strike": float(pick["strike"]),
                    "premium": float(pick.get("lastPrice", pick.get("bid", 0))),
                    "iv": float(pick.get("impliedVolatility", 0)) * 100,
                    "oi": int(pick.get("openInterest", 0)),
                    "reason": "Bullish signal - Sell OTM Put for premium collection",
                }
        except Exception as e:
            logger.error(f"Error finding best put: {e}")
            return None

    def _straddle_recommendation(self, calls: pd.DataFrame, puts: pd.DataFrame, current_price: float) -> dict | None:
        try:
            calls = calls.copy()
            calls["distance"] = abs(calls["strike"] - current_price)
            atm_call = calls.nsmallest(1, "distance").iloc[0]

            puts = puts.copy()
            puts["distance"] = abs(puts["strike"] - current_price)
            atm_put = puts.nsmallest(1, "distance").iloc[0]

            total_premium = float(atm_call.get("lastPrice", 0)) + float(atm_put.get("lastPrice", 0))

            return {
                "action": "SELL STRADDLE" if total_premium > current_price * 0.04 else "BUY STRADDLE",
                "call_strike": float(atm_call["strike"]),
                "put_strike": float(atm_put["strike"]),
                "total_premium": round(total_premium, 2),
                "reason": "Neutral market - Straddle strategy for range-bound movement",
            }
        except Exception as e:
            logger.error(f"Error in straddle analysis: {e}")
            return None

    def find_best_option_trades(self, symbols: list[str] | None = None, count: int = 5) -> list[dict]:
        if symbols is None:
            symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY",
                        "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "TATAMOTORS"]

        all_trades = []
        for symbol in symbols:
            result = self.analyze_options(symbol)
            if result and result["recommendations"]:
                for rec in result["recommendations"]:
                    all_trades.append({
                        "symbol": symbol,
                        "expiry": result["expiry"],
                        "direction": result["direction"],
                        **rec,
                    })

        return all_trades[:count]
