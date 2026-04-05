import pandas as pd
from data.fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from analysis.patterns import PatternDetector
from analysis.fundamental import FundamentalAnalyzer
from analysis.sentiment import SentimentAnalyzer
from analysis.quant import QuantAnalyzer
from config.settings import MarketConfig, TradingConfig
from utils.logger import logger


class StockScreener:
    """Screens and ranks stocks to find top trading opportunities."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.analyzer = TechnicalAnalyzer()
        self.pattern_detector = PatternDetector()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quant_analyzer = QuantAnalyzer()

    def scan_all_stocks(self, symbols: list[str] | None = None) -> list[dict]:
        if symbols is None:
            symbols = MarketConfig.ALL_SCAN_SYMBOLS

        logger.info(f"Scanning {len(symbols)} stocks...")
        results = []

        for symbol in symbols:
            try:
                result = self._analyze_stock(symbol)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")

        # Sort by absolute composite score (strongest signals first)
        results.sort(key=lambda x: abs(x["composite_score"]), reverse=True)
        return results

    def get_top_buy_picks(self, count: int = 10) -> list[dict]:
        all_stocks = self.scan_all_stocks()
        buy_picks = [
            s for s in all_stocks
            if s["composite_score"] >= 20 and s["recommendation"] in ("BUY", "STRONG BUY")
        ]
        return buy_picks[:count]

    def get_top_sell_picks(self, count: int = 10) -> list[dict]:
        all_stocks = self.scan_all_stocks()
        sell_picks = [
            s for s in all_stocks
            if s["composite_score"] <= -20 and s["recommendation"] in ("SELL", "STRONG SELL")
        ]
        return sell_picks[:count]

    def get_swing_trade_picks(self, count: int = 10) -> list[dict]:
        """Find stocks suitable for swing trading (2-10 day holds)."""
        symbols = MarketConfig.ALL_SCAN_SYMBOLS
        results = []

        for symbol in symbols:
            try:
                daily = self.fetcher.get_historical_data(symbol, period="6mo", interval="1d")
                if daily is None or len(daily) < 50:
                    continue

                daily = self.analyzer.compute_all_indicators(daily)
                signals = self.analyzer.get_latest_signals(daily)
                patterns = self.pattern_detector.detect_all(daily)

                last = daily.iloc[-1]
                score = signals.get("composite_score", 0)

                # Swing trade criteria
                rsi = signals.get("rsi", 50)
                adx = signals.get("adx", 0)
                vol_ratio = signals.get("volume_ratio", 0)

                swing_score = 0

                # RSI pullback in uptrend
                if signals["ema_trend"] in ("STRONG_UPTREND", "UPTREND") and 35 <= rsi <= 50:
                    swing_score += 30

                # Strong trend with ADX
                if adx > 25:
                    swing_score += 20

                # Volume confirmation
                if vol_ratio > 1.2:
                    swing_score += 15

                # Near support
                if patterns["support"] and last["close"] <= patterns["support"] * 1.02:
                    swing_score += 20

                # Bullish patterns
                bullish_patterns = [p for p in patterns["patterns"] if p["type"] == "bullish"]
                if bullish_patterns:
                    swing_score += max(p["strength"] for p in bullish_patterns) * 0.2

                # SuperTrend bullish
                if signals["supertrend_dir"] == "BULLISH":
                    swing_score += 15

                if swing_score >= 30:
                    atr = signals.get("atr", 0)
                    results.append({
                        "symbol": symbol,
                        "close": signals["close"],
                        "swing_score": round(swing_score, 1),
                        "entry": round(last["close"], 2),
                        "stop_loss": round(last["close"] - 1.5 * atr, 2),
                        "target_1": round(last["close"] + 2 * atr, 2),
                        "target_2": round(last["close"] + 3 * atr, 2),
                        "rsi": rsi,
                        "adx": adx,
                        "trend": signals["ema_trend"],
                        "supertrend": signals["supertrend_dir"],
                        "patterns": [p["name"] for p in patterns["patterns"]],
                        "support": patterns["support"],
                        "resistance": patterns["resistance"],
                        "recommendation": "SWING BUY",
                        "hold_days": "3-7 days",
                    })

            except Exception as e:
                logger.debug(f"Error in swing scan for {symbol}: {e}")

        results.sort(key=lambda x: x["swing_score"], reverse=True)
        return results[:count]

    def get_intraday_picks(self, count: int = 10) -> list[dict]:
        """Find stocks for intraday trading."""
        symbols = MarketConfig.NIFTY_50_SYMBOLS  # Focus on liquid stocks for intraday
        results = []

        for symbol in symbols:
            try:
                # Use both daily and intraday data
                daily = self.fetcher.get_historical_data(symbol, period="3mo", interval="1d")
                if daily is None or len(daily) < 30:
                    continue

                daily = self.analyzer.compute_all_indicators(daily)
                signals = self.analyzer.get_latest_signals(daily)
                patterns = self.pattern_detector.detect_all(daily)

                last = daily.iloc[-1]
                score = signals.get("composite_score", 0)

                # Intraday criteria: strong momentum + volume
                intraday_score = 0

                # Strong momentum
                if abs(score) >= 30:
                    intraday_score += 25

                # High volume
                if signals.get("volume_ratio", 0) > 1.5:
                    intraday_score += 20

                # Strong trend (ADX > 25)
                if signals.get("adx", 0) > 25:
                    intraday_score += 15

                # Clear direction (not sideways)
                if signals["ema_trend"] in ("STRONG_UPTREND", "STRONG_DOWNTREND"):
                    intraday_score += 20

                # MACD histogram growing
                if abs(signals.get("macd_hist", 0)) > 0:
                    intraday_score += 10

                # Bollinger band breakout potential
                if signals["bb_position"] in ("BELOW_LOWER", "ABOVE_UPPER"):
                    intraday_score += 10

                if intraday_score >= 35:
                    atr = signals.get("atr", 0)
                    direction = "BUY" if score > 0 else "SELL"
                    results.append({
                        "symbol": symbol,
                        "close": signals["close"],
                        "intraday_score": round(intraday_score, 1),
                        "direction": direction,
                        "entry": round(last["close"], 2),
                        "stop_loss": round(last["close"] - atr if direction == "BUY" else last["close"] + atr, 2),
                        "target": round(last["close"] + 1.5 * atr if direction == "BUY" else last["close"] - 1.5 * atr, 2),
                        "rsi": signals["rsi"],
                        "volume_ratio": signals["volume_ratio"],
                        "composite_score": score,
                        "recommendation": f"INTRADAY {direction}",
                    })

            except Exception as e:
                logger.debug(f"Error in intraday scan for {symbol}: {e}")

        results.sort(key=lambda x: x["intraday_score"], reverse=True)
        return results[:count]

    def _analyze_stock(self, symbol: str) -> dict | None:
        daily = self.fetcher.get_historical_data(symbol, period="6mo", interval="1d")
        if daily is None or len(daily) < 30:
            return None

        daily = self.analyzer.compute_all_indicators(daily)
        signals = self.analyzer.get_latest_signals(daily)
        patterns = self.pattern_detector.detect_all(daily)

        # ── Fundamental analysis ──────────────────────────────────────────
        try:
            fund_score = self.fundamental_analyzer.score(symbol)
            fund_label = self.fundamental_analyzer.label(symbol)
            fund_data  = self.fundamental_analyzer.get_fundamentals(symbol)
        except Exception:
            fund_score, fund_label, fund_data = 0.0, "N/A", {}

        # ── Sentiment analysis ────────────────────────────────────────────
        try:
            sent_score = self.sentiment_analyzer.score(symbol)
            sent_label = self.sentiment_analyzer.label(symbol)
            headlines  = self.sentiment_analyzer.get_headlines(symbol, n=3)
        except Exception:
            sent_score, sent_label, headlines = 0.0, "N/A", []

        # ── Quantitative analysis ─────────────────────────────────────────
        try:
            quant_report = self.quant_analyzer.full_report(daily)
        except Exception:
            quant_report = {}

        # ── Combined score: Technical 55% + Fundamental 20% + Sentiment 10% + Quant 15%
        tech_score  = signals.get("composite_score", 0)
        combined    = (
            0.55 * tech_score
            + 0.20 * fund_score
            + 0.10 * sent_score
            + 0.15 * quant_report.get("quant_score", 0)
        )

        return {
            "symbol": symbol,
            **signals,
            "patterns":          [p["name"] for p in patterns["patterns"]],
            "support":           patterns["support"],
            "resistance":        patterns["resistance"],
            # Fundamental
            "fund_score":        round(fund_score, 1),
            "fund_label":        fund_label,
            "pe_ratio":          fund_data.get("pe_ratio"),
            "roe":               fund_data.get("roe"),
            "debt_to_equity":    fund_data.get("debt_to_equity"),
            "earnings_growth":   fund_data.get("earnings_growth"),
            # Sentiment
            "sent_score":        round(sent_score, 1),
            "sent_label":        sent_label,
            "headlines":         headlines,
            # Quant
            "quant_score":       quant_report.get("quant_score", 0),
            "momentum_score":    quant_report.get("momentum_score", 0),
            "mean_rev_z":        quant_report.get("mean_rev_z", 0),
            "sharpe_60":         quant_report.get("sharpe_60", 0),
            "beta_60":           quant_report.get("beta_60", 1),
            # Combined
            "combined_score":    round(combined, 1),
        }
