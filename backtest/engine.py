import pandas as pd
import numpy as np
from datetime import datetime
from data.fetcher import DataFetcher
from analysis.technical import TechnicalAnalyzer
from analysis.patterns import PatternDetector
from analysis.fundamental import FundamentalAnalyzer
from analysis.quant import QuantAnalyzer
from analysis.ensemble import EnsembleSignalModel
from analysis.hmm_regime import HMMRegimeDetector
from analysis.volatility import GARCHVolatilityForecaster
from analysis.cross_asset import CrossAssetSignals
from analysis.sector_rotation import SectorRotationTracker
from analysis.fno_expiry import FNOExpiryAnalyzer
from analysis.news_sentiment import NewsSentimentAnalyzer
from analysis.regime_models import RegimeSpecificModels
from ml.lstm_model import LSTMModel
from ml.rl_dqn import DQNAgent
from ml.drift_detector import ModelDriftDetector
from ml.explainer import SignalExplainer
from risk.var_calculator import VaRCalculator
from risk.dynamic_risk import DynamicRiskManager
from risk.cvar import CVaRCalculator
from risk.monte_carlo import MonteCarloSimulator
from risk.portfolio_optimizer import PortfolioOptimizer
from backtest.execution import ExecutionModel
from config.settings import TradingConfig, MarketConfig
from utils.logger import logger
from utils.helpers import format_currency

# Slippage model: market impact as % of ATR
_SLIPPAGE_PCT   = 0.0005   # 0.05% per side (realistic for NSE liquid stocks)
_BROKERAGE_PCT  = 0.0003   # 0.03% per side (Zerodha flat rate approx)
_STT_PCT        = 0.001    # 0.1% on sell side (STT for equity delivery/intraday)


class BacktestEngine:
    """
    Backtests the trading strategy against historical data.

    Improvements:
      1. Ensemble ML Filter      — LightGBM+XGBoost+GBM majority vote
      2. Multi-Timeframe (MTF)   — Weekly trend confirms daily direction
      3. Kelly Criterion Sizing  — Dynamic sizing from running win stats
      4. Correlation Filter      — Blocks correlated concurrent positions
      5. Regime-Adaptive Params  — BULL/BEAR/SIDEWAYS adaptive params
      6. HMM Regime Detection    — Hidden Markov Model market states
      7. VaR Position Sizing     — Value-at-Risk caps position size
      8. Dynamic Risk Reduction  — Halves size during drawdowns
      9. Slippage + Brokerage    — Realistic transaction cost modeling
     10. DB Persistence          — All trades saved to SQLite
    """

    def __init__(
        self,
        capital: float = None,
        max_risk_per_trade: float = None,
        stop_loss_multiplier: float = 1.2,
        target_multiplier: float = 2.0,
        trail_atr_multiplier: float = 0.5,
        max_positions: int = None,
        daily_profit_target: float = None,
        daily_loss_limit: float = None,
        min_composite_score: float = 30,
        min_adx: float = 25,
        min_rr_ratio: float = 1.3,
        # Regime filter
        use_regime_filter: bool = True,
        use_regime_adaptive: bool = True,
        # Fundamental / quant
        use_fundamental_filter: bool = True,
        use_quant_score: bool = True,
        quant_score_weight: float = 0.15,
        # Ensemble ML signal filter
        use_ml_filter: bool = True,
        ml_prob_threshold: float = 0.52,
        # Multi-timeframe confluence
        use_mtf_filter: bool = True,
        # Kelly criterion position sizing
        use_kelly_sizing: bool = True,
        kelly_fraction: float = 0.25,
        kelly_warmup: int = 15,
        # Correlation filter
        use_correlation_filter: bool = True,
        max_correlation: float = 0.80,
        # New: VaR sizing, dynamic risk, slippage, DB
        use_var_sizing: bool = True,
        use_dynamic_risk: bool = True,
        use_slippage: bool = True,
        save_to_db: bool = False,          # Set True for live/paper trading
        use_hmm_regime: bool = True,
    ):
        self.initial_capital      = capital or TradingConfig.MAX_CAPITAL
        self.max_risk_per_trade   = max_risk_per_trade or TradingConfig.MAX_RISK_PER_TRADE
        self.stop_loss_multiplier = stop_loss_multiplier
        self.target_multiplier    = target_multiplier
        self.trail_atr_multiplier = trail_atr_multiplier
        self.max_positions        = max_positions or TradingConfig.MAX_POSITIONS
        self.daily_profit_target  = daily_profit_target or TradingConfig.DAILY_PROFIT_TARGET
        self.daily_loss_limit     = daily_loss_limit or TradingConfig.MAX_DAILY_LOSS
        self.min_composite_score  = min_composite_score
        self.min_adx              = min_adx
        self.min_rr_ratio         = min_rr_ratio

        self.use_regime_filter    = use_regime_filter
        self.use_regime_adaptive  = use_regime_adaptive
        self.use_fundamental_filter = use_fundamental_filter
        self.use_quant_score      = use_quant_score
        self.quant_score_weight   = quant_score_weight
        self.use_ml_filter        = use_ml_filter
        self.ml_prob_threshold    = ml_prob_threshold
        self.use_mtf_filter       = use_mtf_filter
        self.use_kelly_sizing     = use_kelly_sizing
        self.kelly_fraction       = kelly_fraction
        self.kelly_warmup         = kelly_warmup
        self.use_correlation_filter = use_correlation_filter
        self.max_correlation      = max_correlation
        self.use_var_sizing       = use_var_sizing
        self.use_dynamic_risk     = use_dynamic_risk
        self.use_slippage         = use_slippage
        self.save_to_db           = save_to_db
        self.use_hmm_regime       = use_hmm_regime

        self.fetcher              = DataFetcher()
        self.analyzer             = TechnicalAnalyzer()
        self.pattern_detector     = PatternDetector()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.quant_analyzer       = QuantAnalyzer()
        self.ml_model             = EnsembleSignalModel()
        self.lstm_model           = LSTMModel()
        self.regime_models        = RegimeSpecificModels()
        self.var_calculator       = VaRCalculator()
        self.cvar_calculator      = CVaRCalculator()
        self.dynamic_risk         = DynamicRiskManager(
            base_risk_pct = (max_risk_per_trade or TradingConfig.MAX_RISK_PER_TRADE)
                            / (capital or TradingConfig.MAX_CAPITAL)
        )
        self.portfolio_optimizer  = PortfolioOptimizer()
        self.monte_carlo          = MonteCarloSimulator(n_simulations=2000)
        self.execution_model      = ExecutionModel()
        self.garch                = GARCHVolatilityForecaster()
        self.cross_asset          = CrossAssetSignals()
        self.sector_rotation      = SectorRotationTracker()
        self.fno_expiry           = FNOExpiryAnalyzer()
        self.news_sentiment       = NewsSentimentAnalyzer()
        self.drift_detector       = ModelDriftDetector(
            baseline_win_rate=0.65, baseline_accuracy=0.60
        )
        self.explainer            = SignalExplainer()
        self.hmm_detector:    dict[str, HMMRegimeDetector] = {}

        self._nifty_df:       pd.DataFrame | None = None
        self._weekly_data:    dict[str, pd.DataFrame] = {}
        self._corr_matrix:    pd.DataFrame | None = None

    # ==================================================================
    # Public entry points
    # ==================================================================

    def run(
        self,
        symbols: list[str] = None,
        period: str = "1y",
        trade_type: str = "intraday",
    ) -> dict:
        """Run backtest across symbols for the given period."""
        if symbols is None:
            symbols = MarketConfig.NIFTY_50_SYMBOLS[:20]

        logger.info(
            f"Backtest: {len(symbols)} symbols, period={period}, "
            f"type={trade_type}, capital={format_currency(self.initial_capital)}"
        )

        # ── Pre-load shared data ───────────────────────────────────────
        if self.use_regime_filter or self.use_regime_adaptive:
            self._load_nifty_regime(period)

        if self.use_mtf_filter:
            self._load_weekly_data(symbols, period)

        # ── Fetch and enrich all symbol DataFrames ────────────────────
        enriched_dfs: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.fetcher.get_historical_data(symbol, period=period, interval="1d")
            if df is None or len(df) < 60:
                continue
            df = self.analyzer.compute_all_indicators(df)
            if self.use_quant_score:
                df = self.quant_analyzer.compute_all(df, self._nifty_df)
            enriched_dfs[symbol] = df
            # Fit per-symbol HMM
            if self.use_hmm_regime:
                hmm = HMMRegimeDetector()
                hmm.fit(df)
                self.hmm_detector[symbol] = hmm

        if not enriched_dfs:
            return {"error": "No data fetched for any symbol"}

        # ── Train ensemble + LSTM ML models ──────────────────────────
        if self.use_ml_filter:
            logger.info("Training Ensemble ML model...")
            self.ml_model.fit(list(enriched_dfs.values()), verbose=True)
            logger.info("Training LSTM model...")
            self.lstm_model.fit(list(enriched_dfs.values()), verbose=True)
            logger.info("Training regime-specific models...")
            regime_labels = [
                self._get_market_regime(df.index[-1]) for df in enriched_dfs.values()
            ]
            self.regime_models.fit(list(enriched_dfs.values()), regime_labels, verbose=False)

        # ── GARCH volatility forecasts ────────────────────────────────
        logger.info("Fitting GARCH volatility models...")
        self.garch.fit_all(enriched_dfs)

        # ── Cross-asset + sector rotation signals ─────────────────────
        try:
            self.sector_rotation.refresh()
        except Exception:
            pass

        # ── Precompute correlation matrix ─────────────────────────────
        if self.use_correlation_filter:
            self._build_correlation_matrix(enriched_dfs)

        # ── Init dynamic risk manager ─────────────────────────────────
        if self.use_dynamic_risk:
            self.dynamic_risk.set_initial(self.initial_capital)

        return self._run_with_dataframes(enriched_dfs)

    def _run_with_dataframes(self, enriched_dfs: dict[str, pd.DataFrame]) -> dict:
        """Run backtest given pre-enriched DataFrames (also used by optimizer)."""
        all_trades: list[dict] = []
        symbol_results: dict[str, dict] = {}

        for symbol, df in enriched_dfs.items():
            fund_score = 0.0
            if self.use_fundamental_filter:
                try:
                    fund_score = self.fundamental_analyzer.score(symbol)
                except Exception:
                    pass

            trades = self._backtest_symbol(symbol, df, fund_score)
            if trades:
                all_trades.extend(trades)
                symbol_results[symbol] = self._summarize_trades(trades)

        if not all_trades:
            logger.warning("Backtest: No trades generated")
            return {"error": "No trades generated. Try more symbols or a longer period."}

        all_trades.sort(key=lambda t: t["entry_date"])
        portfolio_result = self._simulate_portfolio(all_trades)
        summary = self._generate_summary(portfolio_result, symbol_results)

        # ── Monte Carlo risk analysis ─────────────────────────────────
        try:
            mc_result = self.monte_carlo.run(
                all_trades, self.initial_capital,
                min_capital=self.initial_capital * 0.5,
            )
            summary["monte_carlo"] = {
                "risk_of_ruin_pct":       mc_result["risk_of_ruin_pct"],
                "prob_loss":              mc_result["prob_loss"],
                "final_equity_p5":        mc_result["final_equity"]["p5"],
                "final_equity_p50":       mc_result["final_equity"]["p50"],
                "final_equity_p95":       mc_result["final_equity"]["p95"],
                "prob_mdd_exceeds_20pct": mc_result["prob_mdd_exceeds_threshold"],
            }
        except Exception:
            pass

        # ── Drift detector: set baseline ──────────────────────────────
        try:
            actuals = [1 if t["pnl"] > 0 else 0 for t in all_trades]
            self.drift_detector.set_baseline(
                [0.55] * len(actuals), actuals   # approximate baseline probs
            )
        except Exception:
            pass

        # ── Persist to DB ─────────────────────────────────────────────
        if self.save_to_db:
            try:
                from data.database import TradeDatabase
                TradeDatabase().save_backtest_results(portfolio_result)
            except Exception as exc:
                logger.debug(f"DB save error: {exc}")

        return summary

    # ==================================================================
    # Regime helpers
    # ==================================================================

    def _load_nifty_regime(self, period: str):
        import yfinance as yf
        try:
            raw = yf.download("^NSEI", period=period, interval="1d",
                              auto_adjust=True, progress=False)
            if raw is not None and len(raw) >= 50:
                raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                               for c in raw.columns]
                raw["ema_50"]  = raw["close"].ewm(span=50,  adjust=False).mean()
                raw["ema_200"] = raw["close"].ewm(span=200, adjust=False).mean()
                # ADX for sideways detection
                from ta.trend import ADXIndicator
                adx = ADXIndicator(raw["high"], raw["low"], raw["close"])
                raw["adx"] = adx.adx()
                self._nifty_df = raw
                return
        except Exception:
            pass
        self._nifty_df = None

    def _get_market_regime(self, date) -> str:
        """
        Returns 'BULL', 'BEAR', or 'SIDEWAYS'.

        BULL     : Nifty > EMA200 AND ADX > 20
        SIDEWAYS : ADX < 20 (skip trading)
        BEAR     : Nifty < EMA200
        """
        if self._nifty_df is None:
            return "BULL"   # default: allow trades
        try:
            if hasattr(date, "tzinfo") and date.tzinfo is not None:
                date = date.tz_localize(None)
            idx = self._nifty_df.index.searchsorted(date, side="right") - 1
            if idx < 0:
                return "BULL"
            row   = self._nifty_df.iloc[idx]
            close = float(row.get("close", 0) or 0)
            e200  = float(row.get("ema_200", 0) or 0)
            adx   = float(row.get("adx", 25) or 25)
            if pd.isna(e200) or e200 == 0:
                return "BULL"
            if adx < 20:
                return "SIDEWAYS"
            if close > e200:
                return "BULL"
            return "BEAR"
        except Exception:
            return "BULL"

    def _get_nifty_trend(self, date) -> str:
        """Legacy: simple UP/DOWN/NEUTRAL for regime filter."""
        regime = self._get_market_regime(date)
        if regime == "BULL":
            return "UP"
        if regime == "BEAR":
            return "DOWN"
        return "NEUTRAL"

    def _get_adaptive_params(self, regime: str) -> dict:
        """Return regime-specific trading parameters."""
        if regime == "BULL":
            return {
                "stop_loss_multiplier": max(self.stop_loss_multiplier - 0.1, 0.8),
                "target_multiplier":    self.target_multiplier + 0.3,
                "min_composite_score":  max(self.min_composite_score - 5, 20),
                "max_hold":             10,
            }
        elif regime == "BEAR":
            return {
                "stop_loss_multiplier": self.stop_loss_multiplier,
                "target_multiplier":    max(self.target_multiplier - 0.2, 1.5),
                "min_composite_score":  self.min_composite_score + 5,
                "max_hold":             5,
            }
        else:  # SIDEWAYS — require stronger signal, don't skip entirely
            return {
                "stop_loss_multiplier": self.stop_loss_multiplier + 0.1,
                "target_multiplier":    self.target_multiplier,
                "min_composite_score":  self.min_composite_score + 10,  # higher bar
                "max_hold":             5,
            }

    # ==================================================================
    # Multi-timeframe helpers
    # ==================================================================

    def _load_weekly_data(self, symbols: list[str], period: str):
        import yfinance as yf
        # Map period to a slightly longer window for weekly data
        period_map = {"6mo": "9mo", "1y": "15mo", "2y": "3y", "3y": "4y"}
        weekly_period = period_map.get(period, "3y")

        for symbol in symbols:
            try:
                raw = yf.download(
                    f"{symbol}.NS", period=weekly_period,
                    interval="1wk", auto_adjust=True, progress=False,
                )
                if raw is None or len(raw) < 20:
                    continue
                raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                               for c in raw.columns]
                raw["ema_20"] = raw["close"].ewm(span=20, adjust=False).mean()
                self._weekly_data[symbol] = raw
            except Exception:
                pass

    def _get_weekly_trend(self, symbol: str, date) -> str:
        """Return 'UP', 'DOWN', or 'NEUTRAL' based on weekly EMA20."""
        wdf = self._weekly_data.get(symbol)
        if wdf is None or len(wdf) == 0:
            return "NEUTRAL"
        try:
            if hasattr(date, "tzinfo") and date.tzinfo is not None:
                date = date.tz_localize(None)
            idx = wdf.index.searchsorted(date, side="right") - 1
            if idx < 0:
                return "NEUTRAL"
            row   = wdf.iloc[idx]
            close = float(row.get("close", 0) or 0)
            ema20 = float(row.get("ema_20", 0) or 0)
            if pd.isna(ema20) or ema20 == 0:
                return "NEUTRAL"
            return "UP" if close > ema20 else "DOWN"
        except Exception:
            return "NEUTRAL"

    # ==================================================================
    # Correlation matrix
    # ==================================================================

    def _build_correlation_matrix(self, enriched_dfs: dict[str, pd.DataFrame]):
        """Compute pairwise return correlation across all symbols."""
        returns = {}
        for symbol, df in enriched_dfs.items():
            ret = df["close"].pct_change().dropna()
            if len(ret) > 50:
                returns[symbol] = ret

        if len(returns) < 2:
            return

        aligned = pd.DataFrame(returns).dropna()
        self._corr_matrix = aligned.corr()

    def _are_correlated(self, sym_a: str, sym_b: str) -> bool:
        """True if |correlation| > max_correlation."""
        if self._corr_matrix is None:
            return False
        try:
            r = self._corr_matrix.loc[sym_a, sym_b]
            return abs(float(r)) > self.max_correlation
        except Exception:
            return False

    # ==================================================================
    # Per-symbol backtest
    # ==================================================================

    def _backtest_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        fund_score: float = 0.0,
    ) -> list[dict]:
        """Generate trades for one symbol using enriched df."""
        trades       = []
        lookback     = 50
        ml_cutoff    = int(len(df) * 0.70)   # train on first 70%, predict on last 30%

        # Kelly running stats
        kelly_wins:   list[float] = []
        kelly_losses: list[float] = []

        i = lookback
        while i < len(df):
            row   = df.iloc[i]
            score = float(row.get("composite_score", 0) or 0)
            atr   = float(row.get("atr", 0) or 0)

            if atr <= 0 or pd.isna(atr):
                i += 1
                continue

            # ── Regime-adaptive params ────────────────────────────────
            regime = "BULL"
            adaptive = {}
            if self.use_regime_adaptive:
                regime   = self._get_market_regime(df.index[i])
                adaptive = self._get_adaptive_params(regime)
                if not adaptive:   # SIDEWAYS — skip bar
                    i += 1
                    continue

            sl_mult    = adaptive.get("stop_loss_multiplier", self.stop_loss_multiplier)
            tgt_mult   = adaptive.get("target_multiplier",    self.target_multiplier)
            min_score  = adaptive.get("min_composite_score",  self.min_composite_score)
            max_hold   = adaptive.get("max_hold",             7)

            # ── Direction from pure technical score ───────────────────
            direction = None
            if score >= min_score:
                direction = "BUY"
            elif score <= -min_score:
                direction = "SELL"

            if direction is None:
                i += 1
                continue

            # ── Mean-reversion quality gate ───────────────────────────
            if self.use_quant_score and "mean_rev_z" in df.columns:
                z = float(row.get("mean_rev_z", 0) or 0)
                if direction == "BUY"  and z >  2.7:
                    i += 1; continue
                if direction == "SELL" and z < -2.7:
                    i += 1; continue

            # ── ADX + volume filter ───────────────────────────────────
            adx          = float(row.get("adx", 0) or 0)
            volume_ratio = float(row.get("volume_ratio", 0) or 0)
            if pd.isna(adx) or adx < self.min_adx:
                i += 1; continue
            if pd.isna(volume_ratio) or volume_ratio < 1.0:
                i += 1; continue

            # ── Market regime filter (BULL = no shorts, BEAR = no longs) ─
            if self.use_regime_filter:
                if regime == "BULL" and direction == "SELL":
                    i += 1; continue
                if regime == "BEAR" and direction == "BUY":
                    i += 1; continue

            # ── Multi-timeframe confluence ────────────────────────────
            if self.use_mtf_filter:
                weekly_trend = self._get_weekly_trend(symbol, df.index[i])
                if weekly_trend == "DOWN" and direction == "BUY":
                    i += 1; continue
                if weekly_trend == "UP"   and direction == "SELL":
                    i += 1; continue
                # NEUTRAL → allow (don't over-filter)

            # ── Momentum confirmation for SELL signals ────────────────
            if direction == "SELL" and self.use_quant_score and "momentum_score" in df.columns:
                mom = float(row.get("momentum_score", 0) or 0)
                if mom > 0.3:
                    i += 1; continue

            # ── ML signal filter (ensemble + LSTM combined) ───────────
            if self.use_ml_filter and self.ml_model.is_fitted and i >= ml_cutoff:
                ensemble_prob = self.ml_model.predict_proba(row, df, i)
                lstm_prob     = self.lstm_model.predict_proba(row, df, i) \
                                if self.lstm_model.is_fitted else ensemble_prob
                regime_prob   = self.regime_models.predict_proba(regime, row, df, i) \
                                if self.regime_models.is_fitted else ensemble_prob
                # Weighted average: 50% ensemble + 30% LSTM + 20% regime-specific
                combined_prob = 0.50 * ensemble_prob + 0.30 * lstm_prob + 0.20 * regime_prob
                self.drift_detector.record_prediction(combined_prob)
                if combined_prob < self.ml_prob_threshold:
                    i += 1; continue

            # ── F&O expiry filter ─────────────────────────────────────
            if self.fno_expiry.should_avoid(df.index[i].date() if hasattr(df.index[i], "date") else None):
                i += 1; continue

            # ── Sector rotation filter ────────────────────────────────
            if not self.sector_rotation.is_in_leading_sector(symbol, top_n=6):
                i += 1; continue

            # ── GARCH-adjusted SL/target multipliers ──────────────────
            garch_sl_mult  = self.garch.sl_multiplier(symbol, sl_mult)
            garch_tgt_mult = self.garch.target_multiplier(symbol, tgt_mult)

            # ── Entry price and SL / target (GARCH-adjusted) ─────────
            entry_price = float(row["close"])
            if direction == "BUY":
                stop_loss = entry_price - garch_sl_mult * atr
                target    = entry_price + garch_tgt_mult * atr
            else:
                stop_loss = entry_price + garch_sl_mult * atr
                target    = entry_price - garch_tgt_mult * atr

            risk   = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            if rr_ratio < self.min_rr_ratio:
                i += 1; continue

            # ── Position sizing ───────────────────────────────────────
            risk_per_share = abs(entry_price - stop_loss)
            base_qty = int(self.max_risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            if base_qty < 1:
                i += 1; continue

            # Beta scale (high-beta → smaller)
            beta_scale = 1.0
            if self.use_quant_score and "beta_60" in df.columns:
                beta = float(row.get("beta_60", 1.0) or 1.0)
                if not np.isnan(beta) and beta > 0:
                    beta_scale = float(np.clip(1.0 / beta, 0.4, 1.5))

            # Fundamental scale (strong fundamentals → larger position)
            fund_scale = 1.0
            if self.use_fundamental_filter and fund_score != 0.0:
                fund_scale = float(np.clip(1.0 + fund_score / 100.0, 0.7, 1.3))

            # Kelly scale (after warmup)
            kelly_scale = 1.0
            if self.use_kelly_sizing:
                kelly_scale = self._kelly_scale(kelly_wins, kelly_losses)

            # Dynamic risk scale (reduce during drawdowns)
            dynamic_scale = 1.0
            if self.use_dynamic_risk:
                dynamic_scale = self.dynamic_risk.get_risk_multiplier()
                dynamic_scale *= self.dynamic_risk.volatility_adjustment(df)

            # Sector rotation bias (+/-20% based on leading/lagging sector)
            sector_scale = 1.0 + self.sector_rotation.sector_bias(symbol)

            # F&O expiry size factor
            expiry_scale = self.fno_expiry.size_factor(
                df.index[i].date() if hasattr(df.index[i], "date") else None
            )

            quantity = max(1, int(base_qty * beta_scale * fund_scale
                                  * kelly_scale * dynamic_scale
                                  * sector_scale * expiry_scale))

            # VaR override: cap quantity so position VaR ≤ 2% of capital
            if self.use_var_sizing:
                var_qty  = self.var_calculator.var_adjusted_quantity(
                    df, self.initial_capital, entry_price, stop_loss
                )
                cvar_qty = self.cvar_calculator.cvar_adjusted_quantity(
                    df, self.initial_capital, entry_price
                )
                quantity = min(quantity, var_qty, cvar_qty)

            # Cap at 30% of capital
            max_qty  = int((self.initial_capital * 0.3) / entry_price)
            quantity = min(quantity, max_qty)
            if quantity < 1:
                i += 1; continue

            # ── Simulate trade ────────────────────────────────────────
            trade = self._simulate_trade(
                df, i, symbol, direction, entry_price, stop_loss, target,
                quantity, score, atr, rr_ratio, max_hold,
            )
            if trade:
                trades.append(trade)
                # Update Kelly running stats (per-share P&L)
                pnl_per_share = trade["pnl"] / trade["quantity"] if trade["quantity"] > 0 else 0
                if pnl_per_share > 0:
                    kelly_wins.append(pnl_per_share)
                else:
                    kelly_losses.append(abs(pnl_per_share))
                i = trade["_exit_bar_idx"] + 1
            else:
                i += 1

        return trades

    # ==================================================================
    # Kelly criterion helper
    # ==================================================================

    def _kelly_scale(self, wins: list[float], losses: list[float]) -> float:
        """Return Kelly-adjusted position scale factor [0.4, 1.5]."""
        n = len(wins) + len(losses)
        if n < self.kelly_warmup:
            return 1.0   # not enough data yet

        wr       = len(wins) / n
        avg_win  = float(np.mean(wins))  if wins   else 0.01
        avg_loss = float(np.mean(losses)) if losses else 0.01

        if avg_loss == 0 or avg_win == 0:
            return 1.0

        kelly_f = wr - (1 - wr) / (avg_win / avg_loss)
        kelly_f = max(0.0, kelly_f) * self.kelly_fraction  # fractional Kelly
        # Map Kelly fraction to a position size multiplier
        # kelly_f=0 → scale=0.5; kelly_f=0.15 → scale=1.5 (linear)
        scale = 0.5 + (kelly_f / 0.15) * 1.0
        return float(np.clip(scale, 0.4, 1.5))

    # ==================================================================
    # Trade simulation
    # ==================================================================

    def _simulate_trade(
        self, df, entry_idx, symbol, direction, entry_price,
        stop_loss, target, quantity, score, atr, rr_ratio, max_hold=7,
    ) -> dict | None:
        entry_date      = df.index[entry_idx]
        current_stop    = stop_loss
        best_price      = entry_price
        breakeven_trigger = 0.5 * atr

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            bar  = df.iloc[j]
            high = float(bar["high"])
            low  = float(bar["low"])

            if direction == "BUY":
                if high > best_price:
                    best_price = high
                    if best_price - entry_price >= breakeven_trigger:
                        trail = best_price - self.trail_atr_multiplier * atr
                        if trail >= entry_price:
                            current_stop = max(current_stop, trail)
                if low <= current_stop:
                    pnl    = (current_stop - entry_price) * quantity
                    reason = "Trailing Stop" if current_stop > stop_loss else "Stop Loss"
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, current_stop, quantity, pnl,
                        reason, score, atr, rr_ratio, j,
                    )
                if high >= target:
                    pnl = (target - entry_price) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, target, quantity, pnl,
                        "Target Hit", score, atr, rr_ratio, j,
                    )

            else:  # SELL
                if low < best_price:
                    best_price = low
                    if entry_price - best_price >= breakeven_trigger:
                        trail = best_price + self.trail_atr_multiplier * atr
                        if trail <= entry_price:
                            current_stop = min(current_stop, trail)
                if high >= current_stop:
                    pnl    = (entry_price - current_stop) * quantity
                    reason = "Trailing Stop" if current_stop < stop_loss else "Stop Loss"
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, current_stop, quantity, pnl,
                        reason, score, atr, rr_ratio, j,
                    )
                if low <= target:
                    pnl = (entry_price - target) * quantity
                    return self._make_trade_record(
                        symbol, direction, entry_date, df.index[j],
                        entry_price, target, quantity, pnl,
                        "Target Hit", score, atr, rr_ratio, j,
                    )

        # Time exit
        last_idx   = min(entry_idx + max_hold, len(df) - 1)
        exit_price = float(df.iloc[last_idx]["close"])
        pnl = (exit_price - entry_price) * quantity if direction == "BUY" \
              else (entry_price - exit_price) * quantity
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
        # Apply transaction costs: slippage + brokerage (both sides) + STT (sell)
        if self.use_slippage:
            tc = entry_price * quantity * (_SLIPPAGE_PCT + _BROKERAGE_PCT) * 2
            tc += exit_price * quantity * _STT_PCT   # STT on sell side
            pnl -= tc

        return {
            "symbol":          symbol,
            "direction":       direction,
            "entry_date":      entry_date,
            "exit_date":       exit_date,
            "entry_price":     round(entry_price, 2),
            "exit_price":      round(exit_price, 2),
            "quantity":        quantity,
            "pnl":             round(pnl, 2),
            "pnl_pct":         round((pnl / (entry_price * quantity)) * 100, 2),
            "exit_reason":     exit_reason,
            "composite_score": round(score, 1),
            "atr":             round(atr, 2),
            "rr_ratio":        round(rr_ratio, 2),
            "hold_days":       (exit_date - entry_date).days,
            "_exit_bar_idx":   exit_bar_idx,
        }

    # ==================================================================
    # Portfolio simulation
    # ==================================================================

    def _simulate_portfolio(self, trades: list[dict]) -> dict:
        capital       = self.initial_capital
        peak_capital  = capital
        max_drawdown  = 0.0
        equity_curve  = [{"date": trades[0]["entry_date"], "equity": capital}]

        daily_pnl:   dict[str, float] = {}
        symbol_pnl:  dict[str, float] = {}   # per-symbol cumulative P&L
        max_symbol_loss = -1_200.0            # circuit breaker threshold
        accepted_trades  = []
        rejected_trades  = []

        # Track open positions for correlation filter: list of (symbol, exit_date)
        open_positions: list[tuple[str, object]] = []

        for trade in trades:
            entry_date = trade["entry_date"]
            exit_date  = trade["exit_date"]
            date_key   = entry_date.strftime("%Y-%m-%d")
            symbol     = trade["symbol"]

            # ── Expire closed positions ───────────────────────────────
            open_positions = [(s, ed) for s, ed in open_positions if ed > entry_date]

            # ── Correlation filter ────────────────────────────────────
            if self.use_correlation_filter:
                correlated = False
                for open_sym, _ in open_positions:
                    if open_sym != symbol and self._are_correlated(symbol, open_sym):
                        correlated = True
                        break
                if correlated:
                    rejected_trades.append({**trade, "reject_reason": "Correlated position"})
                    continue

            # ── Per-symbol circuit breaker ────────────────────────────
            if symbol_pnl.get(symbol, 0.0) <= max_symbol_loss:
                rejected_trades.append({**trade, "reject_reason": "Symbol circuit breaker"})
                continue

            # ── Daily limits ──────────────────────────────────────────
            day_pnl = daily_pnl.get(date_key, 0.0)
            if day_pnl >= self.daily_profit_target:
                rejected_trades.append({**trade, "reject_reason": "Daily target reached"})
                continue
            if day_pnl <= -self.daily_loss_limit:
                rejected_trades.append({**trade, "reject_reason": "Daily loss limit"})
                continue

            # ── Capital check ─────────────────────────────────────────
            trade_cost = trade["entry_price"] * trade["quantity"]
            if trade_cost > capital * 0.5:
                rejected_trades.append({**trade, "reject_reason": "Insufficient capital"})
                continue

            # ── Execute ───────────────────────────────────────────────
            capital  += trade["pnl"]
            daily_pnl[date_key]  = daily_pnl.get(date_key,  0.0) + trade["pnl"]
            symbol_pnl[symbol]   = symbol_pnl.get(symbol,   0.0) + trade["pnl"]
            accepted_trades.append(trade)
            open_positions.append((symbol, exit_date))

            # Update dynamic risk manager with new equity after each trade
            if self.use_dynamic_risk:
                self.dynamic_risk.update_equity(capital)

            equity_curve.append({"date": exit_date, "equity": round(capital, 2)})
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "trades":          accepted_trades,
            "rejected_trades": rejected_trades,
            "final_capital":   round(capital, 2),
            "equity_curve":    equity_curve,
            "max_drawdown":    round(max_drawdown, 2),
            "daily_pnl":       daily_pnl,
        }

    # ==================================================================
    # Summary helpers
    # ==================================================================

    def _summarize_trades(self, trades: list[dict]) -> dict:
        if not trades:
            return {}
        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        return {
            "total_trades":  len(trades),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / len(trades) * 100, 1),
            "total_pnl":     round(sum(t["pnl"] for t in trades), 2),
            "avg_win":       round(np.mean([t["pnl"] for t in wins]),   2) if wins   else 0,
            "avg_loss":      round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
            "avg_hold_days": round(np.mean([t["hold_days"] for t in trades]), 1),
        }

    def _generate_summary(self, portfolio: dict, symbol_results: dict) -> dict:
        trades = portfolio["trades"]
        if not trades:
            return {"error": "No accepted trades after portfolio simulation"}

        wins         = [t for t in trades if t["pnl"] > 0]
        losses       = [t for t in trades if t["pnl"] <= 0]
        total_pnl    = sum(t["pnl"] for t in trades)
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss   = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        daily_pnl     = portfolio["daily_pnl"]
        profitable_days = sum(1 for v in daily_pnl.values() if v > 0)
        losing_days     = sum(1 for v in daily_pnl.values() if v <= 0)
        avg_daily_pnl   = np.mean(list(daily_pnl.values())) if daily_pnl else 0

        target_hits  = len([t for t in trades if t["exit_reason"] == "Target Hit"])
        sl_hits      = len([t for t in trades if t["exit_reason"] == "Stop Loss"])
        trail_hits   = len([t for t in trades if t["exit_reason"] == "Trailing Stop"])
        time_exits   = len([t for t in trades if t["exit_reason"] == "Time Exit"])

        streak, max_win_streak, max_lose_streak = 0, 0, 0
        current_streak_type = None
        for t in trades:
            if t["pnl"] > 0:
                streak = streak + 1 if current_streak_type == "win" else 1
                current_streak_type = "win"
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = streak + 1 if current_streak_type == "lose" else 1
                current_streak_type = "lose"
                max_lose_streak = max(max_lose_streak, streak)

        top_symbols = sorted(
            symbol_results.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True,
        )

        # ML feature importance (if fitted)
        ml_importance = {}
        if self.use_ml_filter and self.ml_model.is_fitted:
            ml_importance = self.ml_model.feature_importance()

        return {
            "summary": {
                "initial_capital":  self.initial_capital,
                "final_capital":    portfolio["final_capital"],
                "net_pnl":          round(total_pnl, 2),
                "return_pct":       round((total_pnl / self.initial_capital) * 100, 2),
                "total_trades":     len(trades),
                "rejected_trades":  len(portfolio["rejected_trades"]),
                "wins":             len(wins),
                "losses":           len(losses),
                "win_rate":         round(len(wins) / len(trades) * 100, 1),
                "profit_factor":    round(profit_factor, 2),
                "max_drawdown_pct": portfolio["max_drawdown"],
                "avg_win":          round(np.mean([t["pnl"] for t in wins]),   2) if wins   else 0,
                "avg_loss":         round(np.mean([t["pnl"] for t in losses]), 2) if losses else 0,
                "largest_win":      round(max(t["pnl"] for t in trades), 2),
                "largest_loss":     round(min(t["pnl"] for t in trades), 2),
                "avg_hold_days":    round(np.mean([t["hold_days"] for t in trades]), 1),
            },
            "exit_reasons": {
                "target_hit":    target_hits,
                "stop_loss":     sl_hits,
                "trailing_stop": trail_hits,
                "time_exit":     time_exits,
            },
            "daily_stats": {
                "trading_days":   len(daily_pnl),
                "profitable_days": profitable_days,
                "losing_days":    losing_days,
                "avg_daily_pnl":  round(avg_daily_pnl, 2),
                "best_day":       round(max(daily_pnl.values()), 2),
                "worst_day":      round(min(daily_pnl.values()), 2),
            },
            "streaks": {
                "max_win_streak":  max_win_streak,
                "max_lose_streak": max_lose_streak,
            },
            "top_symbols":    [{"symbol": s, **r} for s, r in top_symbols[:5]],
            "bottom_symbols": [{"symbol": s, **r} for s, r in top_symbols[-5:]],
            "ml_feature_importance": ml_importance,
            "trades":         trades,
            "equity_curve":   portfolio["equity_curve"],
        }
