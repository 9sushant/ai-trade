"""
Walk-Forward Parameter Optimizer for the backtest engine.

Divides the lookback period into in-sample / out-of-sample folds.
For each fold, runs a grid search on in-sample data, then evaluates
the best parameters on out-of-sample data.  Aggregates OOS metrics.

Usage:
    from backtest.optimizer import WalkForwardOptimizer
    opt = WalkForwardOptimizer(symbols=MarketConfig.NIFTY_50_SYMBOLS[:10])
    result = opt.run()
    print(result["best_params"])
"""
from __future__ import annotations

import itertools
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import MarketConfig
from utils.logger import logger


# Default parameter grid (coarse — keeps total combos ≤ 36)
DEFAULT_GRID = {
    "min_composite_score": [25, 30, 35],
    "stop_loss_multiplier": [1.0, 1.2, 1.5],
    "target_multiplier":   [1.8, 2.0, 2.5],
    "min_adx":             [20, 25],
}


def _score_result(result: dict) -> float:
    """
    Composite objective = risk-adjusted return.
    Rewards high Sharpe-like metric: (return_pct / max_drawdown) × win_rate.
    """
    if "error" in result:
        return -999.0
    s = result.get("summary", {})
    ret   = s.get("return_pct",       0.0)
    mdd   = max(s.get("max_drawdown_pct", 1.0), 0.01)
    wr    = s.get("win_rate",          0.0)
    pf    = s.get("profit_factor",     0.0)
    n     = s.get("total_trades",      0)
    if n < 5:
        return -999.0
    return (ret / mdd) * (wr / 100) * min(pf, 3.0)


class WalkForwardOptimizer:
    """
    Walk-forward parameter optimizer.

    Parameters
    ----------
    symbols : list[str]
        Subset of symbols to run each backtest on (fewer = faster).
    n_folds : int
        Number of walk-forward folds.
    oos_months : int
        Out-of-sample window size per fold (months).
    is_months : int
        In-sample window size per fold (months).
    param_grid : dict | None
        Parameter grid.  Defaults to DEFAULT_GRID.
    """

    def __init__(
        self,
        symbols:    list[str] | None = None,
        n_folds:    int  = 3,
        oos_months: int  = 6,
        is_months:  int  = 12,
        param_grid: dict | None = None,
    ):
        self.symbols    = symbols or MarketConfig.NIFTY_50_SYMBOLS[:10]
        self.n_folds    = n_folds
        self.oos_months = oos_months
        self.is_months  = is_months
        self.param_grid = param_grid or DEFAULT_GRID

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run walk-forward optimization. Returns dict with best_params and OOS stats."""
        folds    = self._build_folds()
        combos   = self._param_combos()
        oos_rows = []

        logger.info(
            f"WalkForwardOptimizer: {len(combos)} param combos × "
            f"{len(folds)} folds = {len(combos) * len(folds)} backtests"
        )

        for fold_idx, fold in enumerate(folds):
            logger.info(
                f"  Fold {fold_idx+1}/{len(folds)} | "
                f"IS: {fold['is_start']} → {fold['is_end']} | "
                f"OOS: {fold['oos_start']} → {fold['oos_end']}"
            )

            # ── Grid search on in-sample ─────────────────────────
            best_is_score = -np.inf
            best_params   = combos[0]

            for params in combos:
                result = self._run_backtest(params, fold["is_start"], fold["is_end"])
                s      = _score_result(result)
                if s > best_is_score:
                    best_is_score = s
                    best_params   = params

            # ── Evaluate best params on OOS ──────────────────────
            oos_result = self._run_backtest(best_params, fold["oos_start"], fold["oos_end"])
            oos_score  = _score_result(oos_result)
            summary    = oos_result.get("summary", {})

            oos_rows.append({
                "fold":                  fold_idx + 1,
                "oos_start":             fold["oos_start"],
                "oos_end":               fold["oos_end"],
                "best_params":           best_params,
                "is_score":              round(best_is_score, 3),
                "oos_score":             round(oos_score, 3),
                "oos_return_pct":        summary.get("return_pct", 0),
                "oos_win_rate":          summary.get("win_rate", 0),
                "oos_max_drawdown":      summary.get("max_drawdown_pct", 0),
                "oos_profit_factor":     summary.get("profit_factor", 0),
                "oos_trades":            summary.get("total_trades", 0),
            })
            logger.info(
                f"    Best IS params: {best_params} | IS score={best_is_score:.3f} | "
                f"OOS score={oos_score:.3f} | OOS return={summary.get('return_pct',0):.2f}%"
            )

        # ── Aggregate results ────────────────────────────────────
        if not oos_rows:
            return {"error": "No OOS results produced"}

        avg_oos_return = np.mean([r["oos_return_pct"]   for r in oos_rows])
        avg_oos_wr     = np.mean([r["oos_win_rate"]      for r in oos_rows])
        avg_oos_mdd    = np.mean([r["oos_max_drawdown"]  for r in oos_rows])
        avg_oos_pf     = np.mean([r["oos_profit_factor"] for r in oos_rows])

        # Most-frequent best params across folds → recommended params
        from collections import Counter
        param_counts = Counter(
            tuple(sorted(r["best_params"].items())) for r in oos_rows
        )
        recommended_params = dict(param_counts.most_common(1)[0][0])

        return {
            "recommended_params": recommended_params,
            "aggregate_oos": {
                "avg_return_pct":    round(avg_oos_return, 2),
                "avg_win_rate":      round(avg_oos_wr, 1),
                "avg_max_drawdown":  round(avg_oos_mdd, 2),
                "avg_profit_factor": round(avg_oos_pf, 2),
            },
            "folds": oos_rows,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_folds(self) -> list[dict]:
        """Build walk-forward fold date ranges."""
        today   = date.today()
        folds   = []
        oos_end = today

        for _ in range(self.n_folds):
            oos_start = oos_end   - timedelta(days=self.oos_months * 30)
            is_end    = oos_start
            is_start  = is_end    - timedelta(days=self.is_months  * 30)
            folds.append({
                "is_start":  is_start.strftime("%Y-%m-%d"),
                "is_end":    is_end.strftime("%Y-%m-%d"),
                "oos_start": oos_start.strftime("%Y-%m-%d"),
                "oos_end":   oos_end.strftime("%Y-%m-%d"),
            })
            oos_end = oos_start

        return list(reversed(folds))

    def _param_combos(self) -> list[dict]:
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _run_backtest(self, params: dict, start_date: str, end_date: str) -> dict:
        """Run a mini-backtest for the given params and date window."""
        # Import here to avoid circular imports
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            stop_loss_multiplier  = params.get("stop_loss_multiplier", 1.2),
            target_multiplier     = params.get("target_multiplier",   2.0),
            min_composite_score   = params.get("min_composite_score", 30),
            min_adx               = params.get("min_adx",             25),
            # Disable slow modules for speed during grid search
            use_ml_filter         = False,
            use_mtf_filter        = False,
            use_regime_filter     = True,
            use_fundamental_filter= False,
            use_quant_score       = True,
            use_kelly_sizing      = False,
            use_correlation_filter= False,
            use_regime_adaptive   = False,
        )

        try:
            raw = self._fetch_date_range(start_date, end_date)
            if not raw:
                return {"error": "No data"}
            return engine._run_with_dataframes(raw)
        except Exception as exc:
            logger.debug(f"Optimizer backtest error ({params}): {exc}")
            return {"error": str(exc)}

    def _fetch_date_range(self, start: str, end: str) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for self.symbols between start and end dates."""
        from data.fetcher import DataFetcher
        from analysis.technical import TechnicalAnalyzer
        from analysis.quant import QuantAnalyzer

        fetcher  = DataFetcher()
        analyzer = TechnicalAnalyzer()
        quant    = QuantAnalyzer()
        result   = {}

        for symbol in self.symbols:
            try:
                raw = yf.download(
                    f"{symbol}.NS",
                    start    = start,
                    end      = end,
                    interval = "1d",
                    auto_adjust = True,
                    progress = False,
                )
                if raw is None or len(raw) < 60:
                    continue
                raw.columns = [
                    c[0].lower() if isinstance(c, tuple) else c.lower()
                    for c in raw.columns
                ]
                df = analyzer.compute_all_indicators(raw)
                df = quant.compute_all(df)
                result[symbol] = df
            except Exception as exc:
                logger.debug(f"Optimizer fetch {symbol}: {exc}")

        return result
