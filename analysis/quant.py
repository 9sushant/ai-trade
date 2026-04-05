"""
Quantitative factor analysis using historical price data only.
All computations use rolling windows — zero look-ahead bias in backtesting.

Factors computed per bar:
  1. Price momentum       (1-month & 3-month return)
  2. Mean-reversion       (z-score vs 20-day MA)
  3. Sharpe ratio         (60-day rolling, annualised)
  4. Beta vs Nifty        (60-day rolling)
  5. Volume momentum      (5-day vs 20-day avg volume)

Combined quant_score column: [-20, +20]
  Positive → quant tailwinds for the current signal direction.
  Negative → quant headwinds (weaker signal quality).
"""
import numpy as np
import pandas as pd
from utils.logger import logger

# India 10-year G-Sec yield as daily risk-free rate
_ANNUAL_RF = 0.065
_DAILY_RF  = _ANNUAL_RF / 252


class QuantAnalyzer:
    """
    Adds quantitative factor columns to a price DataFrame.

    Usage (backtest):
        df = quant.compute_all(df, nifty_df=nifty_df)
        # df now has: momentum_score, mean_rev_z, sharpe_60, beta_60, vol_momentum, quant_score

    Usage (live):
        score = quant.score_latest(df, nifty_df)
    """

    # ------------------------------------------------------------------
    # Core factor computations (column-wise, vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _momentum_series(close: pd.Series, short: int = 20, long: int = 63) -> pd.Series:
        """
        Dual-timeframe momentum: blend of short (1-mo) and long (3-mo) returns.
        Returns series in [-1, +1].
        """
        ret_short = close.pct_change(short)
        ret_long  = close.pct_change(long)
        raw = 0.4 * ret_short + 0.6 * ret_long
        # Map to [-1, +1] via tanh-like clip
        return raw.clip(-0.25, 0.25) / 0.25

    @staticmethod
    def _mean_rev_z_series(close: pd.Series, window: int = 20) -> pd.Series:
        """
        Z-score of price vs rolling mean.  Positive = above MA (overbought).
        Clipped to [-3, +3].
        """
        ma  = close.rolling(window).mean()
        std = close.rolling(window).std()
        z   = (close - ma) / std.replace(0, np.nan)
        return z.clip(-3, 3)

    @staticmethod
    def _sharpe_series(close: pd.Series, window: int = 60) -> pd.Series:
        """
        Rolling Sharpe ratio (annualised, India risk-free rate).
        Clipped to [-4, +4].
        """
        ret   = close.pct_change()
        excess = ret - _DAILY_RF
        mean  = excess.rolling(window).mean()
        std   = excess.rolling(window).std().replace(0, np.nan)
        sharpe = (mean / std) * np.sqrt(252)
        return sharpe.clip(-4, 4)

    @staticmethod
    def _beta_series(close: pd.Series, nifty_close: pd.Series, window: int = 60) -> pd.Series:
        """
        Rolling beta vs Nifty index.
        Clipped to [-2, +4].
        """
        ret_stock = close.pct_change()
        ret_nifty = nifty_close.pct_change()

        # Align on common index
        aligned   = pd.concat([ret_stock, ret_nifty], axis=1, join="inner")
        aligned.columns = ["stock", "nifty"]

        def _roll_beta(window_df: pd.DataFrame) -> float:
            if window_df["nifty"].std() == 0:
                return 1.0
            cov = np.cov(window_df["stock"], window_df["nifty"])[0][1]
            var = window_df["nifty"].var()
            return cov / var if var != 0 else 1.0

        beta = (
            aligned
            .rolling(window, min_periods=max(20, window // 3))
            .apply(lambda x: _roll_beta(
                pd.DataFrame({"stock": x, "nifty": aligned["nifty"].loc[x.index]})
            ), raw=False)
            ["stock"]
        )
        return beta.clip(-2, 4)

    @staticmethod
    def _vol_momentum_series(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
        """Volume ratio of short-term vs long-term average, centred at 0."""
        short_vol = volume.rolling(short).mean()
        long_vol  = volume.rolling(long).mean().replace(0, np.nan)
        ratio = short_vol / long_vol - 1          # 0 = neutral, >0 = accelerating
        return ratio.clip(-1, 1)

    # ------------------------------------------------------------------
    # Composite quant score
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_score(
        momentum_score: pd.Series,
        mean_rev_z:     pd.Series,
        sharpe_60:      pd.Series,
        vol_momentum:   pd.Series,
    ) -> pd.Series:
        """
        Combines factors into a *direction-agnostic* quality score [-20, +20].

        High positive quant_score means:
          - Price momentum is up
          - Price is near MA (not over-extended)
          - Recent Sharpe is positive
          - Volume is expanding

        Interpretation in the engine:
          For BUY  signals → higher quant_score boosts the trade
          For SELL signals → lower quant_score (negative) boosts the trade
        """
        # Weights: momentum 8, mean-rev 4, sharpe 5, volume 3
        score = (
            momentum_score          * 8   # [-8, +8]
            - (mean_rev_z / 3) * 4        # Penalise over-extension: high z → deduct for BUY
            + (sharpe_60 / 4) * 5         # [-5, +5] based on recent Sharpe
            + vol_momentum          * 3   # [-3, +3]
        )
        return score.clip(-20, 20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(
        self,
        df: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Add quant factor columns to *df* in-place (returns modified copy).

        New columns:
            momentum_score, mean_rev_z, sharpe_60, beta_60,
            vol_momentum, quant_score
        """
        if df is None or len(df) < 30:
            return df

        df = df.copy()
        close  = df["close"]
        volume = df["volume"]

        df["momentum_score"] = self._momentum_series(close)
        df["mean_rev_z"]     = self._mean_rev_z_series(close)
        df["sharpe_60"]      = self._sharpe_series(close)
        df["vol_momentum"]   = self._vol_momentum_series(volume)

        # Beta requires aligned Nifty data
        if nifty_df is not None and len(nifty_df) >= 60:
            try:
                # Normalise Nifty index to tz-naive for alignment
                nifty_close = nifty_df["close"].copy()
                if hasattr(nifty_close.index, "tz") and nifty_close.index.tz is not None:
                    nifty_close.index = nifty_close.index.tz_localize(None)
                stock_close = close.copy()
                if hasattr(stock_close.index, "tz") and stock_close.index.tz is not None:
                    stock_close.index = stock_close.index.tz_localize(None)

                beta_s = self._beta_series(stock_close, nifty_close)
                # Re-index back to original df index
                beta_s.index = df.index[:len(beta_s)]
                df["beta_60"] = beta_s.reindex(df.index)
            except Exception as exc:
                logger.debug(f"Beta calculation failed: {exc}")
                df["beta_60"] = 1.0
        else:
            df["beta_60"] = 1.0

        df["quant_score"] = self._composite_score(
            df["momentum_score"],
            df["mean_rev_z"],
            df["sharpe_60"],
            df["vol_momentum"],
        )

        return df

    def score_latest(
        self,
        df: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
        direction: str = "BUY",
    ) -> float:
        """
        Compute quant score for the most recent bar of *df*.
        *direction* flips the mean-reversion and momentum components.
        """
        if df is None or len(df) < 30:
            return 0.0
        try:
            enriched = self.compute_all(df, nifty_df)
            row = enriched.iloc[-1]

            mom = float(row.get("momentum_score", 0) or 0)
            z   = float(row.get("mean_rev_z",     0) or 0)
            sh  = float(row.get("sharpe_60",       0) or 0)
            vol = float(row.get("vol_momentum",    0) or 0)

            if direction == "SELL":
                mom = -mom   # For shorts, downward momentum is good
                z   = -z    # For shorts, above MA (high z) is favourable

            score = mom * 8 - (z / 3) * 4 + (sh / 4) * 5 + vol * 3
            return float(np.clip(score, -20, 20))
        except Exception as exc:
            logger.debug(f"QuantAnalyzer.score_latest failed: {exc}")
            return 0.0

    def beta_adjusted_quantity(
        self,
        base_qty: int,
        df: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
    ) -> int:
        """
        Scale position size inversely with beta.
        High-beta stocks → smaller position; low-beta → larger (capped at 1.5×).
        """
        if df is None or len(df) < 60:
            return base_qty

        try:
            enriched = self.compute_all(df, nifty_df)
            beta = float(enriched["beta_60"].iloc[-1])
            if np.isnan(beta) or beta <= 0:
                return base_qty
            # Scale: beta=1 → 1×, beta=2 → 0.5×, beta=0.5 → 1.5× (capped)
            scale    = np.clip(1.0 / beta, 0.4, 1.5)
            adjusted = int(base_qty * scale)
            return max(1, adjusted)
        except Exception:
            return base_qty

    # ------------------------------------------------------------------
    # Analytics helpers (for reports / display)
    # ------------------------------------------------------------------

    def full_report(self, df: pd.DataFrame, nifty_df: pd.DataFrame | None = None) -> dict:
        """Return a dict of the latest quant factor values."""
        if df is None or len(df) < 30:
            return {}
        enriched = self.compute_all(df, nifty_df)
        row = enriched.iloc[-1]
        return {
            "momentum_score": round(float(row.get("momentum_score", 0) or 0), 3),
            "mean_rev_z":     round(float(row.get("mean_rev_z",     0) or 0), 3),
            "sharpe_60":      round(float(row.get("sharpe_60",       0) or 0), 3),
            "beta_60":        round(float(row.get("beta_60",          1) or 1), 3),
            "vol_momentum":   round(float(row.get("vol_momentum",    0) or 0), 3),
            "quant_score":    round(float(row.get("quant_score",     0) or 0), 2),
        }
