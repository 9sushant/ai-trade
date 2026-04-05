"""Statistical arbitrage pairs trading using cointegration."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

try:
    from statsmodels.tsa.stattools import coint
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

KNOWN_PAIRS = [
    ("HDFCBANK", "ICICIBANK"), ("INFY", "TCS"), ("HINDUNILVR", "ITC"),
    ("AXISBANK", "KOTAKBANK"), ("ONGC", "BPCL"), ("WIPRO", "HCLTECH"),
]


class PairsTrader:
    def __init__(self, min_correlation: float = 0.80, lookback_days: int = 252,
                 zscore_entry: float = 2.0, zscore_exit: float = 0.5):
        self.min_correlation = min_correlation
        self.lookback_days   = lookback_days
        self.zscore_entry    = zscore_entry
        self.zscore_exit     = zscore_exit

    def find_pairs(self, symbols: list[str],
                   price_data: dict[str, pd.DataFrame]) -> list[dict]:
        results = []
        syms    = [s for s in symbols if s in price_data]
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                sym_a, sym_b = syms[i], syms[j]
                try:
                    close_a = price_data[sym_a]["close"].tail(self.lookback_days)
                    close_b = price_data[sym_b]["close"].tail(self.lookback_days)
                    aligned = pd.concat([close_a, close_b], axis=1, join="inner").dropna()
                    if len(aligned) < 60:
                        continue
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if abs(corr) < self.min_correlation:
                        continue
                    p_val, hedge_ratio = 1.0, 1.0
                    if _HAS_STATSMODELS:
                        _, p_val, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
                        # OLS hedge ratio
                        from numpy.linalg import lstsq
                        X = aligned.iloc[:, 1].values.reshape(-1, 1)
                        y = aligned.iloc[:, 0].values
                        hedge_ratio = float(lstsq(X, y, rcond=None)[0][0])
                    if p_val < 0.05 or not _HAS_STATSMODELS:
                        results.append({
                            "sym_a": sym_a, "sym_b": sym_b,
                            "correlation": round(corr, 3),
                            "coint_pvalue": round(p_val, 4),
                            "hedge_ratio":  round(hedge_ratio, 4),
                        })
                except Exception as exc:
                    logger.debug(f"Pairs {sym_a}/{sym_b}: {exc}")
        results.sort(key=lambda x: x["coint_pvalue"])
        return results

    def compute_spread(self, price_a: pd.Series, price_b: pd.Series,
                       hedge_ratio: float) -> pd.Series:
        spread  = price_a - hedge_ratio * price_b
        z_score = (spread - spread.mean()) / spread.std()
        return z_score

    def get_signal(self, sym_a: str, sym_b: str,
                   df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
        try:
            close_a = df_a["close"].tail(self.lookback_days)
            close_b = df_b["close"].tail(self.lookback_days)
            aligned = pd.concat([close_a, close_b], axis=1, join="inner").dropna()
            if len(aligned) < 30:
                return {"entry_signal": False}
            from numpy.linalg import lstsq
            X = aligned.iloc[:, 1].values.reshape(-1, 1)
            y = aligned.iloc[:, 0].values
            hedge_ratio = float(lstsq(X, y, rcond=None)[0][0])
            z = self.compute_spread(aligned.iloc[:, 0], aligned.iloc[:, 1], hedge_ratio)
            current_z = float(z.iloc[-1])
            if current_z > self.zscore_entry:
                return {"entry_signal": True, "direction_a": "SELL", "direction_b": "BUY",
                        "z_score": round(current_z, 2), "hedge_ratio": round(hedge_ratio, 4),
                        "reason": f"{sym_a} overpriced vs {sym_b}"}
            if current_z < -self.zscore_entry:
                return {"entry_signal": True, "direction_a": "BUY", "direction_b": "SELL",
                        "z_score": round(current_z, 2), "hedge_ratio": round(hedge_ratio, 4),
                        "reason": f"{sym_a} underpriced vs {sym_b}"}
            if abs(current_z) < self.zscore_exit:
                return {"entry_signal": False, "close_signal": True, "z_score": round(current_z, 2)}
        except Exception as exc:
            logger.debug(f"Pairs signal error: {exc}")
        return {"entry_signal": False}

    def backtest_pair(self, sym_a: str, sym_b: str,
                      df_a: pd.DataFrame, df_b: pd.DataFrame,
                      capital: float = 50000) -> dict:
        try:
            close_a = df_a["close"]
            close_b = df_b["close"]
            aligned = pd.concat([close_a, close_b], axis=1, join="inner").dropna()
            aligned.columns = ["a", "b"]
            from numpy.linalg import lstsq
            X = aligned["b"].values.reshape(-1, 1)
            hedge_ratio = float(lstsq(X, aligned["a"].values, rcond=None)[0][0])
            spread  = aligned["a"] - hedge_ratio * aligned["b"]
            z       = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
            pnls, in_trade, entry_z = [], False, 0.0
            for i in range(60, len(z)):
                zi = float(z.iloc[i])
                if not in_trade and abs(zi) > self.zscore_entry:
                    in_trade = True; entry_z = zi
                elif in_trade and abs(zi) < self.zscore_exit:
                    pnls.append(-(zi - entry_z) * capital * 0.01)
                    in_trade = False
            wins = [p for p in pnls if p > 0]
            return {"sym_a": sym_a, "sym_b": sym_b, "num_trades": len(pnls),
                    "total_pnl": round(sum(pnls), 2),
                    "win_rate": round(len(wins)/len(pnls)*100, 1) if pnls else 0,
                    "sharpe": round(np.mean(pnls)/np.std(pnls)*np.sqrt(252), 2) if len(pnls)>1 else 0}
        except Exception as exc:
            return {"error": str(exc)}
