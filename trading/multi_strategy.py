"""Multi-strategy capital allocator — dynamically routes capital to best strategies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger


class Strategy:
    """Base container for a named trading strategy with performance tracking."""

    def __init__(self, name: str, description: str = ""):
        self.name        = name
        self.description = description
        self.pnl_history: list[float]  = []
        self.trade_count  = 0
        self.wins         = 0
        self.is_active    = True
        self._allocation  = 0.0   # current capital fraction [0, 1]
        self.params: dict = {}

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trade_count, 1) * 100

    @property
    def total_pnl(self) -> float:
        return sum(self.pnl_history)

    @property
    def sharpe(self) -> float:
        if len(self.pnl_history) < 5:
            return 0.0
        ret = np.array(self.pnl_history)
        return float(ret.mean() / max(ret.std(), 1e-8) * np.sqrt(252))

    @property
    def recent_sharpe(self, window: int = 20) -> float:
        recent = self.pnl_history[-window:]
        if len(recent) < 3:
            return 0.0
        ret = np.array(recent)
        return float(ret.mean() / max(ret.std(), 1e-8) * np.sqrt(252))

    def record_trade(self, pnl: float):
        self.pnl_history.append(pnl)
        self.trade_count += 1
        if pnl > 0:
            self.wins += 1

    def __repr__(self):
        return (f"Strategy({self.name}, alloc={self._allocation:.1%}, "
                f"pnl={self.total_pnl:.0f}, wr={self.win_rate:.1f}%)")


class MultiStrategyAllocator:
    """
    Dynamically allocates capital across multiple trading strategies.

    Allocation methods:
      1. Equal weight — baseline
      2. Performance-based — more to winning strategies
      3. Sharpe-based — proportional to risk-adjusted returns
      4. Kelly-optimal — maximize log-wealth across strategies
      5. Risk-parity — equal risk contribution
    """

    STRATEGIES_DEFAULT = [
        "MOMENTUM_TREND",
        "MEAN_REVERSION",
        "PAIRS_TRADING",
        "ORB_BREAKOUT",
        "ML_ENSEMBLE",
    ]

    def __init__(
        self,
        total_capital: float = 50_000,
        method: str = "sharpe",   # equal | performance | sharpe | kelly | risk_parity
        rebalance_every_n: int = 20,  # trades between rebalancing
        min_allocation: float = 0.05,
        max_allocation: float = 0.50,
        lookback_trades: int = 50,
    ):
        self.total_capital    = total_capital
        self.method           = method
        self.rebalance_every  = rebalance_every_n
        self.min_alloc        = min_allocation
        self.max_alloc        = max_allocation
        self.lookback         = lookback_trades
        self._strategies: dict[str, Strategy] = {}
        self._trade_count     = 0
        self._allocations:    dict[str, float] = {}
        self._rebalance_log:  list[dict] = []

        # Initialize default strategies
        for name in self.STRATEGIES_DEFAULT:
            self.add_strategy(name)

    # ------------------------------------------------------------------
    def add_strategy(self, name: str, description: str = "", params: dict = None):
        self._strategies[name] = Strategy(name, description)
        if params:
            self._strategies[name].params = params
        self._rebalance()

    def record_trade(self, strategy_name: str, pnl: float):
        """Record a completed trade for a strategy."""
        if strategy_name not in self._strategies:
            self.add_strategy(strategy_name)
        self._strategies[strategy_name].record_trade(pnl)
        self._trade_count += 1

        if self._trade_count % self.rebalance_every == 0:
            self._rebalance()

    def get_allocation(self, strategy_name: str) -> float:
        """Get current capital allocation fraction for a strategy."""
        return self._allocations.get(strategy_name, 0.0)

    def get_capital(self, strategy_name: str) -> float:
        """Get current capital amount for a strategy."""
        return self.total_capital * self.get_allocation(strategy_name)

    def get_allocations(self) -> dict[str, float]:
        return dict(self._allocations)

    def get_all_capital(self) -> dict[str, float]:
        return {name: round(self.total_capital * alloc, 2)
                for name, alloc in self._allocations.items()}

    # ------------------------------------------------------------------
    def _rebalance(self):
        """Recompute capital allocations across strategies."""
        active = {k: v for k, v in self._strategies.items() if v.is_active}
        if not active:
            return

        if self.method == "equal":
            weights = self._equal_weights(active)
        elif self.method == "performance":
            weights = self._performance_weights(active)
        elif self.method == "sharpe":
            weights = self._sharpe_weights(active)
        elif self.method == "kelly":
            weights = self._kelly_weights(active)
        elif self.method == "risk_parity":
            weights = self._risk_parity_weights(active)
        else:
            weights = self._equal_weights(active)

        # Apply min/max constraints
        for k in weights:
            weights[k] = np.clip(weights[k], self.min_alloc, self.max_alloc)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        self._allocations = weights
        self._rebalance_log.append({
            "timestamp":   datetime.now().isoformat(),
            "method":      self.method,
            "allocations": {k: round(v, 4) for k, v in weights.items()},
        })
        logger.info(f"MultiStrategy rebalanced ({self.method}): "
                    f"{', '.join(f'{k}={v:.1%}' for k, v in weights.items())}")

    def _equal_weights(self, strategies: dict) -> dict[str, float]:
        n = len(strategies)
        return {k: 1/n for k in strategies}

    def _performance_weights(self, strategies: dict) -> dict[str, float]:
        pnls = {k: max(v.total_pnl, 0) for k, v in strategies.items()}
        total = sum(pnls.values())
        if total <= 0:
            return self._equal_weights(strategies)
        return {k: v / total for k, v in pnls.items()}

    def _sharpe_weights(self, strategies: dict) -> dict[str, float]:
        sharpes = {}
        for k, v in strategies.items():
            s = max(v.recent_sharpe, 0.01)
            sharpes[k] = s
        total = sum(sharpes.values())
        if total <= 0:
            return self._equal_weights(strategies)
        return {k: v / total for k, v in sharpes.items()}

    def _kelly_weights(self, strategies: dict) -> dict[str, float]:
        """Kelly criterion per strategy."""
        weights = {}
        for k, strat in strategies.items():
            recent = strat.pnl_history[-self.lookback:] if strat.pnl_history else []
            if len(recent) < 5:
                weights[k] = 1 / len(strategies)
                continue
            wins   = [p for p in recent if p > 0]
            losses = [abs(p) for p in recent if p < 0]
            wr     = len(wins) / len(recent)
            avg_w  = np.mean(wins) if wins else 0.01
            avg_l  = np.mean(losses) if losses else 0.01
            kelly  = wr - (1 - wr) * avg_l / avg_w
            weights[k] = max(kelly * 0.25, 0.01)   # fractional Kelly
        return weights

    def _risk_parity_weights(self, strategies: dict) -> dict[str, float]:
        """Equal risk contribution (inverse volatility)."""
        vols = {}
        for k, strat in strategies.items():
            recent = strat.pnl_history[-self.lookback:]
            vol    = float(np.std(recent)) if len(recent) >= 3 else 1.0
            vols[k] = max(vol, 1e-8)
        inv_vol = {k: 1/v for k, v in vols.items()}
        total   = sum(inv_vol.values())
        return {k: v/total for k, v in inv_vol.items()}

    def performance_report(self) -> dict:
        return {
            "total_capital": self.total_capital,
            "method":        self.method,
            "strategies": {
                name: {
                    "allocation":   round(self._allocations.get(name, 0), 4),
                    "capital":      round(self.total_capital * self._allocations.get(name, 0), 2),
                    "total_pnl":    round(strat.total_pnl,   2),
                    "win_rate":     round(strat.win_rate,     1),
                    "sharpe":       round(strat.sharpe,       3),
                    "trade_count":  strat.trade_count,
                }
                for name, strat in self._strategies.items()
            },
            "last_rebalance": self._rebalance_log[-1] if self._rebalance_log else None,
        }

    def best_strategy(self) -> str:
        """Return the name of the currently best-performing strategy."""
        if not self._strategies:
            return ""
        return max(self._strategies.items(),
                   key=lambda x: x[1].total_pnl)[0]

    def disable_strategy(self, name: str):
        if name in self._strategies:
            self._strategies[name].is_active = False
            self._rebalance()

    def enable_strategy(self, name: str):
        if name in self._strategies:
            self._strategies[name].is_active = True
            self._rebalance()
