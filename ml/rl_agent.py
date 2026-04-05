"""
Reinforcement Learning position sizer using Thompson Sampling (Multi-Armed Bandit).
Learns which position size multipliers produce the best risk-adjusted returns.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from utils.logger import logger

_DEFAULT_ARMS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]


class RLPositionSizer:
    """
    Thompson Sampling bandit that learns the best position size multiplier.
    Each arm corresponds to a multiplier applied to the base position size.
    """

    def __init__(self, arms: list[float] = None, prior_alpha: float = 1.0,
                 prior_beta: float = 1.0, strategy: str = "thompson",
                 epsilon: float = 0.10):
        self.arms     = arms or _DEFAULT_ARMS
        self.n_arms   = len(self.arms)
        self.strategy = strategy
        self.epsilon  = epsilon
        self.alpha    = [prior_alpha] * self.n_arms
        self.beta     = [prior_beta]  * self.n_arms
        self._last_arm: int | None = None

    # ------------------------------------------------------------------
    def select_arm(self) -> int:
        if self.strategy == "epsilon_greedy":
            if np.random.random() < self.epsilon:
                return int(np.random.randint(self.n_arms))
            means = [a / (a + b) for a, b in zip(self.alpha, self.beta)]
            return int(np.argmax(means))
        # Thompson sampling
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return int(np.argmax(samples))

    def get_multiplier(self) -> float:
        self._last_arm = self.select_arm()
        return self.arms[self._last_arm]

    def update(self, arm_idx: int, reward: bool):
        """Update Beta distribution for arm_idx. reward=True → win, False → loss."""
        if 0 <= arm_idx < self.n_arms:
            if reward:
                self.alpha[arm_idx] += 1
            else:
                self.beta[arm_idx]  += 1

    def update_from_trade(self, multiplier_used: float, pnl: float):
        """Find closest arm to multiplier_used and update from trade outcome."""
        diffs   = [abs(m - multiplier_used) for m in self.arms]
        arm_idx = int(np.argmin(diffs))
        self.update(arm_idx, reward=(pnl > 0))

    def get_best_arm(self) -> tuple[float, float]:
        """Return (best_multiplier, confidence) where confidence = α/(α+β)."""
        means   = [a / (a + b) for a, b in zip(self.alpha, self.beta)]
        best    = int(np.argmax(means))
        return self.arms[best], round(means[best], 3)

    def get_stats(self) -> list[dict]:
        return [
            {"multiplier": m, "mean": round(a/(a+b), 3),
             "uncertainty": round(np.sqrt(a*b/((a+b)**2*(a+b+1))), 4),
             "n_pulls": int(a + b - 2)}
            for m, a, b in zip(self.arms, self.alpha, self.beta)
        ]

    def set_strategy(self, strategy: str, epsilon: float = 0.10):
        self.strategy = strategy
        self.epsilon  = epsilon

    def reset(self):
        self.alpha = [1.0] * self.n_arms
        self.beta  = [1.0] * self.n_arms

    def save(self, path: str = "models/rl_agent.json"):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump({"arms": self.arms, "alpha": self.alpha, "beta": self.beta,
                       "strategy": self.strategy, "epsilon": self.epsilon}, f)
        logger.info(f"RLPositionSizer saved to {path}")

    def load(self, path: str = "models/rl_agent.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            self.arms     = d["arms"]
            self.n_arms   = len(self.arms)
            self.alpha    = d["alpha"]
            self.beta     = d["beta"]
            self.strategy = d.get("strategy", "thompson")
            self.epsilon  = d.get("epsilon", 0.10)
            logger.info(f"RLPositionSizer loaded from {path}")
        except Exception as exc:
            logger.debug(f"RLPositionSizer load failed: {exc}")
