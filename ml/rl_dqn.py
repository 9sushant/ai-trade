"""Deep Q-Network (DQN) agent for position sizing and trade decisions."""
from __future__ import annotations
import json
import random
import numpy as np
from collections import deque
from pathlib import Path
from utils.logger import logger

# Action space: position size multipliers
_ACTIONS     = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
_STATE_DIM   = 12   # state features
_HIDDEN      = 64
_GAMMA       = 0.95
_LR          = 1e-3
_EPSILON_MIN = 0.05
_BUFFER_SIZE = 10_000
_BATCH_SIZE  = 64
_TARGET_UPDATE = 50   # steps between target network sync


def _build_state(row, df, i: int, capital: float, peak_capital: float) -> np.ndarray:
    """Build 12-feature state vector from current bar."""
    drawdown  = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
    rsi       = float(row.get("rsi", 50) or 50) / 100
    adx       = float(row.get("adx", 25) or 25) / 100
    atr_pct   = float(row.get("atr", 0) or 0) / max(float(row.get("close", 1) or 1), 1)
    score     = float(row.get("composite_score", 0) or 0) / 100
    vol_ratio = float(row.get("volume_ratio", 1) or 1)
    mom       = float(row.get("momentum_score", 0) or 0)
    bb_pos    = 0.5
    if "bb_upper" in row and "bb_lower" in row:
        bb_u = float(row.get("bb_upper", 0) or 0)
        bb_l = float(row.get("bb_lower", 0) or 0)
        close = float(row.get("close", 0) or 0)
        if bb_u > bb_l:
            bb_pos = (close - bb_l) / (bb_u - bb_l)

    ret_5  = 0.0
    ret_20 = 0.0
    if i >= 5:
        ret_5  = float(df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
    if i >= 20:
        ret_20 = float(df["close"].iloc[i] / df["close"].iloc[i - 20] - 1)

    return np.array([
        drawdown, rsi, adx, atr_pct, score,
        np.clip(vol_ratio, 0, 5) / 5,
        np.clip(mom, -1, 1),
        np.clip(bb_pos, 0, 1),
        np.clip(ret_5,  -0.1, 0.1) / 0.1,
        np.clip(ret_20, -0.2, 0.2) / 0.2,
        capital / 100_000,     # normalized capital
        1.0 if score > 0 else 0.0,  # direction
    ], dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int = _BUFFER_SIZE):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self._buf.append((state, action_idx, reward, next_state, done))

    def sample(self, n: int):
        batch = random.sample(self._buf, min(n, len(self._buf)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self._buf)


class DQNAgent:
    """
    DQN agent for adaptive position sizing.

    Falls back to random valid sizing if PyTorch not installed.
    State: 12-dim market feature vector
    Action: index into _ACTIONS (position size multiplier)
    Reward: risk-adjusted trade P&L (Sharpe-like)
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon   = epsilon
        self.n_actions = len(_ACTIONS)
        self._step     = 0
        self._buf      = ReplayBuffer()
        self._net      = None
        self._tgt      = None
        self._opt      = None
        self._has_torch = False
        self._init_network()

    def _init_network(self):
        try:
            import torch
            import torch.nn as nn

            class _QNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(_STATE_DIM, _HIDDEN), nn.ReLU(),
                        nn.Linear(_HIDDEN, _HIDDEN),    nn.ReLU(),
                        nn.Linear(_HIDDEN, len(_ACTIONS))
                    )
                def forward(self, x):
                    return self.net(x)

            self._net = _QNet()
            self._tgt = _QNet()
            self._tgt.load_state_dict(self._net.state_dict())
            self._opt = torch.optim.Adam(self._net.parameters(), lr=_LR)
            self._has_torch = True
        except ImportError:
            logger.info("DQNAgent: torch not found, using random policy")

    # ------------------------------------------------------------------
    def get_multiplier(self, state: np.ndarray) -> float:
        """Epsilon-greedy action selection → size multiplier."""
        if random.random() < self.epsilon or not self._has_torch:
            return random.choice(_ACTIONS)
        import torch
        with torch.no_grad():
            q = self._net(torch.tensor(state).unsqueeze(0))
            idx = int(q.argmax().item())
        return _ACTIONS[idx]

    def get_multiplier_from_bar(self, row, df, i: int,
                                 capital: float, peak_capital: float) -> float:
        state = _build_state(row, df, i, capital, peak_capital)
        return self.get_multiplier(state)

    def update(self, state, action_mult, reward, next_state, done: bool = False):
        """Store transition and train if buffer has enough samples."""
        if not self._has_torch:
            return
        action_idx = _ACTIONS.index(action_mult) if action_mult in _ACTIONS \
                     else int(np.argmin([abs(a - action_mult) for a in _ACTIONS]))
        self._buf.push(state, action_idx, reward, next_state, done)
        self._step += 1

        if len(self._buf) >= _BATCH_SIZE:
            self._train_step()

        # Decay epsilon
        self.epsilon = max(_EPSILON_MIN, self.epsilon * 0.999)

        # Sync target network
        if self._step % _TARGET_UPDATE == 0:
            self._tgt.load_state_dict(self._net.state_dict())

    def update_from_trade(self, row, df, i: int, capital: float, peak_capital: float,
                          multiplier_used: float, pnl: float, trade_cost: float):
        """Convenience: compute reward and update after a trade closes."""
        state      = _build_state(row, df, i, capital, peak_capital)
        next_state = state.copy()   # approximation
        # Sharpe-like reward: penalize large losses more
        reward = pnl / max(abs(trade_cost), 1) if trade_cost > 0 else pnl / 1000
        reward = np.clip(reward, -5, 5)
        self.update(state, multiplier_used, float(reward), next_state, False)

    def _train_step(self):
        import torch
        import torch.nn.functional as F
        s, a, r, ns, d = self._buf.sample(_BATCH_SIZE)
        s  = torch.tensor(s)
        a  = torch.tensor(a, dtype=torch.long)
        r  = torch.tensor(r)
        ns = torch.tensor(ns)
        d  = torch.tensor(d)

        q_vals    = self._net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._tgt(ns).max(1)[0]
            target = r + _GAMMA * next_q * (1 - d)

        loss = F.mse_loss(q_vals, target)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
        self._opt.step()

    # ------------------------------------------------------------------
    def save(self, path: str = "models/dqn_agent.pt"):
        if not self._has_torch:
            return
        import torch
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            "net":     self._net.state_dict(),
            "epsilon": self.epsilon,
            "step":    self._step,
        }, path)

    def load(self, path: str = "models/dqn_agent.pt"):
        if not self._has_torch or not Path(path).exists():
            return
        import torch
        ckpt = torch.load(path, map_location="cpu")
        self._net.load_state_dict(ckpt["net"])
        self._tgt.load_state_dict(ckpt["net"])
        self.epsilon = ckpt.get("epsilon", _EPSILON_MIN)
        self._step   = ckpt.get("step", 0)
