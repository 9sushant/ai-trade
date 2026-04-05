"""Full RL trading environment — agent learns to trade, hold, and exit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

# Actions
ACTION_HOLD  = 0
ACTION_BUY   = 1
ACTION_SELL  = 2
ACTION_CLOSE = 3

_STATE_DIM   = 18
_HIDDEN      = 128
_GAMMA       = 0.99
_LR          = 3e-4
_CLIP_EPS    = 0.2    # PPO clip
_EPOCHS_PPO  = 4
_BATCH_SIZE  = 64


def _get_state(df: pd.DataFrame, i: int, position: int,
               entry_price: float, capital: float, peak: float) -> np.ndarray:
    row      = df.iloc[i]
    close    = float(row.get("close",  0) or 0)
    rsi      = float(row.get("rsi",   50) or 50) / 100
    adx      = float(row.get("adx",   25) or 25) / 100
    atr_pct  = float(row.get("atr",    0) or 0) / max(close, 1)
    score    = float(row.get("composite_score", 0) or 0) / 100
    vol_r    = min(float(row.get("volume_ratio", 1) or 1), 5) / 5
    mom      = np.clip(float(row.get("momentum_score", 0) or 0), -1, 1)
    drawdown = (peak - capital) / peak if peak > 0 else 0
    unreal_pnl = 0.0
    if position != 0 and entry_price > 0:
        unreal_pnl = (close - entry_price) / entry_price * np.sign(position)

    ret_5  = float(df["close"].iloc[i] / df["close"].iloc[max(0, i-5)]  - 1) if i >= 5  else 0
    ret_10 = float(df["close"].iloc[i] / df["close"].iloc[max(0, i-10)] - 1) if i >= 10 else 0
    ret_20 = float(df["close"].iloc[i] / df["close"].iloc[max(0, i-20)] - 1) if i >= 20 else 0

    bb_pos = 0.5
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bu, bl = float(row.get("bb_upper", 0) or 0), float(row.get("bb_lower", 0) or 0)
        if bu > bl:
            bb_pos = (close - bl) / (bu - bl)

    ema_cross = 0.0
    if "ema_9" in df.columns and "ema_21" in df.columns:
        e9  = float(row.get("ema_9",  close) or close)
        e21 = float(row.get("ema_21", close) or close)
        ema_cross = (e9 - e21) / max(close, 1)

    return np.array([
        rsi, adx, atr_pct, np.clip(score, -1, 1),
        vol_r, mom, np.clip(bb_pos, 0, 1),
        np.clip(ret_5,  -0.1, 0.1) / 0.1,
        np.clip(ret_10, -0.15, 0.15) / 0.15,
        np.clip(ret_20, -0.2, 0.2) / 0.2,
        np.clip(ema_cross, -0.05, 0.05) / 0.05,
        float(position),        # -1 short / 0 flat / 1 long
        np.clip(unreal_pnl, -0.1, 0.1) / 0.1,
        np.clip(drawdown, 0, 0.2) / 0.2,
        capital / 100_000,
        float(i / max(len(df) - 1, 1)),   # time in episode
        1.0 if rsi > 0.7 else 0.0,        # overbought
        1.0 if rsi < 0.3 else 0.0,        # oversold
    ], dtype=np.float32)


class TradingEnvironment:
    """OpenAI Gym-style trading environment."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 50_000,
                 max_risk_pct: float = 0.02, transaction_cost: float = 0.0013):
        self.df               = df.reset_index(drop=True)
        self.initial_capital  = initial_capital
        self.max_risk_pct     = max_risk_pct
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.capital      = self.initial_capital
        self.peak         = self.capital
        self.position     = 0    # -1, 0, +1
        self.entry_price  = 0.0
        self.step_idx     = 20
        self.done         = False
        self.total_reward = 0.0
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._state(), 0.0, True, {}

        row       = self.df.iloc[self.step_idx]
        close     = float(row.get("close", 0) or 0)
        atr       = float(row.get("atr", close * 0.01) or close * 0.01)
        prev_cap  = self.capital
        reward    = 0.0
        info      = {}

        # ── Execute action ────────────────────────────────────────────
        if action == ACTION_BUY and self.position == 0:
            qty           = max(1, int(self.capital * self.max_risk_pct / (atr + 1e-8)))
            qty           = min(qty, int(self.capital * 0.25 / max(close, 1)))
            cost          = close * qty * self.transaction_cost
            self.capital -= cost
            self.position = 1
            self.entry_price = close
            info["action"] = "BUY"

        elif action == ACTION_SELL and self.position == 0:
            qty           = max(1, int(self.capital * self.max_risk_pct / (atr + 1e-8)))
            qty           = min(qty, int(self.capital * 0.25 / max(close, 1)))
            cost          = close * qty * self.transaction_cost
            self.capital -= cost
            self.position = -1
            self.entry_price = close
            info["action"] = "SELL"

        elif action == ACTION_CLOSE and self.position != 0:
            pnl = (close - self.entry_price) * np.sign(self.position)
            pnl_pct = pnl / max(self.entry_price, 1)
            cost    = close * self.transaction_cost
            self.capital += pnl - cost * abs(self.position)
            self.position = 0
            self.entry_price = 0.0
            info["action"] = "CLOSE"
            info["pnl_pct"] = pnl_pct

        # ── Compute reward ────────────────────────────────────────────
        # Sharpe-like: reward change in capital, penalize drawdown
        cap_change = (self.capital - prev_cap) / max(prev_cap, 1)
        reward     = cap_change * 100

        if self.capital > self.peak:
            self.peak = self.capital
        drawdown = (self.peak - self.capital) / max(self.peak, 1)
        reward  -= drawdown * 50    # penalize drawdown heavily

        # Penalize holding losing position too long
        if self.position != 0 and self.entry_price > 0:
            unreal = (close - self.entry_price) / self.entry_price * np.sign(self.position)
            if unreal < -0.02:
                reward -= 1.0

        self.total_reward += reward

        # ── Advance ───────────────────────────────────────────────────
        self.step_idx += 1
        if self.step_idx >= len(self.df) - 1:
            # Force close at end
            if self.position != 0:
                close_last = float(self.df.iloc[-1].get("close", close) or close)
                pnl = (close_last - self.entry_price) * np.sign(self.position)
                self.capital += pnl
                self.position = 0
            self.done = True

        # Force close on 15% drawdown
        if drawdown > 0.15:
            self.done = True

        return self._state(), float(reward), self.done, info

    def _state(self) -> np.ndarray:
        return _get_state(self.df, self.step_idx, self.position,
                          self.entry_price, self.capital, self.peak)

    @property
    def state_dim(self):
        return _STATE_DIM

    @property
    def n_actions(self):
        return 4


class PPOAgent:
    """
    Proximal Policy Optimization agent for the trading environment.
    Falls back to rule-based agent if torch unavailable.
    """

    def __init__(self):
        self._actor  = None
        self._critic = None
        self._opt_a  = None
        self._opt_c  = None
        self._use_torch = False
        self._init()

    def _init(self):
        try:
            import torch
            import torch.nn as nn

            class _Net(nn.Module):
                def __init__(self, out):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(_STATE_DIM, _HIDDEN), nn.Tanh(),
                        nn.Linear(_HIDDEN, _HIDDEN),   nn.Tanh(),
                        nn.Linear(_HIDDEN, out)
                    )
                def forward(self, x):
                    return self.net(x)

            self._actor  = _Net(4)   # 4 actions
            self._critic = _Net(1)
            self._opt_a  = torch.optim.Adam(self._actor.parameters(),  lr=_LR)
            self._opt_c  = torch.optim.Adam(self._critic.parameters(), lr=_LR)
            self._use_torch = True
        except ImportError:
            pass

    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float]:
        """Return (action, log_prob)."""
        if not self._use_torch:
            return self._rule_act(state), 0.0
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            logits = self._actor(torch.tensor(state).unsqueeze(0))
            probs  = F.softmax(logits, dim=-1).squeeze(0)
            if deterministic:
                action = int(probs.argmax().item())
                return action, float(torch.log(probs[action]).item())
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())

    def train_episode(self, env: TradingEnvironment) -> dict:
        """Collect one episode and update policy."""
        if not self._use_torch:
            return self._run_rule_episode(env)

        import torch
        import torch.nn.functional as F

        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        state = env.reset()
        while not env.done:
            s_t = torch.tensor(state).unsqueeze(0)
            with torch.no_grad():
                logits = self._actor(s_t)
                probs  = F.softmax(logits, dim=-1).squeeze(0)
                dist   = torch.distributions.Categorical(probs)
                action = dist.sample()
                lp     = dist.log_prob(action)
                value  = self._critic(s_t).squeeze()

            next_state, reward, done, _ = env.step(int(action.item()))
            states.append(state); actions.append(int(action.item()))
            log_probs.append(float(lp.item())); rewards.append(reward)
            dones.append(done); values.append(float(value.item()))
            state = next_state

        # GAE returns
        returns = self._gae_returns(rewards, values, dones)

        # PPO update
        S  = torch.tensor(np.array(states))
        A  = torch.tensor(actions, dtype=torch.long)
        LP = torch.tensor(log_probs)
        R  = torch.tensor(returns, dtype=torch.float32)
        V  = torch.tensor(values)
        ADV = (R - V).detach()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        for _ in range(_EPOCHS_PPO):
            idx = torch.randperm(len(S))
            for start in range(0, len(S), _BATCH_SIZE):
                b   = idx[start:start + _BATCH_SIZE]
                logits_new = self._actor(S[b])
                probs_new  = F.softmax(logits_new, dim=-1)
                dist_new   = torch.distributions.Categorical(probs_new)
                lp_new     = dist_new.log_prob(A[b])
                ratio      = (lp_new - LP[b]).exp()
                surr1 = ratio * ADV[b]
                surr2 = ratio.clamp(1 - _CLIP_EPS, 1 + _CLIP_EPS) * ADV[b]
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(self._critic(S[b]).squeeze(), R[b])

                self._opt_a.zero_grad(); actor_loss.backward();  self._opt_a.step()
                self._opt_c.zero_grad(); critic_loss.backward(); self._opt_c.step()

        return {
            "total_reward": env.total_reward,
            "final_capital": env.capital,
            "return_pct": (env.capital / env.initial_capital - 1) * 100,
        }

    def _gae_returns(self, rewards, values, dones,
                     gamma: float = _GAMMA, lam: float = 0.95) -> list[float]:
        returns = []
        gae     = 0
        next_v  = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + gamma * next_v * (1 - d) - v
            gae   = delta + gamma * lam * (1 - d) * gae
            returns.insert(0, gae + v)
            next_v = v
        return returns

    @staticmethod
    def _rule_act(state: np.ndarray) -> int:
        """Simple rule-based fallback."""
        rsi    = state[0] * 100
        score  = state[3]
        pos    = state[11]
        unreal = state[12] * 0.1

        if pos == 0:
            if score > 0.3 and rsi < 65:  return ACTION_BUY
            if score < -0.3 and rsi > 35: return ACTION_SELL
        elif pos != 0:
            if unreal < -0.015: return ACTION_CLOSE   # stop loss
            if unreal >  0.020: return ACTION_CLOSE   # take profit
        return ACTION_HOLD

    def _run_rule_episode(self, env: TradingEnvironment) -> dict:
        state = env.reset()
        while not env.done:
            action, _ = self.act(state)
            state, _, done, _ = env.step(action)
        return {"total_reward": env.total_reward,
                "final_capital": env.capital,
                "return_pct": (env.capital / env.initial_capital - 1) * 100}

    def save(self, path: str = "models/ppo_agent.pt"):
        if not self._use_torch:
            return
        import torch
        from pathlib import Path
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({"actor": self._actor.state_dict(),
                    "critic": self._critic.state_dict()}, path)

    def load(self, path: str = "models/ppo_agent.pt"):
        if not self._use_torch:
            return
        import torch
        from pathlib import Path
        if not Path(path).exists():
            return
        ckpt = torch.load(path, map_location="cpu")
        self._actor.load_state_dict(ckpt["actor"])
        self._critic.load_state_dict(ckpt["critic"])
