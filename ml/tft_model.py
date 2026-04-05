"""Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

_SEQ_LEN    = 30
_PRED_LEN   = 5     # predict next 5 bars
_D_MODEL    = 64
_N_HEADS    = 4
_EPOCHS     = 40
_BATCH      = 32

_STATIC_COLS  = []   # per-symbol static features (none for now)
_TEMPORAL_COLS = [
    "close", "volume", "rsi", "macd", "atr", "adx",
    "bb_upper", "bb_lower", "composite_score",
    "volume_ratio", "momentum_score",
]


def _normalize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu  = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return (arr - mu) / std, mu, std


def _build_sequences(df: pd.DataFrame, seq_len: int = _SEQ_LEN, pred_len: int = _PRED_LEN):
    cols = [c for c in _TEMPORAL_COLS if c in df.columns]
    arr  = df[cols].fillna(0).values.astype(np.float32)
    arr, mu, std = _normalize(arr)

    X, y = [], []
    for i in range(seq_len, len(arr) - pred_len):
        X.append(arr[i - seq_len:i])
        # label: 1 if price is higher at pred_len bars ahead
        future = df["close"].iloc[i + pred_len]
        now    = df["close"].iloc[i]
        y.append(1 if future > now * 1.003 else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), mu, std, cols


class TemporalFusionTransformer:
    """
    TFT for stock direction prediction.

    Architecture:
      - Variable Selection Network (VSN) — learns which features matter
      - LSTM encoder for temporal context
      - Multi-head self-attention for long-range dependencies
      - Quantile output (down/neutral/up)

    Falls back to attention-LSTM (sklearn MLP) if torch unavailable.
    """

    def __init__(self, seq_len: int = _SEQ_LEN, pred_len: int = _PRED_LEN):
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.is_fitted = False
        self._model    = None
        self._mu       = None
        self._std      = None
        self._cols: list[str] = []
        self._use_torch = False

    # ------------------------------------------------------------------
    def fit(self, df_list: list[pd.DataFrame], verbose: bool = False):
        all_X, all_y, all_mu, all_std = [], [], [], []
        for df in df_list:
            if df is None or len(df) < self.seq_len + self.pred_len + 10:
                continue
            X, y, mu, std, cols = _build_sequences(df, self.seq_len, self.pred_len)
            if len(X) > 0:
                all_X.append(X); all_y.append(y)
                all_mu.append(mu); all_std.append(std)
                self._cols = cols

        if not all_X:
            logger.warning("TFT: no training data")
            return

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        self._mu  = np.mean(all_mu,  axis=0)
        self._std = np.mean(all_std, axis=0)

        try:
            self._fit_torch(X_all, y_all, verbose)
            self._use_torch = True
        except Exception:
            self._fit_mlp(X_all, y_all)

        self.is_fitted = True
        if verbose:
            logger.info(f"TFT fitted: {len(X_all)} samples, "
                        f"{'torch' if self._use_torch else 'mlp'} backend")

    def predict_proba(self, row, df: pd.DataFrame, i: int) -> float:
        if not self.is_fitted or i < self.seq_len:
            return 0.5
        try:
            cols = [c for c in self._cols if c in df.columns]
            arr  = df[cols].fillna(0).iloc[max(0, i - self.seq_len):i].values.astype(np.float32)
            if len(arr) < self.seq_len:
                return 0.5
            arr = (arr - self._mu[:len(cols)]) / (self._std[:len(cols)] + 1e-8)
            x   = arr[-self.seq_len:][np.newaxis]

            if self._use_torch:
                return self._predict_torch(x)
            return self._predict_mlp(x)
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    def _fit_torch(self, X: np.ndarray, y: np.ndarray, verbose: bool):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        n_feat = X.shape[2]

        class VSN(nn.Module):
            """Variable Selection Network."""
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(n_feat, n_feat)
                self.sm = nn.Softmax(dim=-1)
            def forward(self, x):
                w = self.sm(self.fc(x))
                return x * w

        class _TFT(nn.Module):
            def __init__(self):
                super().__init__()
                self.vsn  = VSN()
                self.lstm = nn.LSTM(n_feat, _D_MODEL, num_layers=2,
                                    batch_first=True, dropout=0.1)
                self.attn = nn.MultiheadAttention(_D_MODEL, _N_HEADS,
                                                   batch_first=True, dropout=0.1)
                self.norm = nn.LayerNorm(_D_MODEL)
                self.fc   = nn.Sequential(
                    nn.Linear(_D_MODEL, 32), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(32, 1), nn.Sigmoid()
                )

            def forward(self, x):
                x          = self.vsn(x)
                lstm_out, _ = self.lstm(x)
                attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
                out        = self.norm(lstm_out + attn_out)
                ctx        = out[:, -1, :]   # last timestep
                return self.fc(ctx).squeeze(-1)

        Xt = torch.tensor(X)
        yt = torch.tensor(y, dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH, shuffle=True)

        net = _TFT()
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, _EPOCHS)
        loss_fn = nn.BCELoss()

        net.train()
        for _ in range(_EPOCHS):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sch.step()

        net.eval()
        self._model = net

    def _predict_torch(self, x: np.ndarray) -> float:
        import torch
        with torch.no_grad():
            return float(self._model(torch.tensor(x)).item())

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray):
        from sklearn.neural_network import MLPClassifier
        X_flat = X.reshape(len(X), -1)
        self._model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=300,
            early_stopping=True, validation_fraction=0.1, random_state=42
        )
        self._model.fit(X_flat, y)

    def _predict_mlp(self, x: np.ndarray) -> float:
        return float(self._model.predict_proba(x.reshape(1, -1))[0][1])
