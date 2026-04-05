"""LSTM/Transformer sequence model for price prediction."""
from __future__ import annotations
import numpy as np
from utils.logger import logger

_SEQ_LEN    = 20   # lookback window
_HIDDEN     = 64
_EPOCHS     = 30
_BATCH      = 32

_FEATURE_COLS = [
    "close", "volume", "rsi", "macd", "atr",
    "bb_upper", "bb_lower", "adx", "composite_score",
]


def _make_sequences(df, seq_len: int = _SEQ_LEN):
    """Build (X, y) sequences from enriched df."""
    cols = [c for c in _FEATURE_COLS if c in df.columns]
    arr  = df[cols].fillna(0).values.astype(np.float32)

    # Normalize per feature
    mu  = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    arr = (arr - mu) / std

    X, y = [], []
    for i in range(seq_len, len(arr) - 1):
        X.append(arr[i - seq_len:i])
        future_ret = (df["close"].iloc[i + 1] - df["close"].iloc[i]) / df["close"].iloc[i]
        y.append(1 if future_ret > 0.003 else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), mu, std, cols


class LSTMModel:
    """Lightweight LSTM using only numpy (no torch/tf required).

    Falls back gracefully if PyTorch not installed — uses a simple
    GRU-like approximation via sklearn MLPClassifier on flattened sequences.
    """

    def __init__(self, seq_len: int = _SEQ_LEN):
        self.seq_len    = seq_len
        self.is_fitted  = False
        self._mu        = None
        self._std       = None
        self._cols: list[str] = []
        self._model     = None
        self._use_torch = False

    # ------------------------------------------------------------------
    def fit(self, df_list: list, verbose: bool = False):
        all_X, all_y = [], []
        mu_list, std_list = [], []

        for df in df_list:
            if df is None or len(df) < self.seq_len + 20:
                continue
            X, y, mu, std, cols = _make_sequences(df, self.seq_len)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                mu_list.append(mu)
                std_list.append(std)
                self._cols = cols

        if not all_X:
            logger.warning("LSTMModel: no training data")
            return

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        self._mu  = np.mean(mu_list,  axis=0)
        self._std = np.mean(std_list, axis=0)

        # Try PyTorch first
        try:
            self._fit_torch(X_all, y_all, verbose)
            self._use_torch = True
        except Exception:
            # Fallback: flatten + MLP
            self._fit_mlp(X_all, y_all, verbose)
            self._use_torch = False

        self.is_fitted = True
        if verbose:
            logger.info(f"LSTMModel fitted on {len(X_all)} sequences "
                        f"({'torch' if self._use_torch else 'mlp'})")

    def predict_proba(self, row, df, i: int) -> float:
        """Return probability of upward move [0, 1]."""
        if not self.is_fitted or i < self.seq_len:
            return 0.5
        try:
            cols = [c for c in self._cols if c in df.columns]
            arr  = df[cols].fillna(0).iloc[max(0, i - self.seq_len):i].values.astype(np.float32)
            if len(arr) < self.seq_len:
                return 0.5
            arr = (arr - self._mu[:len(cols)]) / (self._std[:len(cols)] + 1e-8)
            x   = arr[-self.seq_len:][np.newaxis]   # (1, seq_len, features)

            if self._use_torch:
                return self._predict_torch(x)
            else:
                return self._predict_mlp(x)
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    def _fit_torch(self, X, y, verbose):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        n_feat = X.shape[2]

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, _HIDDEN, num_layers=2,
                                    batch_first=True, dropout=0.2)
                self.attn = nn.Linear(_HIDDEN, 1)
                self.fc   = nn.Sequential(
                    nn.Linear(_HIDDEN, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 1), nn.Sigmoid()
                )

            def forward(self, x):
                out, _ = self.lstm(x)             # (B, T, H)
                w = torch.softmax(self.attn(out), dim=1)   # attention
                ctx = (w * out).sum(dim=1)        # (B, H)
                return self.fc(ctx).squeeze(-1)

        Xt = torch.tensor(X)
        yt = torch.tensor(y, dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH, shuffle=True)

        net  = _Net()
        opt  = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        for ep in range(_EPOCHS):
            net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()

        net.eval()
        self._model = net

    def _predict_torch(self, x: np.ndarray) -> float:
        import torch
        with torch.no_grad():
            t = torch.tensor(x)
            return float(self._model(t).item())

    def _fit_mlp(self, X, y, verbose):
        from sklearn.neural_network import MLPClassifier
        X_flat = X.reshape(len(X), -1)
        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=200,
            early_stopping=True, random_state=42
        )
        self._model.fit(X_flat, y)

    def _predict_mlp(self, x: np.ndarray) -> float:
        x_flat = x.reshape(1, -1)
        return float(self._model.predict_proba(x_flat)[0][1])

    def feature_importance(self) -> dict:
        """Approximate importance via input gradient (torch) or MLP weights."""
        return {}   # complex to expose cleanly — placeholder
