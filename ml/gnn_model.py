"""Graph Neural Network — models stock correlations as a graph for signal generation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import logger

_HIDDEN     = 64
_EPOCHS     = 30
_CORR_THRESH = 0.40   # minimum correlation to add an edge


def _build_graph(returns_df: pd.DataFrame, threshold: float = _CORR_THRESH):
    """Build adjacency matrix from return correlations."""
    corr   = returns_df.corr().fillna(0).values
    adj    = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)   # self-loops
    # Normalize: D^{-1/2} A D^{-1/2}
    deg    = adj.sum(axis=1)
    d_inv  = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-8)))
    return d_inv @ adj @ d_inv


class GNNStockModel:
    """
    Graph Convolutional Network for stock signal generation.

    Each node = one stock. Node features = technical indicators.
    Edges = correlation above threshold.
    Output = per-node buy probability.

    Falls back to correlation-weighted ensemble if torch unavailable.
    """

    def __init__(self, hidden: int = _HIDDEN, n_layers: int = 2):
        self.hidden    = hidden
        self.n_layers  = n_layers
        self.is_fitted = False
        self._model    = None
        self._symbols: list[str] = []
        self._adj: np.ndarray | None = None
        self._feat_mu: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._use_torch = False
        self._n_feat    = 0

    # ------------------------------------------------------------------
    def fit(self, enriched_dfs: dict[str, pd.DataFrame], verbose: bool = False):
        symbols = list(enriched_dfs.keys())
        if len(symbols) < 3:
            return

        # Build return matrix for graph edges
        returns = {}
        for sym, df in enriched_dfs.items():
            ret = df["close"].pct_change().dropna()
            if len(ret) > 50:
                returns[sym] = ret

        returns_df = pd.DataFrame(returns).dropna()
        if len(returns_df.columns) < 3:
            return

        self._symbols = list(returns_df.columns)
        self._adj     = _build_graph(returns_df)

        # Node features: last-bar indicators
        feat_cols = ["rsi", "adx", "volume_ratio", "composite_score",
                     "atr", "momentum_score", "mean_rev_z"]
        X_rows, y_rows = [], []
        for sym in self._symbols:
            df = enriched_dfs.get(sym)
            if df is None or len(df) < 60:
                X_rows.append(np.zeros(len(feat_cols)))
                y_rows.append(0)
                continue
            row = df.iloc[-1]
            feat = np.array([float(row.get(c, 0) or 0) for c in feat_cols], dtype=np.float32)
            X_rows.append(feat)
            # Label: next-bar positive return
            if len(df) >= 2:
                y_rows.append(1 if df["close"].iloc[-1] > df["close"].iloc[-2] else 0)
            else:
                y_rows.append(0)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)
        self._n_feat   = X.shape[1]
        self._feat_mu  = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feat_mu) / self._feat_std

        try:
            self._fit_torch(X_norm, y, verbose)
            self._use_torch = True
        except Exception:
            self._fit_simple(X_norm, y)

        self.is_fitted = True
        if verbose:
            logger.info(f"GNN fitted: {len(self._symbols)} nodes, "
                        f"{'torch' if self._use_torch else 'simple'} backend")

    def predict_all(self, enriched_dfs: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Return buy probability for each symbol."""
        if not self.is_fitted:
            return {}

        feat_cols = ["rsi", "adx", "volume_ratio", "composite_score",
                     "atr", "momentum_score", "mean_rev_z"]
        X_rows = []
        for sym in self._symbols:
            df  = enriched_dfs.get(sym)
            row = df.iloc[-1] if df is not None and len(df) > 0 else {}
            feat = np.array([float(row.get(c, 0) or 0) if hasattr(row, 'get') else 0
                             for c in feat_cols], dtype=np.float32)
            X_rows.append(feat)

        X = np.array(X_rows, dtype=np.float32)
        X = (X - self._feat_mu) / self._feat_std

        if self._use_torch:
            probs = self._predict_torch(X)
        else:
            probs = self._predict_simple(X)

        return {sym: round(float(p), 4) for sym, p in zip(self._symbols, probs)}

    def predict_symbol(self, symbol: str, enriched_dfs: dict[str, pd.DataFrame]) -> float:
        probs = self.predict_all(enriched_dfs)
        return probs.get(symbol, 0.5)

    # ------------------------------------------------------------------
    def _fit_torch(self, X: np.ndarray, y: np.ndarray, verbose: bool):
        import torch
        import torch.nn as nn

        adj_t = torch.tensor(self._adj)

        class GCNLayer(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.fc = nn.Linear(in_f, out_f, bias=False)
                self.bn = nn.BatchNorm1d(out_f)
            def forward(self, x, adj):
                return torch.relu(self.bn(adj @ self.fc(x)))

        class _GCN(nn.Module):
            def __init__(self, n_feat, hidden, n_layers):
                super().__init__()
                layers = [GCNLayer(n_feat, hidden)]
                for _ in range(n_layers - 1):
                    layers.append(GCNLayer(hidden, hidden))
                self.gcn = nn.ModuleList(layers)
                self.fc  = nn.Sequential(
                    nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(32, 1), nn.Sigmoid()
                )
            def forward(self, x, adj):
                for layer in self.gcn:
                    x = layer(x, adj)
                return self.fc(x).squeeze(-1)

        Xt = torch.tensor(X)
        yt = torch.tensor(y, dtype=torch.float32)

        net = _GCN(self._n_feat, self.hidden, self.n_layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        net.train()
        for _ in range(_EPOCHS):
            opt.zero_grad()
            pred = net(Xt, adj_t)
            loss_fn(pred, yt).backward()
            opt.step()

        net.eval()
        self._model = (net, adj_t)

    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        import torch
        net, adj_t = self._model
        with torch.no_grad():
            return net(torch.tensor(X), adj_t).numpy()

    def _fit_simple(self, X: np.ndarray, y: np.ndarray):
        """Correlation-weighted logistic regression fallback."""
        from sklearn.linear_model import LogisticRegression
        # Propagate features through adjacency
        X_prop = self._adj @ X
        self._model = LogisticRegression(max_iter=200, random_state=42)
        try:
            self._model.fit(X_prop, y)
        except Exception:
            self._model = None

    def _predict_simple(self, X: np.ndarray) -> np.ndarray:
        X_prop = self._adj @ X
        if self._model is None:
            return np.full(len(X), 0.5)
        try:
            return self._model.predict_proba(X_prop)[:, 1]
        except Exception:
            return np.full(len(X), 0.5)
