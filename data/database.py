"""
SQLite trade database — persists all trades, signals, and equity curve.
Thread-safe with WAL mode for concurrent reads during live trading.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from utils.logger import logger


class TradeDatabase:
    def __init__(self, db_path: str = "logs/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT,
                    entry_date TEXT, exit_date TEXT,
                    entry_price REAL, exit_price REAL,
                    quantity INTEGER, pnl REAL, pnl_pct REAL,
                    exit_reason TEXT, composite_score REAL,
                    atr REAL, rr_ratio REAL, hold_days INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, date TEXT, direction TEXT,
                    score REAL, ml_prob REAL, regime TEXT,
                    entry REAL, stop_loss REAL, target REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, equity REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, metric_name TEXT, value REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_entry  ON trades(entry_date);
                CREATE INDEX IF NOT EXISTS idx_equity_date   ON equity_curve(date);
            """)

    # ------------------------------------------------------------------
    def save_trade(self, trade: dict):
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO trades
                  (symbol,direction,entry_date,exit_date,entry_price,exit_price,
                   quantity,pnl,pnl_pct,exit_reason,composite_score,atr,rr_ratio,hold_days)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade.get("symbol"), trade.get("direction"),
                str(trade.get("entry_date",""))[:10], str(trade.get("exit_date",""))[:10],
                trade.get("entry_price"), trade.get("exit_price"),
                trade.get("quantity"),  trade.get("pnl"), trade.get("pnl_pct"),
                trade.get("exit_reason"), trade.get("composite_score"),
                trade.get("atr"), trade.get("rr_ratio"), trade.get("hold_days"),
            ))

    def save_signal(self, signal: dict):
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT INTO signals (symbol,date,direction,score,ml_prob,regime,entry,stop_loss,target)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                signal.get("symbol"), signal.get("date", datetime.now().strftime("%Y-%m-%d")),
                signal.get("direction"), signal.get("score"), signal.get("ml_prob"),
                signal.get("regime"), signal.get("entry"),
                signal.get("stop_loss"), signal.get("target"),
            ))

    def save_equity_point(self, date, equity: float):
        with self._lock, self._conn() as conn:
            conn.execute("INSERT INTO equity_curve (date,equity) VALUES (?,?)",
                         (str(date)[:10], equity))

    def save_metric(self, metric_name: str, value: float):
        with self._lock, self._conn() as conn:
            conn.execute("INSERT INTO model_metrics (date,metric_name,value) VALUES (?,?,?)",
                         (datetime.now().strftime("%Y-%m-%d"), metric_name, value))

    # ------------------------------------------------------------------
    def get_trades(self, symbol: str = None, start_date: str = None,
                   end_date: str = None) -> list[dict]:
        clauses, params = [], []
        if symbol:
            clauses.append("symbol=?"); params.append(symbol)
        if start_date:
            clauses.append("entry_date>=?"); params.append(start_date)
        if end_date:
            clauses.append("entry_date<=?"); params.append(end_date)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(f"SELECT * FROM trades {where} ORDER BY entry_date", params).fetchall()
            return [dict(r) for r in rows]

    def get_equity_curve(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT date,equity FROM equity_curve ORDER BY date").fetchall()
            return [dict(r) for r in rows]

    def get_symbol_stats(self, symbol: str) -> dict:
        with self._conn() as conn:
            rows = conn.execute("SELECT pnl FROM trades WHERE symbol=?", (symbol,)).fetchall()
        if not rows:
            return {"symbol": symbol, "trade_count": 0, "win_rate": 0, "total_pnl": 0}
        pnls       = [r["pnl"] for r in rows]
        wins       = [p for p in pnls if p > 0]
        return {
            "symbol":       symbol,
            "trade_count":  len(pnls),
            "win_rate":     round(len(wins) / len(pnls) * 100, 1),
            "total_pnl":    round(sum(pnls), 2),
            "avg_pnl":      round(sum(pnls) / len(pnls), 2),
        }

    def get_all_stats(self) -> dict:
        trades = self.get_trades()
        if not trades:
            return {"total_trades": 0}
        pnls  = [t["pnl"] for t in trades]
        wins  = [p for p in pnls if p > 0]
        curve = self.get_equity_curve()
        final = curve[-1]["equity"] if curve else 50000
        init  = curve[0]["equity"]  if curve else 50000
        return {
            "total_trades": len(trades),
            "win_rate":     round(len(wins) / len(trades) * 100, 1),
            "total_pnl":    round(sum(pnls), 2),
            "return_pct":   round((final - init) / init * 100, 2),
            "final_equity": final,
        }

    def save_backtest_results(self, results: dict):
        """Bulk-save all trades and equity curve from a backtest run."""
        for t in results.get("trades", []):
            try:
                self.save_trade(t)
            except Exception:
                pass
        for point in results.get("equity_curve", []):
            try:
                self.save_equity_point(point["date"], point["equity"])
            except Exception:
                pass
        logger.info(f"DB: saved {len(results.get('trades',[]))} trades")
