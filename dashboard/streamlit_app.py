"""
AI Trade - Live Streamlit Dashboard

Run with:
    streamlit run dashboard/streamlit_app.py

Features:
  - Live equity curve with auto-refresh
  - Open positions table
  - Today's signals
  - Per-symbol P&L breakdown
  - Regime indicator
  - ML model stats
  - Walk-forward optimization results
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# ===========================================================================
# Page config
# ===========================================================================

def setup_page():
    st.set_page_config(
        page_title  = "AI Trade Dashboard",
        page_icon   = "📈",
        layout      = "wide",
        initial_sidebar_state = "expanded",
    )
    st.markdown("""
    <style>
    .metric-card { background: #1e1e2e; border-radius: 8px; padding: 16px; margin: 4px; }
    .positive    { color: #00ff88; font-weight: bold; }
    .negative    { color: #ff4444; font-weight: bold; }
    .neutral     { color: #aaaaaa; }
    div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 8px; padding: 8px; }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# Data loaders (cached with ttl)
# ===========================================================================

@st.cache_data(ttl=30)
def load_equity_curve():
    try:
        from data.database import TradeDatabase
        db = TradeDatabase()
        return db.get_equity_curve()
    except Exception:
        # Return demo data if DB not initialised
        dates  = pd.date_range(end=datetime.now(), periods=60, freq="D")
        equity = [50000.0]
        np.random.seed(42)
        for _ in range(59):
            equity.append(equity[-1] * (1 + np.random.normal(0.002, 0.008)))
        return [{"date": d, "equity": e} for d, e in zip(dates, equity)]


@st.cache_data(ttl=30)
def load_trades():
    try:
        from data.database import TradeDatabase
        db = TradeDatabase()
        return db.get_trades()
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_top_signals():
    try:
        from screener.stock_screener import StockScreener
        screener = StockScreener()
        return screener.get_top_buy_picks(count=5)
    except Exception:
        return []


@st.cache_data(ttl=300)
def load_regime():
    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine()
        engine._load_nifty_regime("3mo")
        regime = engine._get_market_regime(datetime.now())
        return regime
    except Exception:
        return "UNKNOWN"


@st.cache_data(ttl=120)
def run_quick_backtest():
    try:
        from backtest.engine import BacktestEngine
        from config.settings import MarketConfig
        engine = BacktestEngine(
            use_ml_filter=False,
            use_mtf_filter=False,
            use_fundamental_filter=False,
        )
        return engine.run(
            symbols=MarketConfig.NIFTY_50_SYMBOLS[:10],
            period="6mo",
        )
    except Exception:
        return {}


# ===========================================================================
# Chart helpers
# ===========================================================================

def equity_chart(curve_data: list[dict]):
    if not curve_data:
        return go.Figure()

    df = pd.DataFrame(curve_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    initial = df["equity"].iloc[0]
    color   = "#00ff88" if df["equity"].iloc[-1] >= initial else "#ff4444"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x          = df["date"],
        y          = df["equity"],
        mode       = "lines",
        line       = dict(color=color, width=2),
        fill       = "tozeroy",
        fillcolor  = color.replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in color
                      else f"rgba(0,255,136,0.1)" if color == "#00ff88" else "rgba(255,68,68,0.1)",
        name       = "Equity",
        hovertemplate = "₹%{y:,.0f}<br>%{x|%d %b %Y}<extra></extra>",
    ))
    fig.add_hline(
        y          = initial,
        line_dash  = "dash",
        line_color = "#666",
        annotation_text = f"Initial: ₹{initial:,.0f}",
    )
    fig.update_layout(
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="#ffffff"),
        xaxis         = dict(gridcolor="#2a2a3e", showgrid=True),
        yaxis         = dict(gridcolor="#2a2a3e", showgrid=True, tickprefix="₹", tickformat=","),
        margin        = dict(l=10, r=10, t=10, b=10),
        height        = 300,
        showlegend    = False,
    )
    return fig


def symbol_pnl_chart(trades: list[dict]):
    if not trades:
        return go.Figure()

    df      = pd.DataFrame(trades)
    by_sym  = df.groupby("symbol")["pnl"].sum().sort_values()

    colors  = ["#00ff88" if v >= 0 else "#ff4444" for v in by_sym.values]
    fig     = go.Figure(go.Bar(
        x           = by_sym.values,
        y           = by_sym.index,
        orientation = "h",
        marker_color= colors,
        hovertemplate = "%{y}: ₹%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="#ffffff"),
        xaxis         = dict(gridcolor="#2a2a3e", tickprefix="₹", tickformat=","),
        yaxis         = dict(gridcolor="#2a2a3e"),
        margin        = dict(l=10, r=10, t=10, b=10),
        height        = max(200, len(by_sym) * 30),
        showlegend    = False,
    )
    return fig


def drawdown_chart(curve_data: list[dict]):
    if not curve_data:
        return go.Figure()

    df          = pd.DataFrame(curve_data)
    df["date"]  = pd.to_datetime(df["date"])
    df          = df.sort_values("date")
    peak        = df["equity"].cummax()
    drawdown    = (df["equity"] - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x         = df["date"],
        y         = drawdown,
        fill      = "tozeroy",
        fillcolor = "rgba(255,68,68,0.3)",
        line      = dict(color="#ff4444", width=1),
        name      = "Drawdown %",
        hovertemplate = "%{y:.2f}%<br>%{x|%d %b}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="#ffffff"),
        xaxis         = dict(gridcolor="#2a2a3e"),
        yaxis         = dict(gridcolor="#2a2a3e", ticksuffix="%"),
        margin        = dict(l=10, r=10, t=10, b=10),
        height        = 150,
        showlegend    = False,
    )
    return fig


# ===========================================================================
# Main dashboard
# ===========================================================================

def main():
    if not HAS_STREAMLIT:
        print("Install streamlit: pip install streamlit plotly")
        return

    setup_page()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ AI Trade")
        st.caption("NSE Algorithmic Trading System")
        st.divider()

        auto_refresh = st.toggle("Auto Refresh (30s)", value=True)
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        st.divider()
        regime = load_regime()
        regime_color = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(regime, "⚪")
        st.metric("Market Regime", f"{regime_color} {regime}")

        st.divider()
        period = st.selectbox("Backtest Period", ["6mo", "1y", "2y"], index=1)
        if st.button("Run Backtest", use_container_width=True):
            st.cache_data.clear()

    # ── Header metrics ────────────────────────────────────────────────────
    st.title("📈 AI Trade Dashboard")

    curve_data = load_equity_curve()
    trades     = load_trades()

    if curve_data:
        initial = curve_data[0]["equity"]
        final   = curve_data[-1]["equity"]
        pnl     = final - initial
        ret_pct = pnl / initial * 100

        wins    = [t for t in trades if t.get("pnl", 0) > 0] if trades else []
        losses  = [t for t in trades if t.get("pnl", 0) <= 0] if trades else []
        wr      = len(wins) / max(len(trades), 1) * 100

        equities = [d["equity"] for d in curve_data]
        peak     = max(equities)
        mdd      = (peak - min(equities[equities.index(peak):])) / peak * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Portfolio Value", f"₹{final:,.0f}",
                    f"{'+'if pnl>=0 else ''}{pnl:,.0f} ({ret_pct:+.2f}%)")
        col2.metric("Total Trades", len(trades))
        col3.metric("Win Rate", f"{wr:.1f}%",
                    f"+{len(wins)}W / -{len(losses)}L")
        col4.metric("Max Drawdown", f"{mdd:.2f}%",
                    "Good" if mdd < 3 else "Watch" if mdd < 6 else "High",
                    delta_color="inverse")
        col5.metric("Avg Daily P&L",
                    f"₹{np.mean([t.get('pnl',0) for t in trades]):.0f}" if trades else "N/A")

    st.divider()

    # ── Main content ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Equity Curve", "🎯 Signals", "📋 Trade History", "🔬 Analytics"]
    )

    with tab1:
        col_eq, col_dd = st.columns([3, 1])
        with col_eq:
            st.subheader("Equity Curve")
            st.plotly_chart(equity_chart(curve_data), use_container_width=True)
        with col_dd:
            st.subheader("Drawdown")
            st.plotly_chart(drawdown_chart(curve_data), use_container_width=True)

        st.subheader("P&L by Symbol")
        if trades:
            st.plotly_chart(symbol_pnl_chart(trades), use_container_width=True)
        else:
            st.info("No trades in database yet. Run a backtest to populate.")

    with tab2:
        st.subheader("Today's Top Signals")
        with st.spinner("Scanning market..."):
            signals = load_top_signals()

        if signals:
            sig_df = pd.DataFrame([{
                "Symbol":     s.get("symbol"),
                "Direction":  s.get("recommendation", ""),
                "Price":      f"₹{s.get('close', 0):,.2f}",
                "Score":      s.get("composite_score", 0),
                "RSI":        s.get("rsi", 0),
                "ADX":        s.get("adx", 0),
                "Fund":       s.get("fund_label", "N/A"),
                "Sentiment":  s.get("sent_label", "N/A"),
            } for s in signals])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No strong signals found right now.")

        st.subheader("Opening Range Breakout Scanner")
        if st.button("Scan ORB Signals"):
            try:
                from analysis.orb import ORBStrategy
                from data.fetcher import DataFetcher
                from config.settings import MarketConfig
                orb     = ORBStrategy()
                fetcher = DataFetcher()
                results = orb.scan_orb_signals(MarketConfig.NIFTY_50_SYMBOLS[:10], fetcher)
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                else:
                    st.info("No ORB signals found yet (market may not be open)")
            except Exception as e:
                st.warning(f"ORB scan error: {e}")

    with tab3:
        st.subheader("Trade History")
        if trades:
            df = pd.DataFrame(trades)
            for col in ["entry_date", "exit_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).dt.strftime("%d %b %Y")

            display_cols = [c for c in
                ["symbol","direction","entry_date","exit_date","entry_price","exit_price",
                 "quantity","pnl","pnl_pct","exit_reason","hold_days"]
                if c in df.columns]

            def color_pnl(val):
                if isinstance(val, (int, float)):
                    return "color: #00ff88" if val > 0 else "color: #ff4444" if val < 0 else ""
                return ""

            styled = df[display_cols].style.applymap(color_pnl, subset=["pnl","pnl_pct"] if "pnl" in display_cols else [])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "trades.csv", "text/csv")
        else:
            st.info("No trades yet. Trades will appear here after running the backtest.")

    with tab4:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Win Rate by Symbol")
            if trades:
                df      = pd.DataFrame(trades)
                wr_sym  = df.groupby("symbol").apply(
                    lambda x: (x["pnl"] > 0).mean() * 100
                ).sort_values(ascending=False)
                fig_wr  = px.bar(
                    x      = wr_sym.values,
                    y      = wr_sym.index,
                    orientation = "h",
                    labels = {"x": "Win Rate %", "y": "Symbol"},
                    color  = wr_sym.values,
                    color_continuous_scale = ["#ff4444", "#ffaa00", "#00ff88"],
                )
                fig_wr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                     font=dict(color="#fff"), showlegend=False,
                                     coloraxis_showscale=False, height=300)
                st.plotly_chart(fig_wr, use_container_width=True)

        with col_b:
            st.subheader("Exit Reason Distribution")
            if trades:
                df        = pd.DataFrame(trades)
                exit_cnts = df["exit_reason"].value_counts()
                fig_exit  = px.pie(
                    values = exit_cnts.values,
                    names  = exit_cnts.index,
                    color_discrete_sequence = ["#00ff88","#ff4444","#ffaa00","#4488ff"],
                )
                fig_exit.update_layout(paper_bgcolor="#0e1117", font=dict(color="#fff"),
                                       height=300)
                st.plotly_chart(fig_exit, use_container_width=True)

        st.subheader("Pairs Trading Opportunities")
        if st.button("Find Pairs"):
            try:
                from analysis.pairs import PairsTrader
                from data.fetcher import DataFetcher
                from config.settings import MarketConfig
                from analysis.technical import TechnicalAnalyzer

                fetcher  = DataFetcher()
                analyzer = TechnicalAnalyzer()
                pairs    = PairsTrader()

                with st.spinner("Computing cointegration..."):
                    symbols    = MarketConfig.NIFTY_50_SYMBOLS[:15]
                    price_data = {}
                    for sym in symbols:
                        df = fetcher.get_historical_data(sym, period="1y", interval="1d")
                        if df is not None and len(df) > 50:
                            price_data[sym] = df

                    valid_pairs = pairs.find_pairs(symbols, price_data)

                if valid_pairs:
                    st.dataframe(pd.DataFrame(valid_pairs), use_container_width=True, hide_index=True)
                else:
                    st.info("No cointegrated pairs found with current data.")
            except Exception as e:
                st.warning(f"Pairs analysis error: {e}")

    # ── Auto refresh ──────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(0.1)
        st.rerun()


if __name__ == "__main__":
    if not HAS_STREAMLIT:
        print("Install: pip install streamlit plotly")
    else:
        main()
