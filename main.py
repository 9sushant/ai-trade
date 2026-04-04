#!/usr/bin/env python3
"""
AI Trade - Smart Stock Trading System
NSE/BSE Technical Analysis + Angel One Auto Trading

Usage:
    python main.py                  # Interactive mode
    python main.py --scan           # Quick scan only (no broker needed)
    python main.py --auto           # Auto trading mode (requires Angel One credentials)
    python main.py --top10          # Show top 10 picks and exit
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import logger
from dashboard.terminal_ui import (
    console, print_banner, print_top_10_picks, print_intraday_picks,
    print_swing_picks, print_option_trades, print_risk_summary,
    print_trade_execution, print_menu,
)
from agents.trading_orchestrator import TradingOrchestrator
from agents.scanner_agent import ScannerAgent
from utils.helpers import format_currency


def run_scan_only():
    """Run a scan without broker connection - just show recommendations."""
    print_banner()
    console.print("[bold yellow]SCAN MODE - No broker connection required[/]\n")

    scanner = ScannerAgent()

    console.print("[bold]Scanning markets... (this may take a few minutes)[/]\n")
    top_10 = scanner.get_top_10_daily()
    scan = scanner.run_morning_scan()

    if top_10:
        print_top_10_picks(top_10)

    intraday = scan.get("intraday", [])
    if intraday:
        print_intraday_picks(intraday)

    swing = scan.get("swing", [])
    if swing:
        print_swing_picks(swing)

    options = scan.get("options", [])
    if options:
        print_option_trades(options)

    if not top_10 and not intraday and not swing:
        console.print("[yellow]No strong signals found at the moment. Market might be in consolidation.[/]")


def run_top10():
    """Quick top 10 picks."""
    print_banner()
    console.print("[bold]Fetching top 10 picks...[/]\n")
    scanner = ScannerAgent()
    top_10 = scanner.get_top_10_daily()
    if top_10:
        print_top_10_picks(top_10)
    else:
        console.print("[yellow]No strong picks found at the moment.[/]")


def run_interactive():
    """Full interactive mode with broker connection."""
    print_banner()

    orchestrator = TradingOrchestrator(auto_trade=False)

    # Try to connect to broker
    broker_connected = False
    try:
        broker_connected = orchestrator.initialize()
    except Exception as e:
        console.print(f"[yellow]Broker connection failed: {e}[/]")
        console.print("[dim]Running in scan-only mode. Set up .env for full features.[/]\n")

    # Run morning scan
    console.print("[bold]Running morning scan...[/]\n")
    recommendations = orchestrator.get_recommendations()

    top_10 = recommendations.get("top_10_picks", [])
    if top_10:
        print_top_10_picks(top_10)

    print_menu()

    while True:
        try:
            choice = console.input("[bold cyan]Enter command (0-9): [/]").strip()

            if choice == "0":
                orchestrator.shutdown()
                console.print("[bold]Goodbye! Happy Trading![/]")
                break

            elif choice == "1":
                top_10 = orchestrator.scanner.get_top_10_daily()
                if top_10:
                    print_top_10_picks(top_10)
                else:
                    console.print("[yellow]No picks available.[/]")

            elif choice == "2":
                from screener.stock_screener import StockScreener
                screener = StockScreener()
                intraday = screener.get_intraday_picks(10)
                if intraday:
                    print_intraday_picks(intraday)
                else:
                    console.print("[yellow]No intraday signals found.[/]")

            elif choice == "3":
                from screener.stock_screener import StockScreener
                screener = StockScreener()
                swing = screener.get_swing_trade_picks(10)
                if swing:
                    print_swing_picks(swing)
                else:
                    console.print("[yellow]No swing trade signals found.[/]")

            elif choice == "4":
                options = recommendations.get("option_trades", [])
                if not options:
                    from options.analyzer import OptionsAnalyzer
                    analyzer = OptionsAnalyzer()
                    options = analyzer.find_best_option_trades()
                print_option_trades(options)

            elif choice == "5":
                if broker_connected:
                    risk = orchestrator.risk_agent.get_risk_summary()
                    print_risk_summary(risk)
                else:
                    console.print("[yellow]Broker not connected. Set up .env for risk dashboard.[/]")

            elif choice == "6":
                if not broker_connected:
                    console.print("[yellow]Broker not connected. Cannot execute trades.[/]")
                    continue

                top_10 = recommendations.get("top_10_picks", [])
                if not top_10:
                    console.print("[yellow]No recommendations to execute.[/]")
                    continue

                console.print("[bold]Available trades:[/]")
                for i, pick in enumerate(top_10[:5], 1):
                    console.print(
                        f"  {i}. {pick['symbol']} - {pick.get('recommendation', '')} "
                        f"@ ₹{pick.get('entry', 0):,.2f}"
                    )

                idx = console.input("[bold]Enter trade # to execute (or 'all' for top 3): [/]").strip()
                if idx.lower() == "all":
                    results = orchestrator.executor.execute_batch(top_10[:3], auto_execute=False)
                    for r in results:
                        print_trade_execution(r)
                elif idx.isdigit() and 1 <= int(idx) <= len(top_10):
                    result = orchestrator.executor.execute_trade(
                        top_10[int(idx) - 1], auto_execute=False
                    )
                    if result:
                        print_trade_execution(result)

            elif choice == "7":
                if not broker_connected:
                    console.print("[red]Cannot auto-trade without broker connection![/]")
                    continue
                confirm = console.input("[bold red]Start AUTO TRADING? This will place real orders! (yes/no): [/]")
                if confirm.lower() == "yes":
                    orchestrator.auto_trade = True
                    orchestrator.run_trading_loop()

            elif choice == "8":
                console.print("[bold]Refreshing scan...[/]\n")
                orchestrator.scanner.screener.fetcher.clear_cache()
                recommendations = orchestrator.get_recommendations()
                top_10 = recommendations.get("top_10_picks", [])
                if top_10:
                    print_top_10_picks(top_10)

            elif choice == "9":
                if broker_connected:
                    positions = orchestrator.broker.get_positions()
                    holdings = orchestrator.broker.get_holdings()
                    pnl = orchestrator.broker.get_pnl()

                    console.print(f"\n[bold]P&L: {format_currency(pnl['total_pnl'])}[/]")
                    console.print(f"Open Positions: {len([p for p in positions if int(p.get('netqty', 0)) != 0])}")
                    console.print(f"Holdings: {len(holdings)}")

                    if positions:
                        from rich.table import Table
                        from rich import box
                        t = Table(title="Positions", box=box.SIMPLE)
                        t.add_column("Symbol")
                        t.add_column("Qty", justify="right")
                        t.add_column("Avg Price", justify="right")
                        t.add_column("LTP", justify="right")
                        t.add_column("P&L", justify="right")
                        for p in positions:
                            if int(p.get("netqty", 0)) != 0:
                                pnl_val = float(p.get("pnl", 0))
                                t.add_row(
                                    p.get("tradingsymbol", ""),
                                    str(p.get("netqty", 0)),
                                    p.get("buyavgprice", "0"),
                                    p.get("ltp", "0"),
                                    f"[{'green' if pnl_val >= 0 else 'red'}]{format_currency(pnl_val)}[/]",
                                )
                        console.print(t)
                else:
                    console.print("[yellow]Broker not connected.[/]")

            else:
                print_menu()

        except KeyboardInterrupt:
            console.print("\n[bold]Shutting down...[/]")
            orchestrator.shutdown()
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            logger.error(f"Interactive mode error: {e}")


def run_backtest(period: str = "1y", symbols_count: int = 20):
    """Run backtest on historical data."""
    from dashboard.terminal_ui import print_banner
    from backtest.engine import BacktestEngine
    from backtest.display import print_backtest_results

    print_banner()
    console.print("[bold magenta]BACKTEST MODE[/]")
    console.print(f"[dim]Period: {period} | Symbols: top {symbols_count} Nifty stocks[/]\n")

    from config.settings import MarketConfig
    symbols = MarketConfig.NIFTY_50_SYMBOLS[:symbols_count]

    console.print("[bold]Running backtest... (this may take a few minutes)[/]\n")
    engine = BacktestEngine()
    results = engine.run(symbols=symbols, period=period)
    print_backtest_results(results)


def run_auto_trading():
    """Auto trading mode."""
    print_banner()
    console.print("[bold red]AUTO TRADING MODE[/]")
    console.print("[yellow]This will place REAL orders on your Angel One account![/]\n")

    orchestrator = TradingOrchestrator(auto_trade=True)

    if not orchestrator.initialize():
        console.print("[red]Failed to initialize. Check your .env configuration.[/]")
        return

    # Morning routine
    summary = orchestrator.run_morning_routine()
    top_10 = orchestrator.scanner.get_top_10_daily()
    if top_10:
        print_top_10_picks(top_10)

    # Start auto trading loop
    console.print("\n[bold]Starting auto trading loop (Ctrl+C to stop)...[/]\n")
    try:
        orchestrator.run_trading_loop(interval_seconds=300)
    except KeyboardInterrupt:
        pass
    finally:
        eod = orchestrator.run_eod_routine()
        print_risk_summary(eod)
        orchestrator.shutdown()


def main():
    parser = argparse.ArgumentParser(description="AI Trade - Smart Stock Trading System")
    parser.add_argument("--scan", action="store_true", help="Scan mode - show recommendations only")
    parser.add_argument("--auto", action="store_true", help="Auto trading mode")
    parser.add_argument("--top10", action="store_true", help="Show top 10 picks and exit")
    parser.add_argument("--backtest", action="store_true", help="Backtest strategy on historical data")
    parser.add_argument("--period", type=str, default="1y", help="Backtest period (e.g., 6mo, 1y, 2y)")
    parser.add_argument("--symbols", type=int, default=20, help="Number of Nifty 50 symbols to backtest")
    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    if args.scan:
        run_scan_only()
    elif args.top10:
        run_top10()
    elif args.backtest:
        run_backtest(period=args.period, symbols_count=args.symbols)
    elif args.auto:
        run_auto_trading()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
