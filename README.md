# BITS-2 Trading Platform

A local, ML-powered algorithm trading toolkit with stock screening, backtesting, portfolio allocation, and repository code-assembly workflows.

## What this provides
- Stock screener optimized around investment KPIs
- Lightweight ML signal model and long-only backtester
- Allocation guidance for daily personal research
- Local repository scanner that extracts trading-related code/components from user repos
- CLI workflows for analysis, daily reporting, and repository assembly

## Quick start
```bash
python -m unittest -v
python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py analyze /absolute/path/to/prices.csv --top-n 10
python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py assemble --scan-root /absolute/path/to/repos --repo-name-contains Bauer
```

## Commands
- `analyze`: screener + ML model + backtest + allocations
- `assemble`: scan local repos and extract trading-related code/components
- `daily-report`: combine market analysis with optional repository assembly output

CSV schema required:
- `date` (ISO date or datetime)
- `symbol`
- `close`
- `volume`

See `/home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/USER_GUIDE.md` and `/home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/USER_MANUAL.md`.
