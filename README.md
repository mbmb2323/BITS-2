# BITS-2 Trading Platform

A local, ML-powered algorithm trading toolkit with stock screening, a 7-engine ensemble, backtesting, portfolio allocation, and repository code-assembly workflows.

## What this provides
- Stock screener optimized around investment KPIs
- Seven-engine ensemble that blends ML probability with technical analysis
- Allocation guidance for daily personal research
- Local repository scanner that extracts trading-related code/components from user repos
- CLI workflows for analysis, daily reporting, and repository assembly

## Quick start
```bash
python -m unittest -v
python trading_platform.py analyze /absolute/path/to/prices.csv --top-n 10
python trading_platform.py assemble --scan-root /absolute/path/to/repos --repo-name-contains Bauer
```

## Commands
- `analyze`: screener + 7-engine ensemble + backtest + allocations + execution plan
- `assemble`: scan local repos and extract trading-related code/components
- `daily-report`: combine market analysis with optional repository assembly output

CSV schema required:
- `date` (ISO date or datetime)
- `symbol`
- `close`
- `volume`

See `USER_GUIDE.md` and `USER_MANUAL.md`.
