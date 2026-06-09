# BITS-2 Trading Platform

A local, ML-powered algorithm trading and stock screener toolkit assembled into one workflow.

## What this provides
- Stock screening with KPI optimization (return/volatility/sharpe/momentum score)
- Lightweight ML signal model (logistic classifier)
- Backtesting engine with key investment KPIs
- CLI-first workflow for daily personal investment research

## Quick start
```bash
python -m unittest -v
python trading_platform.py /absolute/path/to/prices.csv --top-n 10
```

CSV schema required:
- `date` (ISO date or datetime)
- `symbol`
- `close`
- `volume`

See `/home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/USER_GUIDE.md` and `/home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/USER_MANUAL.md`.
