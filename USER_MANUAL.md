# User Manual

## Command reference
```bash
python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py <csv_path> [--symbol TICKER] [--min-avg-volume N] [--min-price N] [--top-n N]
```

### Inputs
- `<csv_path>`: absolute path to data file
- `--symbol`: force target symbol for model/backtest
- `--min-avg-volume`: liquidity filter (default: 100000)
- `--min-price`: minimum last close filter (default: 5)
- `--top-n`: number of ranked symbols (default: 10)

### Output format
JSON payload containing:
- `screen`: list of ranking records
- `symbol`: selected symbol for signal model
- `backtest`: strategy KPIs
- `error`: included when data is insufficient

## Operational checklist
- Keep historical data refreshed daily.
- Re-run before any investment decision.
- Compare KPI shifts over time.
- Add independent risk limits before live trading.
