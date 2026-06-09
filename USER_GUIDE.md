# User Guide

## Daily workflow
1. Export market data into one CSV (`date,symbol,close,volume`).
2. Run screener + model + backtest:
   ```bash
   python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py /absolute/path/to/prices.csv --top-n 10
   ```
3. Review JSON output:
   - `screen`: ranked candidates
   - `symbol`: modeled ticker
   - `backtest`: KPI summary

## KPI interpretation
- `total_return_pct`: total period return
- `annualized_return_pct`: yearly return equivalent
- `annualized_volatility_pct`: annual risk proxy
- `sharpe_ratio`: risk-adjusted return
- `max_drawdown_pct`: largest equity drop
- `hit_rate_pct`: winning-trade ratio
- `trades`: number of executed entries

## Safety note
This tool is for research support only, not financial advice. Validate assumptions with your own risk controls.
