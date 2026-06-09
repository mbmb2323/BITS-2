# User Manual

## Command reference
```bash
python trading_platform.py analyze <csv_path> [--symbol TICKER] [--min-avg-volume N] [--min-price N] [--top-n N]
python trading_platform.py assemble --scan-root <repo_root> [--repo-name-contains NAME] [--max-components-per-repo N] [--output FILE]
python trading_platform.py daily-report <csv_path> [--symbol TICKER] [--scan-root <repo_root>] [--repo-name-contains NAME] [--output FILE]
```

## Analyze output
JSON payload containing:
- `screen`: ranked symbols
- `ensemble`: selected-symbol 7-engine snapshot blending ML and technical analysis
- `allocations`: normalized target weights
- `symbol`: selected symbol for the model/backtest
- `backtest`: KPI summary when enough history exists
- `execution_plan`: entry, sizing, and risk rules tied to ensemble probability
- `daily_plan`: repeatable operating checklist
- `error`: included when data is insufficient

## Assemble output
- `repositories_scanned`: matched local Git repositories
- `components`: extracted trading-related functions, classes, and README bullets
- `categories`: counts for screening, modeling, risk, execution, and backtesting coverage
- Optional Markdown report when `--output` is supplied

## Operational checklist
- Refresh market data daily.
- Re-run analysis before any investment decision.
- Review extracted repo components periodically to consolidate reusable ideas.
- Add independent broker, compliance, and risk controls before live trading.
