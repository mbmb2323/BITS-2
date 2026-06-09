# User Manual

## Purpose
The trading platform is a local CLI tool that helps you:
- Screen symbols from market CSV data
- Build a seven-engine probability snapshot for one symbol
- Produce allocation and execution guidance for research
- Scan local repositories for trading-related reusable components

## Input requirements

### Market data CSV
All analysis commands require a CSV with these columns:
- `date` (ISO date/datetime)
- `symbol`
- `close`
- `volume`

Example row:
`2026-05-01,AAPL,195.23,48900000`

## Command reference

### 1) Analyze market data
```bash
python trading_platform.py analyze <csv_path> [--symbol TICKER] [--min-avg-volume N] [--min-price N] [--top-n N]
```

Options:
- `--symbol`: force one symbol for model/backtest
- `--min-avg-volume`: minimum 20-day average volume filter (default: `100000`)
- `--min-price`: minimum latest close filter (default: `5.0`)
- `--top-n`: number of ranked symbols in `screen` (default: `10`)

Primary output fields:
- `screen`
- `symbol`
- `ensemble`
- `backtest`
- `allocations`
- `execution_plan`
- `daily_plan`
- `error` (only when data is insufficient)

### 2) Assemble repository components
```bash
python trading_platform.py assemble --scan-root <repo_root> [--repo-name-contains NAME] [--max-components-per-repo N] [--output FILE]
```

Options:
- `--scan-root` (required): root directory containing local repositories
- `--repo-name-contains`: repository-name filter (repeatable)
- `--max-components-per-repo`: cap extracted matches per repository (default: `25`)
- `--output`: optional markdown report path

Primary output fields:
- `repositories_scanned`
- `components`
- `categories`

### 3) Daily combined report
```bash
python trading_platform.py daily-report <csv_path> [--symbol TICKER] [--scan-root <repo_root>] [--repo-name-contains NAME] [--output FILE]
```

Options:
- `--symbol`: optional symbol override for analysis
- `--scan-root`: optional repository scan root
- `--repo-name-contains`: optional repo-name filter (repeatable)
- `--output`: optional JSON output file path

Output:
- Includes all `analyze` fields
- Adds `assembly` if repository scan options are provided

## Exit behavior and error handling
- Invalid or missing CSV columns returns an error.
- Missing/low history for symbols may reduce or skip model/backtest data.
- When no tradable symbols pass filters, `error` is returned in the payload.

## Operational checklist
- Refresh price/volume data before each run.
- Re-run analysis before decisions.
- Validate allocations and execution plan against your own risk framework.
- Use assembly output to consolidate duplicate logic across repos.

## Safety
This software supports research workflows only and does not provide financial advice.
