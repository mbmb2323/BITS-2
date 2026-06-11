# User Guide

This guide focuses on practical, day-to-day usage.

## 1) Prepare data
Create or export a CSV file with:
- `date`
- `symbol`
- `close`
- `volume`

Store it at an absolute path, for example:
`/absolute/path/to/prices.csv`

## 2) Run a basic analysis
From repository root:

```bash
python trading_platform.py analyze /absolute/path/to/prices.csv --top-n 10
```

What to check first in output:
- `screen[0]` for top-ranked symbol
- `ensemble.ensemble_probability` for current model confidence
- `execution_plan` for risk-aware entry and sizing guidance

## 3) Analyze a specific symbol
If you already know the ticker you want to inspect:

```bash
python trading_platform.py analyze /absolute/path/to/prices.csv --symbol AAPL
```

Use this mode when you need focused backtest and plan output for one symbol.

## 4) Scan local repositories for reusable trading components
```bash
python trading_platform.py assemble --scan-root /absolute/path/to/repos --repo-name-contains Bauer --output /absolute/path/to/assembly_report.md
```

Use this output to identify:
- Existing signal/risk/backtest logic
- Overlapping implementations across repositories
- Candidates for consolidation

## 5) Run one combined daily command
```bash
python trading_platform.py daily-report /absolute/path/to/prices.csv --scan-root /absolute/path/to/repos --repo-name-contains Bauer
```

This returns market analysis plus optional assembly details in one JSON payload.

## 6) Suggested daily routine
1. Refresh market CSV after data close.
2. Run `analyze` (or `daily-report`) and inspect top signal changes.
3. Review execution and allocation outputs against your own risk policy.
4. Save output artifacts for traceability.
5. Repeat on the next session with updated data.

## Troubleshooting
- If you get a CSV column error, verify headers exactly match: `date,symbol,close,volume`.
- If output has limited symbols, your filters may be too strict (`--min-price`, `--min-avg-volume`).
- If a symbol lacks enough history, backtest and model sections may be reduced.

## Safety note
This tool is for research support only and is not financial advice.
