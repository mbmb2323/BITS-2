# User Guide

## Daily workflow
1. Export market data into one CSV (`date,symbol,close,volume`).
2. Run market analysis:
   ```bash
   python trading_platform.py analyze /absolute/path/to/prices.csv --top-n 10
   ```
3. Optionally scan your local repo folder to assemble trading-related code already spread across repos:
   ```bash
   python trading_platform.py assemble --scan-root /absolute/path/to/repos --repo-name-contains Bauer --output /absolute/path/to/assembly_report.md
   ```
4. For one combined run, use:
   ```bash
   python trading_platform.py daily-report /absolute/path/to/prices.csv --scan-root /absolute/path/to/repos --repo-name-contains Bauer
   ```

## Output highlights
- `screen`: ranked stock candidates
- `ensemble`: latest 7-engine probability snapshot for the selected symbol
- `allocations`: suggested portfolio weights
- `backtest`: KPI summary
- `execution_plan`: actionable entry, sizing, and risk guardrails
- `daily_plan`: pre-market / market-hours / post-market checklist
- `assembly`: repository-scan results when requested

## Safety note
This tool is for research support only, not financial advice. Validate assumptions with your own risk controls before investing.
