# User Guide

## Daily workflow
1. Export market data into one CSV (`date,symbol,close,volume`).
2. Run market analysis:
   ```bash
   python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py analyze /absolute/path/to/prices.csv --top-n 10
   ```
3. Optionally scan your local repo folder to assemble trading-related code already spread across repos:
   ```bash
   python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py assemble --scan-root /absolute/path/to/repos --repo-name-contains Bauer --output /absolute/path/to/assembly_report.md
   ```
4. For one combined run, use:
   ```bash
   python /home/runner/work/BITS-2/BITS-2/mbmb2323/BITS-2/trading_platform.py daily-report /absolute/path/to/prices.csv --scan-root /absolute/path/to/repos --repo-name-contains Bauer
   ```

## Output highlights
- `screen`: ranked stock candidates
- `allocations`: suggested portfolio weights
- `backtest`: KPI summary
- `daily_plan`: pre-market / market-hours / post-market checklist
- `assembly`: repository-scan results when requested

## Safety note
This tool is for research support only, not financial advice. Validate assumptions with your own risk controls before investing.
