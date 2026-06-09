from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

DEFAULT_CODE_KEYWORDS = (
    "trade",
    "trading",
    "signal",
    "portfolio",
    "risk",
    "screen",
    "screener",
    "backtest",
    "execution",
    "model",
    "alpha",
    "allocation",
)
SOURCE_FILE_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}


@dataclass(frozen=True)
class PriceRow:
    date: datetime
    symbol: str
    close: float
    volume: float


@dataclass(frozen=True)
class BacktestResult:
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    hit_rate_pct: float
    trades: int


@dataclass(frozen=True)
class ExtractedComponent:
    repository: str
    path: str
    line_number: int
    kind: str
    signature: str
    matched_keywords: tuple[str, ...]


def load_price_csv(path: str) -> List[PriceRow]:
    rows: List[PriceRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_map = {str(name).lower(): str(name) for name in (reader.fieldnames or [])}
        required = {"date", "symbol", "close", "volume"}
        if not required.issubset(field_map):
            raise ValueError("CSV must contain date,symbol,close,volume columns")
        for row in reader:
            rows.append(
                PriceRow(
                    date=datetime.fromisoformat(row[field_map["date"]]),
                    symbol=row[field_map["symbol"]],
                    close=float(row[field_map["close"]]),
                    volume=float(row[field_map["volume"]]),
                )
            )
    rows.sort(key=lambda r: (r.symbol, r.date))
    return rows


def group_by_symbol(rows: Sequence[PriceRow]) -> Dict[str, List[PriceRow]]:
    grouped: Dict[str, List[PriceRow]] = {}
    for row in rows:
        grouped.setdefault(row.symbol, []).append(row)
    return grouped


def _daily_returns(prices: Sequence[float]) -> List[float]:
    return [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices)) if prices[i - 1] > 0]


def screen_symbols(
    grouped: Dict[str, List[PriceRow]],
    min_avg_volume: float = 100_000,
    min_price: float = 5.0,
    top_n: int = 10,
) -> List[dict]:
    rankings: List[dict] = []
    for symbol, rows in grouped.items():
        if len(rows) < 30:
            continue
        closes = [r.close for r in rows]
        vols = [r.volume for r in rows]
        avg_volume = mean(vols[-20:])
        latest = closes[-1]
        if avg_volume < min_avg_volume or latest < min_price:
            continue
        rets = _daily_returns(closes[-90:])
        if len(rets) < 20:
            continue
        annual_ret = mean(rets) * 252
        annual_vol = (pstdev(rets) if len(rets) > 1 else 0.0) * math.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
        momentum = (closes[-1] / closes[-21]) - 1.0
        score = (0.45 * sharpe) + (0.35 * annual_ret) + (0.20 * momentum)
        rankings.append(
            {
                "symbol": symbol,
                "score": score,
                "annual_return": annual_ret,
                "annual_volatility": annual_vol,
                "sharpe": sharpe,
                "momentum_1m": momentum,
                "avg_volume_20d": avg_volume,
                "last_price": latest,
            }
        )
    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings[:top_n]


def build_features(rows: Sequence[PriceRow], lookback: int = 5) -> tuple[List[List[float]], List[int]]:
    if len(rows) <= lookback + 1:
        return [], []
    closes = [r.close for r in rows]
    volumes = [r.volume for r in rows]
    X: List[List[float]] = []
    y: List[int] = []
    for i in range(lookback, len(rows) - 1):
        daily = _daily_returns(closes[i - lookback : i + 1])
        avg_ret = mean(daily)
        vol = pstdev(daily) if len(daily) > 1 else 0.0
        momentum = (closes[i] / closes[i - lookback]) - 1.0
        vol_spike = volumes[i] / max(1.0, mean(volumes[i - lookback : i]))
        X.append([avg_ret, vol, momentum, vol_spike])
        y.append(1 if closes[i + 1] > closes[i] else 0)
    return X, y


class LogisticSignalModel:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 600):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._weights: List[float] = []
        self._bias = 0.0
        self._means: List[float] = []
        self._stds: List[float] = []

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(min(z, 30), -30)
        return 1.0 / (1.0 + math.exp(-z))

    def _standardize(self, X: Sequence[Sequence[float]], fit: bool) -> List[List[float]]:
        cols = len(X[0])
        if fit:
            self._means = [mean(row[c] for row in X) for c in range(cols)]
            self._stds = [pstdev(row[c] for row in X) or 1.0 for c in range(cols)]
        return [[(row[c] - self._means[c]) / self._stds[c] for c in range(cols)] for row in X]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        if not X or len(X) != len(y):
            raise ValueError("Training data is invalid")
        Xn = self._standardize(X, fit=True)
        cols = len(Xn[0])
        self._weights = [0.0] * cols
        self._bias = 0.0
        n = len(Xn)
        for _ in range(self.epochs):
            grad_w = [0.0] * cols
            grad_b = 0.0
            for i, row in enumerate(Xn):
                pred = self._sigmoid(sum(w * x for w, x in zip(self._weights, row)) + self._bias)
                err = pred - y[i]
                for c in range(cols):
                    grad_w[c] += err * row[c]
                grad_b += err
            for c in range(cols):
                self._weights[c] -= self.learning_rate * (grad_w[c] / n)
            self._bias -= self.learning_rate * (grad_b / n)

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        if not self._weights:
            raise ValueError("Model is not fitted")
        Xn = self._standardize(X, fit=False)
        return [self._sigmoid(sum(w * x for w, x in zip(self._weights, row)) + self._bias) for row in Xn]


def backtest_long_only(
    rows: Sequence[PriceRow],
    probabilities: Sequence[float],
    threshold: float = 0.55,
    fee_bps: float = 5.0,
) -> BacktestResult:
    if len(rows) < 3 or len(probabilities) != len(rows) - 1:
        raise ValueError("Rows/probabilities shape mismatch")

    equity = [1.0]
    trade_returns: List[float] = []
    wins = 0
    position = 0

    for i in range(1, len(rows)):
        p = probabilities[i - 1]
        next_pos = 1 if p >= threshold else 0
        prev_close, close = rows[i - 1].close, rows[i].close
        daily_ret = (close / prev_close) - 1.0 if prev_close > 0 else 0.0
        turn_cost = (fee_bps / 10_000.0) if next_pos != position else 0.0
        strat_ret = (daily_ret * next_pos) - turn_cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        if next_pos and daily_ret > 0:
            wins += 1
        if next_pos:
            trade_returns.append(daily_ret - turn_cost)
        position = next_pos

    total_return = equity[-1] - 1.0
    periods = len(equity) - 1
    ann_return = (equity[-1] ** (252 / periods) - 1.0) if periods > 0 else 0.0
    daily_equity_returns = _daily_returns(equity)
    ann_vol = (pstdev(daily_equity_returns) if len(daily_equity_returns) > 1 else 0.0) * math.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    trades = len(trade_returns)
    hit_rate = (wins / trades) if trades else 0.0
    return BacktestResult(
        total_return_pct=total_return * 100.0,
        annualized_return_pct=ann_return * 100.0,
        annualized_volatility_pct=ann_vol * 100.0,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd * 100.0,
        hit_rate_pct=hit_rate * 100.0,
        trades=trades,
    )


def recommend_allocations(rankings: Sequence[dict], max_names: int = 5) -> List[dict]:
    selected = list(rankings[:max_names])
    if not selected:
        return []
    raw_scores = []
    for item in selected:
        risk = max(item["annual_volatility"], 0.01)
        reward = max(item["annual_return"], 0.0) + max(item["momentum_1m"], 0.0) + 0.01
        raw_scores.append(reward / risk)
    total = sum(raw_scores) or float(len(raw_scores))
    allocations = []
    for item, raw in zip(selected, raw_scores):
        allocations.append(
            {
                "symbol": item["symbol"],
                "target_weight_pct": round((raw / total) * 100.0, 2),
                "annual_return": item["annual_return"],
                "annual_volatility": item["annual_volatility"],
                "momentum_1m": item["momentum_1m"],
            }
        )
    return allocations


def build_daily_investment_plan(rankings: Sequence[dict], allocations: Sequence[dict], target_symbol: str | None) -> dict:
    candidates = [item["symbol"] for item in rankings[:3]]
    focus_names = [item["symbol"] for item in allocations[:3]]
    plan = {
        "pre_market": [
            f"Review top screened symbols: {', '.join(candidates) or 'none'}.",
            "Validate price, liquidity, and overnight news before acting.",
            f"Confirm target allocations: {', '.join(focus_names) or 'none'}.",
        ],
        "market_hours": [
            f"Monitor ML signal and price action for {target_symbol or 'the selected symbol' }.",
            "Only take long exposure when the model probability clears the confidence threshold.",
            "Track realized drawdown versus plan before adding risk.",
        ],
        "post_market": [
            "Re-run the backtest on refreshed data.",
            "Compare KPI drift versus prior day output.",
            "Adjust thresholds only after reviewing risk-adjusted performance.",
        ],
    }
    return plan


def run_pipeline(
    csv_path: str,
    symbol: str | None = None,
    min_avg_volume: float = 100_000,
    min_price: float = 5.0,
    top_n: int = 10,
) -> dict:
    rows = load_price_csv(csv_path)
    grouped = group_by_symbol(rows)
    rankings = screen_symbols(grouped, min_avg_volume=min_avg_volume, min_price=min_price, top_n=top_n)
    allocations = recommend_allocations(rankings)

    target_symbol = symbol or (rankings[0]["symbol"] if rankings else None)
    result = {
        "screen": rankings,
        "allocations": allocations,
        "symbol": target_symbol,
        "daily_plan": build_daily_investment_plan(rankings, allocations, target_symbol),
    }
    if not target_symbol or target_symbol not in grouped:
        result["error"] = "No symbol available for model/backtest"
        return result

    series = grouped[target_symbol]
    X, y = build_features(series)
    if len(X) < 30:
        result["error"] = "Not enough data for model"
        return result

    split = max(1, int(len(X) * 0.7))
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]
    if not X_test:
        result["error"] = "Not enough holdout data for backtest"
        return result
    model = LogisticSignalModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    rows_for_backtest = series[-(len(probs) + 1) :]
    bt = backtest_long_only(rows_for_backtest, probs)

    result["backtest"] = {
        "total_return_pct": bt.total_return_pct,
        "annualized_return_pct": bt.annualized_return_pct,
        "annualized_volatility_pct": bt.annualized_volatility_pct,
        "sharpe_ratio": bt.sharpe_ratio,
        "max_drawdown_pct": bt.max_drawdown_pct,
        "hit_rate_pct": bt.hit_rate_pct,
        "trades": bt.trades,
    }
    return result


def find_git_repositories(scan_root: str, repo_name_contains: Sequence[str] | None = None) -> List[Path]:
    root = Path(scan_root)
    filters = [item.lower() for item in (repo_name_contains or []) if item]
    repos: List[Path] = []

    def matches(path: Path) -> bool:
        lowered = path.name.lower()
        return not filters or any(fragment in lowered for fragment in filters)

    if (root / ".git").exists() and matches(root):
        repos.append(root)

    for child in sorted(root.iterdir() if root.exists() else []):
        if child.is_dir() and (child / ".git").exists() and matches(child):
            repos.append(child)
    return repos


def _matched_keywords(text: str, keywords: Sequence[str]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(keyword for keyword in keywords if keyword in lowered)


_DEF_OR_CLASS_RE = re.compile(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def extract_repository_components(
    repo_path: str | Path,
    keywords: Sequence[str] = DEFAULT_CODE_KEYWORDS,
    max_components: int = 25,
) -> List[ExtractedComponent]:
    repo = Path(repo_path)
    extracted: List[ExtractedComponent] = []
    for file_path in sorted(repo.rglob("*")):
        if len(extracted) >= max_components:
            break
        if any(part in SKIP_DIR_NAMES for part in file_path.parts):
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in SOURCE_FILE_SUFFIXES:
            continue
        try:
            if file_path.stat().st_size > 256_000:
                continue
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        relative = str(file_path.relative_to(repo))
        for idx, line in enumerate(lines, start=1):
            if len(extracted) >= max_components:
                break
            stripped = line.strip()
            if not stripped:
                continue
            match = _DEF_OR_CLASS_RE.match(stripped)
            search_window = " ".join(lines[idx - 1 : min(idx + 3, len(lines))])
            matched = _matched_keywords(f"{relative} {stripped} {search_window}", keywords)
            if match and matched:
                extracted.append(
                    ExtractedComponent(
                        repository=repo.name,
                        path=relative,
                        line_number=idx,
                        kind=match.group(1),
                        signature=stripped,
                        matched_keywords=matched,
                    )
                )
                continue
            if file_path.suffix.lower() == ".md" and stripped[:1] in {"-", "*"} and matched:
                extracted.append(
                    ExtractedComponent(
                        repository=repo.name,
                        path=relative,
                        line_number=idx,
                        kind="note",
                        signature=stripped,
                        matched_keywords=matched,
                    )
                )
    return extracted


def assemble_codebase(
    scan_root: str,
    repo_name_contains: Sequence[str] | None = None,
    keywords: Sequence[str] = DEFAULT_CODE_KEYWORDS,
    max_components_per_repo: int = 25,
) -> dict:
    repositories = find_git_repositories(scan_root, repo_name_contains)
    assembled: List[ExtractedComponent] = []
    for repo in repositories:
        assembled.extend(extract_repository_components(repo, keywords=keywords, max_components=max_components_per_repo))

    categories = {
        "screening": 0,
        "modeling": 0,
        "risk": 0,
        "execution": 0,
        "backtesting": 0,
    }
    for component in assembled:
        matched = set(component.matched_keywords)
        if matched & {"screen", "screener"}:
            categories["screening"] += 1
        if matched & {"model", "signal", "alpha"}:
            categories["modeling"] += 1
        if matched & {"risk", "allocation", "portfolio"}:
            categories["risk"] += 1
        if matched & {"execution", "trade", "trading"}:
            categories["execution"] += 1
        if "backtest" in matched:
            categories["backtesting"] += 1

    components = [
        {
            "repository": item.repository,
            "path": item.path,
            "line_number": item.line_number,
            "kind": item.kind,
            "signature": item.signature,
            "matched_keywords": list(item.matched_keywords),
        }
        for item in assembled
    ]
    return {
        "scan_root": str(scan_root),
        "repositories_scanned": [repo.name for repo in repositories],
        "repositories_found": len(repositories),
        "components_found": len(components),
        "categories": categories,
        "components": components,
    }


def render_assembly_markdown(report: dict) -> str:
    lines = ["# Repository Assembly Report", ""]
    lines.append(f"- Scan Root: `{report['scan_root']}`")
    lines.append(f"- Repositories Found: {report['repositories_found']}")
    lines.append(f"- Components Found: {report['components_found']}")
    lines.append("")
    lines.append("## Repositories Scanned")
    for repo in report["repositories_scanned"] or ["No repositories found"]:
        lines.append(f"- {repo}")
    lines.append("")
    lines.append("## Category Coverage")
    for category, count in report["categories"].items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Extracted Components")
    for component in report["components"] or [{"repository": "n/a", "path": "n/a", "line_number": 0, "kind": "note", "signature": "No matching code or docs found", "matched_keywords": []}]:
        lines.append(
            f"- `{component['repository']}/{component['path']}:{component['line_number']}` [{component['kind']}] {component['signature']}"
        )
    return "\n".join(lines)


def _analyze_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("analyze", help="Run the screener, ML model, and backtest")
    p.add_argument("csv", help="Path to OHLCV CSV with date,symbol,close,volume")
    p.add_argument("--symbol", help="Optional symbol to model/backtest")
    p.add_argument("--min-avg-volume", type=float, default=100_000)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=10)


def _assemble_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("assemble", help="Scan local repositories and extract trading-related code components")
    p.add_argument("--scan-root", required=True, help="Root directory containing repositories to scan")
    p.add_argument("--repo-name-contains", action="append", default=[], help="Optional repository name filter")
    p.add_argument("--max-components-per-repo", type=int, default=25)
    p.add_argument("--output", help="Optional markdown output path")


def _daily_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("daily-report", help="Combine market analysis with optional repo assembly output")
    p.add_argument("csv", help="Path to OHLCV CSV with date,symbol,close,volume")
    p.add_argument("--symbol", help="Optional symbol to model/backtest")
    p.add_argument("--scan-root", help="Optional repository root for assembly scan")
    p.add_argument("--repo-name-contains", action="append", default=[], help="Optional repository name filter")
    p.add_argument("--output", help="Optional JSON output path")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML-powered trading platform and repository assembler")
    subparsers = parser.add_subparsers(dest="command")
    _analyze_parser(subparsers)
    _assemble_parser(subparsers)
    _daily_report_parser(subparsers)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else None
    if args_list and args_list[0] not in {"analyze", "assemble", "daily-report", "-h", "--help"}:
        args_list = ["analyze", *args_list]
    parser = _parser()
    args = parser.parse_args(args_list)

    if args.command in {None, "analyze"}:
        result = run_pipeline(
            csv_path=args.csv,
            symbol=getattr(args, "symbol", None),
            min_avg_volume=getattr(args, "min_avg_volume", 100_000),
            min_price=getattr(args, "min_price", 5.0),
            top_n=getattr(args, "top_n", 10),
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "assemble":
        result = assemble_codebase(
            scan_root=args.scan_root,
            repo_name_contains=args.repo_name_contains,
            max_components_per_repo=args.max_components_per_repo,
        )
        if args.output:
            Path(args.output).write_text(render_assembly_markdown(result), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "daily-report":
        result = run_pipeline(csv_path=args.csv, symbol=args.symbol)
        if args.scan_root:
            result["assembly"] = assemble_codebase(args.scan_root, args.repo_name_contains)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps(result, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
