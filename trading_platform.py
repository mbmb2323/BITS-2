from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


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


def load_price_csv(path: str) -> List[PriceRow]:
    rows: List[PriceRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"date", "symbol", "close", "volume"}
        if not required.issubset({k.lower() for k in (reader.fieldnames or [])}):
            raise ValueError("CSV must contain date,symbol,close,volume columns")
        for row in reader:
            rows.append(
                PriceRow(
                    date=datetime.fromisoformat(row["date"]),
                    symbol=row["symbol"],
                    close=float(row["close"]),
                    volume=float(row["volume"]),
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

    target_symbol = symbol or (rankings[0]["symbol"] if rankings else None)
    if not target_symbol or target_symbol not in grouped:
        return {"screen": rankings, "error": "No symbol available for model/backtest"}

    series = grouped[target_symbol]
    X, y = build_features(series)
    if len(X) < 30:
        return {"screen": rankings, "symbol": target_symbol, "error": "Not enough data for model"}

    split = int(len(X) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]
    model = LogisticSignalModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    rows_for_backtest = series[-(len(probs) + 1) :]
    bt = backtest_long_only(rows_for_backtest, probs)

    return {
        "screen": rankings,
        "symbol": target_symbol,
        "backtest": {
            "total_return_pct": bt.total_return_pct,
            "annualized_return_pct": bt.annualized_return_pct,
            "annualized_volatility_pct": bt.annualized_volatility_pct,
            "sharpe_ratio": bt.sharpe_ratio,
            "max_drawdown_pct": bt.max_drawdown_pct,
            "hit_rate_pct": bt.hit_rate_pct,
            "trades": bt.trades,
        },
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ML-powered algorithm trading and stock screener pipeline")
    p.add_argument("csv", help="Path to OHLCV CSV with date,symbol,close,volume")
    p.add_argument("--symbol", help="Optional symbol to model/backtest")
    p.add_argument("--min-avg-volume", type=float, default=100_000)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=10)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_pipeline(
        csv_path=args.csv,
        symbol=args.symbol,
        min_avg_volume=args.min_avg_volume,
        min_price=args.min_price,
        top_n=args.top_n,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
