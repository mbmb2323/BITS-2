import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from trading_platform import (
    LogisticSignalModel,
    PriceRow,
    assemble_codebase,
    backtest_long_only,
    build_features,
    group_by_symbol,
    recommend_allocations,
    render_assembly_markdown,
    screen_symbols,
)


class TradingPlatformTests(unittest.TestCase):
    def _rows(self, symbol: str, start_price: float, growth: float, n: int = 80):
        base = datetime(2024, 1, 1)
        rows = []
        price = start_price
        for i in range(n):
            price *= 1 + growth
            rows.append(PriceRow(date=base + timedelta(days=i), symbol=symbol, close=price, volume=200000 + i * 100))
        return rows

    def test_screening_ranks_stronger_symbol_higher(self):
        rows = self._rows("AAA", 50, 0.004) + self._rows("BBB", 50, 0.001)
        ranked = screen_symbols(group_by_symbol(rows), top_n=2)
        self.assertEqual(ranked[0]["symbol"], "AAA")
        self.assertEqual(len(ranked), 2)

    def test_model_produces_probabilities(self):
        rows = self._rows("AAA", 25, 0.003, n=70)
        X, y = build_features(rows)
        m = LogisticSignalModel(epochs=200)
        m.fit(X[:40], y[:40])
        probs = m.predict_proba(X[40:50])
        self.assertEqual(len(probs), 10)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_backtest_counts_trades_and_returns(self):
        rows = self._rows("AAA", 100, 0.002, n=40)
        probs = [0.8] * 39
        res = backtest_long_only(rows, probs, threshold=0.55, fee_bps=0)
        self.assertGreater(res.total_return_pct, 0)
        self.assertEqual(res.trades, 39)

    def test_allocations_are_normalized(self):
        rows = self._rows("AAA", 50, 0.004) + self._rows("BBB", 50, 0.002) + self._rows("CCC", 50, 0.001)
        ranked = screen_symbols(group_by_symbol(rows), top_n=3)
        allocations = recommend_allocations(ranked, max_names=3)
        self.assertEqual(len(allocations), 3)
        self.assertAlmostEqual(sum(item["target_weight_pct"] for item in allocations), 100.0, places=1)

    def test_code_assembly_extracts_trading_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "Bauer-Alpha"
            repo.mkdir()
            (repo / ".git").mkdir()
            (repo / "README.md").write_text("- trading dashboard\n- risk controls\n", encoding="utf-8")
            (repo / "engine.py").write_text(
                "def build_trading_signal(data):\n    return data\n\nclass PortfolioRiskModel:\n    pass\n",
                encoding="utf-8",
            )
            report = assemble_codebase(str(root), repo_name_contains=["Bauer"])
            self.assertEqual(report["repositories_found"], 1)
            self.assertGreaterEqual(report["components_found"], 2)
            self.assertIn("Bauer-Alpha", report["repositories_scanned"])
            markdown = render_assembly_markdown(report)
            self.assertIn("Repository Assembly Report", markdown)
            json.dumps(report)


if __name__ == "__main__":
    unittest.main()
