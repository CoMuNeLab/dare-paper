import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd

from ps_dare.auto_arima import run


class FakeModel:
    def __init__(self, seasonal_period: int):
        self.order = (1, 0, 0)
        self.seasonal_order = (0, 0, 0, seasonal_period)
        self.with_intercept = False


class AutoArimaThresholdTests(unittest.TestCase):
    def _dataset(
        self,
        root: Path,
        name: str,
        frequency: str,
        cases: list[int],
    ) -> Path:
        path = root / f"{name}.csv"
        date_frequency = "7D" if frequency == "weekly" else "ME"
        pd.DataFrame(
            {
                "valid_time": pd.date_range(
                    "2024-01-01" if frequency == "weekly" else "2024-01-31",
                    periods=len(cases),
                    freq=date_frequency,
                ),
                "n_accesses": cases,
            }
        ).to_csv(path, index=False)
        return path

    def test_skips_threshold_boundaries_and_only_saves_eligible_orders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weekly_skipped = self._dataset(
                root, "weekly_skipped", "weekly", [11, 10, 12]
            )
            weekly_eligible = self._dataset(
                root, "weekly_eligible", "weekly", [11, 12, 13]
            )
            monthly_skipped = self._dataset(
                root, "monthly_skipped", "monthly", [101, 100, 102]
            )
            monthly_eligible = self._dataset(
                root, "monthly_eligible", "monthly", [101, 102, 103]
            )
            configuration = root / "configuration.csv"
            pd.DataFrame(
                [
                    self._configuration_row(weekly_skipped, "weekly"),
                    self._configuration_row(weekly_eligible, "weekly"),
                    self._configuration_row(monthly_skipped, "monthly"),
                    self._configuration_row(monthly_eligible, "monthly"),
                ]
            ).to_csv(configuration, index=False)

            calls = []
            fake_pmdarima = ModuleType("pmdarima")

            def fake_auto_arima(y, **parameters):
                calls.append(y.copy())
                return FakeModel(parameters["m"])

            fake_pmdarima.auto_arima = fake_auto_arima
            fake_pmdarima.ARIMA = object
            with (
                patch.dict(os.environ, {"DATA_BASE_DIR": str(root)}),
                patch.dict(sys.modules, {"pmdarima": fake_pmdarima}),
            ):
                outputs = run(configuration)

            self.assertEqual(len(calls), 2)
            self.assertEqual(len(outputs), 4)

            weekly_orders_path = next(
                path
                for path in outputs
                if path.parent == root / "data" / "weekly" / "orders"
                and path.name.startswith("orders_")
            )
            monthly_orders_path = next(
                path
                for path in outputs
                if path.parent == root / "data" / "monthly" / "orders"
                and path.name.startswith("orders_")
            )
            weekly_orders = pd.read_csv(weekly_orders_path)
            monthly_orders = pd.read_csv(monthly_orders_path)
            self.assertEqual(
                weekly_orders["path_to_training_file"].tolist(),
                [str(weekly_eligible)],
            )
            self.assertEqual(
                monthly_orders["path_to_training_file"].tolist(),
                [str(monthly_eligible)],
            )

            weekly_report = pd.read_csv(
                root / "data" / "weekly" / "orders" / "skipped.csv"
            )
            monthly_report = pd.read_csv(
                root / "data" / "monthly" / "orders" / "skipped.csv"
            )
            self.assertEqual(
                weekly_report["path_to_training_file"].tolist(),
                [str(weekly_skipped)],
            )
            self.assertEqual(weekly_report["case_threshold"].tolist(), [10])
            self.assertEqual(
                weekly_report["timesteps_at_or_below_threshold"].tolist(), [1]
            )
            self.assertEqual(
                monthly_report["path_to_training_file"].tolist(),
                [str(monthly_skipped)],
            )
            self.assertEqual(monthly_report["case_threshold"].tolist(), [100])
            self.assertEqual(
                monthly_report["timesteps_at_or_below_threshold"].tolist(), [1]
            )

    def _configuration_row(self, dataset: Path, aggregation: str) -> dict[str, str]:
        return {
            "dataset": str(dataset),
            "aggregation": aggregation,
            "driver_kind": "none",
            "discretization": "continuous",
            "train_start": "2024-01-01",
            "train_end": "2024-12-31",
            "test_end": "2025-12-31",
            "lag": "0",
        }


if __name__ == "__main__":
    unittest.main()
