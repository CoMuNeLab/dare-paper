import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ps_dare.predict_arima import model_metrics
from ps_dare.predict_baseline import forecast_baseline, run, save_baseline


def baseline_row(dataset: Path) -> pd.Series:
    return pd.Series(
        {
            "dataset": str(dataset),
            "aggregation": "weekly",
            "driver_kind": "none",
            "discretization": "continuous",
            "train_start": "2024-01-01",
            "train_end": "2024-01-15",
            "test_end": "2024-01-29",
            "lag": "0",
        }
    )


class PredictBaselineTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        path = root / "general.csv"
        pd.DataFrame(
            {
                "valid_time": pd.date_range("2024-01-01", periods=6, freq="7D"),
                "n_accesses": [10, 11, 12, 20, 30, 40],
            }
        ).to_csv(path, index=False)
        return path

    def test_predicts_each_testing_period_from_previous_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = forecast_baseline(
                baseline_row(self._dataset(Path(temporary)))
            )

        testing = result.loc[result["phase"].eq("testing")]
        self.assertEqual(result["model_id"].unique().tolist(), ["baseline"])
        self.assertEqual(result["method"].unique().tolist(), ["baseline"])
        self.assertEqual(testing["prediction"].tolist(), [12.0, 20.0])
        self.assertTrue(result["prediction_lower_bound"].isna().all())
        self.assertTrue(result["prediction_upper_bound"].isna().all())
        self.assertTrue(result["training_aic"].isna().all())

        metrics = model_metrics(result).iloc[0]
        self.assertTrue(pd.isna(metrics["aic"]))
        self.assertAlmostEqual(metrics["mape"], (0.4 + 1 / 3) / 2 * 100)
        self.assertAlmostEqual(metrics["directional_accuracy"], 100.0)

    def test_run_deduplicates_driver_variants_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            configuration = root / "configuration.csv"
            rows = pd.DataFrame([baseline_row(dataset), baseline_row(dataset)])
            rows.loc[1, "driver_kind"] = "forced"
            rows.to_csv(configuration, index=False)

            old_data_base = __import__("os").environ.get("DATA_BASE_DIR")
            __import__("os").environ["DATA_BASE_DIR"] = str(root)
            try:
                outputs = run([configuration], timestamp="20260916_120000")
            finally:
                if old_data_base is None:
                    __import__("os").environ.pop("DATA_BASE_DIR", None)
                else:
                    __import__("os").environ["DATA_BASE_DIR"] = old_data_base

            self.assertEqual(len(outputs), 2)
            self.assertTrue(outputs[0].is_file())
            self.assertTrue(outputs[1].is_file())
            saved_metrics = pd.read_csv(outputs[1])
            self.assertEqual(saved_metrics["model_id"].tolist(), ["baseline"])
            self.assertTrue(saved_metrics["aic"].isna().all())
            self.assertEqual(saved_metrics["directional_accuracy"].tolist(), [100.0])

    def test_pipeline_style_row_preserves_model_number_in_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = baseline_row(self._dataset(root))
            row["model_number"] = 7
            old_data_base = __import__("os").environ.get("DATA_BASE_DIR")
            __import__("os").environ["DATA_BASE_DIR"] = str(root)
            try:
                outputs = save_baseline(row, "20260916_120000")
            finally:
                if old_data_base is None:
                    __import__("os").environ.pop("DATA_BASE_DIR", None)
                else:
                    __import__("os").environ["DATA_BASE_DIR"] = old_data_base

            self.assertIn("model_007", outputs[0].name)
            predictions = pd.read_csv(outputs[0])
            self.assertEqual(predictions["model_number"].unique().tolist(), [7])


if __name__ == "__main__":
    unittest.main()
