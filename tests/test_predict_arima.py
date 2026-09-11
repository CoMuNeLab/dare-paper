import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ps_dare.predict_arima import (
    forecast_order,
    orders_timestamp,
    read_orders,
    save_predictions,
)


class RecordingArima:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.fit_calls = []
        self.predict_calls = []
        self.update_calls = []
        self.history = []
        self.__class__.instances.append(self)

    def fit(self, y, X=None):
        self.fit_calls.append((y.copy(), None if X is None else X.copy()))
        self.history = list(y)
        return self

    def predict(self, n_periods, X=None, return_conf_int=False, alpha=0.05):
        self.predict_calls.append(
            (n_periods, None if X is None else X.copy(), return_conf_int, alpha)
        )
        driver_effect = 0 if X is None else float(X.iloc[0, 0])
        prediction = self.history[-1] + driver_effect
        if return_conf_int:
            return [prediction], [[prediction - 2.0, prediction + 2.0]]
        return [prediction]

    def update(self, y, X=None):
        self.update_calls.append((list(y), None if X is None else X.copy()))
        self.history.extend(y)
        return self


def order_row(dataset: Path, driver: str = "none", lag: str = "0") -> pd.Series:
    return pd.Series(
        {
            "aggregation": "weekly",
            "discretization": "continuous",
            "drivers_used": driver,
            "lag": lag,
            "path_to_training_file": str(dataset),
            "date_start_training": "2024-01-01",
            "date_end_training": "2024-01-15",
            "date_end_testing": "2024-01-29",
            "p": 1,
            "d": 0,
            "q": 1,
            "P": 0,
            "D": 0,
            "Q": 0,
            "seasonal_period": 52,
            "intercept": False,
        }
    )


class PredictArimaTests(unittest.TestCase):
    def setUp(self):
        RecordingArima.instances.clear()

    def _dataset(self, root: Path, with_driver: bool = False) -> Path:
        path = root / "general.csv"
        frame = pd.DataFrame(
            {
                "valid_time": pd.date_range("2024-01-01", periods=6, freq="7D"),
                "n_accesses": [10, 11, 12, 20, 30, 40],
            }
        )
        if with_driver:
            frame["NO2_mean"] = [1, 2, 3, 4, 5, 6]
        frame.to_csv(path, index=False)
        return path

    def test_fits_once_then_predicts_and_updates_one_observation_at_a_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._dataset(Path(temporary))
            result = forecast_order(order_row(path), model_class=RecordingArima)

        model = RecordingArima.instances[0]
        self.assertEqual(len(model.fit_calls), 1)
        self.assertEqual(len(model.predict_calls), 2)
        self.assertEqual(len(model.update_calls), 2)
        self.assertEqual(model.update_calls[0][0], [20.0])
        self.assertEqual(model.update_calls[1][0], [30.0])
        self.assertEqual(
            result.loc[result["phase"].eq("testing"), "prediction"].tolist(),
            [12.0, 20.0],
        )
        self.assertEqual(
            result.loc[
                result["phase"].eq("testing"), "prediction_lower_bound"
            ].tolist(),
            [10.0, 18.0],
        )
        self.assertEqual(
            result.loc[
                result["phase"].eq("testing"), "prediction_upper_bound"
            ].tolist(),
            [14.0, 22.0],
        )
        self.assertTrue(all(call[2] for call in model.predict_calls))
        self.assertTrue(all(call[3] == 0.05 for call in model.predict_calls))
        outside_testing = result["phase"].ne("testing")
        self.assertTrue(
            result.loc[outside_testing, "prediction_lower_bound"].isna().all()
        )
        self.assertTrue(
            result.loc[outside_testing, "prediction_upper_bound"].isna().all()
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(result["n_accesses"].tolist(), [10, 11, 12, 20, 30, 40])
        self.assertEqual(
            result["date_start_training"].unique().tolist(), ["2024-01-01"]
        )
        self.assertEqual(result["date_end_training"].unique().tolist(), ["2024-01-15"])
        self.assertEqual(result["date_end_testing"].unique().tolist(), ["2024-01-29"])
        self.assertEqual(result["prediction_horizon"].unique().tolist(), [1])

    def test_passes_lagged_driver_to_fit_predict_and_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._dataset(Path(temporary), with_driver=True)
            row = order_row(path, driver="NO2_mean", lag="1")
            row["date_start_training"] = "2024-01-08"
            result = forecast_order(
                row,
                model_class=RecordingArima,
            )

        model = RecordingArima.instances[0]
        fit_X = model.fit_calls[0][1]
        self.assertEqual(fit_X.columns.tolist(), ["NO2_mean_lag1"])
        self.assertEqual(fit_X.iloc[:, 0].tolist(), [1.0, 2.0])
        first_prediction_X = model.predict_calls[0][1]
        first_update_X = model.update_calls[0][1]
        self.assertEqual(first_prediction_X.iloc[0, 0], 3.0)
        pd.testing.assert_frame_equal(first_prediction_X, first_update_X)
        self.assertEqual(result["drivers_used"].unique().tolist(), ["NO2_mean"])
        self.assertEqual(result["driver_value"].tolist(), [1, 2, 3, 4, 5, 6])

    def test_missing_testing_driver_ends_testing_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._dataset(Path(temporary), with_driver=True)
            frame = pd.read_csv(path)
            frame.loc[3:, "NO2_mean"] = None
            frame.to_csv(path, index=False)
            row = order_row(path, driver="NO2_mean")
            with self.assertLogs("ps_dare.predict_arima", level="WARNING"):
                result = forecast_order(row, model_class=RecordingArima)

        model = RecordingArima.instances[0]
        self.assertEqual(len(model.predict_calls), 0)
        self.assertEqual(len(model.update_calls), 0)
        self.assertEqual(
            result["phase"].tolist(),
            [
                "training",
                "training",
                "training",
                "after_testing",
                "after_testing",
                "after_testing",
            ],
        )
        self.assertTrue(result["prediction"].isna().all())

    def test_supports_monthly_order_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monthly.csv"
            pd.DataFrame(
                {
                    "valid_time": pd.date_range("2024-01-31", periods=6, freq="ME"),
                    "n_accesses": [10, 11, 12, 20, 30, 40],
                }
            ).to_csv(path, index=False)
            row = order_row(path)
            row["aggregation"] = "monthly"
            row["date_start_training"] = "2024-01-01"
            row["date_end_training"] = "2024-03-31"
            row["date_end_testing"] = "2024-05-31"
            row["seasonal_period"] = 12

            result = forecast_order(row, model_class=RecordingArima)

        testing = result.loc[result["phase"].eq("testing")]
        self.assertEqual(testing["date"].dt.month.tolist(), [4, 5])
        self.assertEqual(testing["prediction"].tolist(), [12.0, 20.0])
        self.assertEqual(result["seasonal_period"].unique().tolist(), [12])

    def test_rejects_incomplete_order_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "orders.csv"
            pd.DataFrame({"aggregation": ["weekly"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing order columns"):
                read_orders(path)

    def test_extracts_timestamp_from_orders_filename(self):
        self.assertEqual(
            orders_timestamp("data/weekly/orders/orders_20260909_153012.csv"),
            "20260909_153012",
        )
        with self.assertRaisesRegex(ValueError, "orders_YYYYMMDD_HHMMSS.csv"):
            orders_timestamp("data/weekly/orders/orders.csv")

    def test_saves_one_csv_per_training_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "aggregation": ["weekly", "weekly", "monthly"],
                    "path_to_training_file": [
                        "/input/general.csv",
                        "/input/general.csv",
                        "/input/pediatrico.pkl",
                    ],
                    "date": pd.to_datetime(
                        ["2024-01-01", "2024-01-08", "2024-01-31"]
                    ),
                    "prediction": [12.0, 13.0, 20.0],
                    "prediction_lower_bound": [10.0, 11.0, 18.0],
                    "prediction_upper_bound": [14.0, 15.0, 22.0],
                }
            )
            old_data_base = __import__("os").environ.get("DATA_BASE_DIR")
            __import__("os").environ["DATA_BASE_DIR"] = str(root)
            try:
                outputs = save_predictions(predictions, "20260909_153012")
            finally:
                if old_data_base is None:
                    __import__("os").environ.pop("DATA_BASE_DIR", None)
                else:
                    __import__("os").environ["DATA_BASE_DIR"] = old_data_base

            self.assertEqual(
                outputs,
                [
                    root
                    / "data"
                    / "weekly"
                    / "predictions"
                    / "general_20260909_153012.csv",
                    root
                    / "data"
                    / "monthly"
                    / "predictions"
                    / "pediatrico_20260909_153012.csv",
                ],
            )
            weekly = pd.read_csv(outputs[0])
            monthly = pd.read_csv(outputs[1])
            self.assertEqual(weekly["date"].tolist(), ["2024-01-01", "2024-01-08"])
            self.assertEqual(monthly["date"].tolist(), ["2024-01-31"])
            self.assertEqual(
                weekly["prediction_lower_bound"].tolist(), [10.0, 11.0]
            )
            self.assertEqual(
                weekly["prediction_upper_bound"].tolist(), [14.0, 15.0]
            )


if __name__ == "__main__":
    unittest.main()
