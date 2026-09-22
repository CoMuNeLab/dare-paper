import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ps_dare.predict_arima import (
    _forecast_order_isolated,
    _prediction_output_path,
    _update_model,
    forecast_order,
    main,
    model_metrics,
    orders_timestamp,
    read_orders,
    save_predictions,
)


class RecordingArima:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.order = parameters.get("order", (1, 0, 1))
        self.seasonal_order = parameters.get("seasonal_order", (0, 0, 0, 52))
        self.fit_calls = []
        self.predict_calls = []
        self.update_calls = []
        self.update_kwargs = []
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

    def update(self, y, X=None, **kwargs):
        self.update_calls.append((list(y), None if X is None else X.copy()))
        self.update_kwargs.append(kwargs)
        self.history.extend(y)
        return self

    def aic(self):
        return 123.5

    def params(self):
        # Depending on history makes the test verify extraction happens before updates.
        return [float(len(self.history)), 0.25]

    def pvalues(self):
        return [0.01, 0.02]

    @property
    def arima_res_(self):
        class Result:
            param_names = ["ar.L1", "sigma2"]

        return Result()


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
        self.assertEqual(model.update_kwargs, [{}, {}])
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
        self.assertEqual(result["training_aic"].unique().tolist(), [123.5])
        self.assertEqual(
            json.loads(result["fitted_parameters"].iloc[0]),
            {
                "parameters": {"ar.L1": 3.0, "sigma2": 0.25},
                "pvalues": {"ar.L1": 0.01, "sigma2": 0.02},
            },
        )

    def test_limits_mle_iterations_for_each_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._dataset(Path(temporary))
            forecast_order(
                order_row(path),
                model_class=RecordingArima,
                update_maxiter=1,
            )

        model = RecordingArima.instances[0]
        self.assertEqual(model.update_kwargs, [{"maxiter": 1}, {"maxiter": 1}])

    def test_state_only_update_appends_without_refitting(self):
        class AppendableResults:
            def __init__(self):
                self.nobs = 3
                self.calls = []
                self.model = SimpleNamespace(
                    data=SimpleNamespace(
                        # A None name reproduces the pmdarima model that exposed
                        # the Statsmodels column-concatenation regression.
                        orig_endog=pd.Series([10.0, 11.0, 12.0], name=None),
                        orig_exog=pd.DataFrame(
                            {"NO2_mean_lag1": [0.0, 1.0, 2.0]}
                        ),
                    )
                )

            def append(self, endog, exog=None, refit=True):
                self.calls.append((endog.copy(), exog.copy(), refit))
                self.nobs += len(endog)
                return self

        results = AppendableResults()
        model = SimpleNamespace(
            arima_res_=results,
            nobs_=3,
            update=lambda *args, **kwargs: self.fail("update() must not be called"),
        )
        date = pd.Timestamp("2024-01-22")
        X_step = pd.DataFrame({"NO2_mean_lag1": [3.0]}, index=[date])

        _update_model(
            model,
            20.0,
            date,
            X_step,
            state_only_updates=True,
            update_maxiter=None,
        )

        observation, appended_X, refit = results.calls[0]
        self.assertEqual(observation.tolist(), [20.0])
        self.assertEqual(observation.index.tolist(), [date])
        self.assertIsNone(observation.name)
        pd.testing.assert_frame_equal(appended_X, X_step)
        self.assertFalse(refit)
        self.assertEqual(model.nobs_, 4)

    def test_rejects_combining_state_only_and_limited_updates(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            forecast_order(
                pd.Series(dtype="object"),
                state_only_updates=True,
                update_maxiter=1,
            )

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

    def test_serializes_initial_fit_before_rolling_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._dataset(root)
            model_path = root / "models" / "general" / "model.pkl"
            forecast_order(
                order_row(path),
                model_class=RecordingArima,
                model_output_path=model_path,
            )

            with model_path.open("rb") as stream:
                saved_model = pickle.load(stream)

        self.assertEqual(saved_model.history, [10, 11, 12])
        self.assertEqual(saved_model.update_calls, [])

    def test_loads_auto_arima_fit_without_refitting_for_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._dataset(root)
            row = order_row(path)
            model_path = root / "models" / "auto_arima.pkl"
            fitted_model = RecordingArima(
                order=(1, 0, 1),
                seasonal_order=(0, 0, 0, 52),
                with_intercept=False,
            ).fit(pd.Series([10, 11, 12]))
            model_path.parent.mkdir(parents=True)
            with model_path.open("wb") as stream:
                pickle.dump(fitted_model, stream)
            row["model_file"] = str(model_path)

            with patch.object(
                RecordingArima,
                "fit",
                side_effect=AssertionError("saved model must not be refitted"),
            ) as refit:
                result = forecast_order(row)
            refit.assert_not_called()

            with model_path.open("rb") as stream:
                unchanged_model = pickle.load(stream)

        self.assertEqual(len(RecordingArima.instances), 1)
        self.assertEqual(len(fitted_model.fit_calls), 1)
        self.assertEqual(unchanged_model.update_calls, [])
        self.assertEqual(
            result.loc[result["phase"].eq("testing"), "prediction"].tolist(),
            [12.0, 20.0],
        )

    def test_worker_saves_each_completed_simulation_immediately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._dataset(root)
            model_path = root / "models" / "model.pkl"
            prediction_path = root / "predictions" / "model.csv"
            completed = pd.DataFrame(
                {
                    "model_id": ["model_001_general_none"],
                    "phase": ["testing"],
                    "n_accesses": [10.0],
                    "prediction": [12.0],
                    "path_to_training_file": [str(path)],
                }
            )
            with patch(
                "ps_dare.predict_arima.forecast_order", return_value=completed
            ) as mocked_forecast:
                outputs = _forecast_order_isolated(
                    (order_row(path), 0, model_path, prediction_path, False, None)
                )

            metrics_path = root / "metrics" / "model.csv"
            self.assertEqual(outputs, [prediction_path, metrics_path])
            self.assertTrue(prediction_path.is_file())
            self.assertTrue(metrics_path.is_file())
            saved = pd.read_csv(prediction_path)
            self.assertEqual(saved["model_id"].nunique(), 1)
            self.assertEqual(
                mocked_forecast.call_args.kwargs["model_output_path"], model_path
            )

    def test_prediction_filename_places_progressive_model_before_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary), with_driver=True)
            row = order_row(dataset, driver="NO2_mean", lag="1")

            output = _prediction_output_path(row, 6, "20260914_153012")

        self.assertEqual(
            output.name,
            "general_no2_mean_weekly_lag_1_model_007_20260914_153012.csv",
        )

    def test_builds_one_complete_metrics_row_from_testing_predictions(self):
        predictions = pd.DataFrame(
            {
                "model_id": ["model_007_ili_giustiniani_cat_no2_mean"] * 3,
                "model_number": [7] * 3,
                "aggregation": ["weekly"] * 3,
                "driver_kind": ["pollution"] * 3,
                "drivers_used": ["NO2_mean"] * 3,
                "lag": ["1"] * 3,
                "discretization": ["continuous"] * 3,
                "category": ["respiratory"] * 3,
                "path_to_training_file": [
                    "/data/weekly/ili_giustiniani_cat.csv"
                ]
                * 3,
                "date_start_training": ["2016-01-01"] * 3,
                "date_end_training": ["2022-12-31"] * 3,
                "date_end_testing": ["2026-01-31"] * 3,
                "date": pd.to_datetime(
                    ["2022-12-26", "2023-01-02", "2023-01-09"]
                ),
                "phase": ["training", "testing", "testing"],
                "n_accesses": [10.0, 20.0, 40.0],
                "prediction": [None, 18.0, 44.0],
                "p": [1] * 3,
                "d": [0] * 3,
                "q": [1] * 3,
                "P": [0] * 3,
                "D": [0] * 3,
                "Q": [0] * 3,
                "seasonal_period": [52] * 3,
                "intercept": [False] * 3,
                "update_mode": ["state_only"] * 3,
                "update_maxiter": [None] * 3,
                "model_file": ["/models/model.pkl"] * 3,
                "training_aic": [123.5] * 3,
            }
        )

        metrics = model_metrics(predictions)

        self.assertEqual(len(metrics), 1)
        row = metrics.iloc[0]
        self.assertEqual(row["dataset_name"], "ili_giustiniani_cat")
        self.assertEqual(row["hospital"], "giustiniani")
        self.assertEqual(row["subset"], "ili")
        self.assertTrue(row["categorized_dataset"])
        self.assertEqual(row["temporal_aggregation"], "weekly")
        self.assertEqual(row["testing_start"], "2023-01-02")
        self.assertEqual(row["scored_predictions"], 2)
        self.assertEqual(row["aic"], 123.5)
        self.assertAlmostEqual(row["ae"], 3.0)
        self.assertAlmostEqual(row["total_absolute_error"], 6.0)
        self.assertAlmostEqual(row["mape"], 10.0)

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

    def test_cli_accepts_the_auto_arima_manifest_with_orders_option(self):
        manifest = "data/orders/20260915_114627/orders_20260915_114627.csv"
        with (
            patch(
                "sys.argv",
                [
                    "predict_arima",
                    "--orders",
                    manifest,
                    "--jobs",
                    "3",
                    "--state-only-updates",
                ],
            ),
            patch("ps_dare.predict_arima.run", return_value=[]) as mocked_run,
        ):
            main()

        mocked_run.assert_called_once_with(
            manifest,
            jobs=3,
            state_only_updates=True,
            update_maxiter=None,
        )

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
                    / "output"
                    / "20260909_153012"
                    / "weekly"
                    / "predictions"
                    / "general_20260909_153012.csv",
                    root
                    / "data"
                    / "output"
                    / "20260909_153012"
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
