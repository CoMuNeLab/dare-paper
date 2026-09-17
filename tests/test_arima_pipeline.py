import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ps_dare.arima_pipeline import LOGGER, _select_predict_and_save, main, run


class ArimaPipelineTests(unittest.TestCase):
    def test_runs_configured_baseline_without_fitting_arima(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "general.csv"
            pd.DataFrame(
                {
                    "valid_time": pd.date_range(
                        "2024-01-01", periods=6, freq="7D"
                    ),
                    "n_accesses": [10, 11, 12, 20, 30, 40],
                }
            ).to_csv(dataset, index=False)
            configuration = root / "configuration.csv"
            pd.DataFrame(
                [
                    {
                        "dataset": str(dataset),
                        "aggregation": "weekly",
                        "driver_kind": "baseline",
                        "discretization": "continuous",
                        "train_start": "2024-01-01",
                        "train_end": "2024-01-15",
                        "test_end": "2024-01-29",
                        "lag": 0,
                    }
                ]
            ).to_csv(configuration, index=False)

            with (
                patch.dict(__import__("os").environ, {"DATA_BASE_DIR": str(root)}),
                patch(
                    "ps_dare.arima_pipeline.auto_arima._analyze_and_save_model",
                    side_effect=AssertionError("baseline must not fit ARIMA"),
                ),
            ):
                outputs = run([configuration])

            self.assertEqual(len(outputs), 2)
            predictions = pd.read_csv(outputs[0])
            metrics = pd.read_csv(outputs[1])
            self.assertEqual(predictions["method"].unique().tolist(), ["baseline"])
            self.assertEqual(
                predictions.loc[
                    predictions["phase"].eq("testing"), "prediction"
                ].tolist(),
                [12.0, 20.0],
            )
            self.assertTrue(predictions["prediction_lower_bound"].isna().all())
            self.assertTrue(predictions["prediction_upper_bound"].isna().all())
            self.assertTrue(metrics["aic"].isna().all())

    @patch("ps_dare.arima_pipeline.predict_arima.run")
    @patch("ps_dare.arima_pipeline.auto_arima.iter_run")
    def test_selects_calibrates_saves_and_tests_each_configuration(
        self, auto_iter, predict_run
    ):
        first_orders = Path("data/weekly/orders/orders_20260911_120000.csv")
        second_orders = Path("data/monthly/orders/orders_20260911_120001.csv")
        skipped = Path("data/weekly/orders/skipped.csv")
        first_predictions = Path(
            "data/weekly/predictions/general_20260911_120000.csv"
        )
        second_predictions = Path(
            "data/monthly/predictions/general_20260911_120001.csv"
        )
        auto_iter.side_effect = [
            [first_orders, skipped],
            [second_orders],
        ]
        predict_run.side_effect = [[first_predictions], [second_predictions]]

        outputs = run(["first.csv", "second.csv"])

        self.assertEqual(
            outputs,
            [
                first_orders,
                first_predictions,
                skipped,
                second_orders,
                second_predictions,
            ],
        )
        self.assertEqual(auto_iter.call_args_list[0].args, ("first.csv",))
        self.assertEqual(auto_iter.call_args_list[0].kwargs, {"jobs": 1})
        self.assertEqual(auto_iter.call_args_list[1].args, ("second.csv",))
        self.assertEqual(auto_iter.call_args_list[1].kwargs, {"jobs": 1})
        self.assertEqual(predict_run.call_args_list[0].args, (first_orders,))
        self.assertEqual(predict_run.call_args_list[0].kwargs, {"jobs": 1})
        self.assertEqual(predict_run.call_args_list[1].args, (second_orders,))
        self.assertEqual(predict_run.call_args_list[1].kwargs, {"jobs": 1})

    @patch("ps_dare.arima_pipeline.predict_arima.run")
    @patch("ps_dare.arima_pipeline.auto_arima.iter_run")
    def test_skips_testing_when_every_model_is_ineligible(
        self, auto_iter, predict_run
    ):
        skipped = Path("data/weekly/orders/skipped.csv")
        auto_iter.return_value = [skipped]

        self.assertEqual(run(["configuration.csv"]), [skipped])
        predict_run.assert_not_called()

    def test_requires_a_configuration(self):
        with self.assertRaisesRegex(ValueError, "At least one configuration"):
            run([])

    def test_rejects_nonpositive_jobs(self):
        with self.assertRaisesRegex(ValueError, "jobs must be at least 1"):
            run(["configuration.csv"], jobs=0)

    @patch("ps_dare.arima_pipeline.gc.collect")
    @patch("ps_dare.arima_pipeline.predict_arima.run")
    @patch("ps_dare.arima_pipeline.auto_arima._analyze_and_save_model")
    def test_completes_selection_and_save_before_predicting(
        self, select_and_save, predict_run, collect
    ):
        order_path = Path("orders_20260914_153012.csv")
        prediction_path = Path("prediction.csv")
        events = []

        def save_order(task):
            events.append("model_saved")
            return order_path

        def save_prediction(path, jobs):
            self.assertEqual(events, ["model_saved"])
            events.append("prediction_saved")
            return [prediction_path]

        select_and_save.side_effect = save_order
        predict_run.side_effect = save_prediction

        outputs = _select_predict_and_save(
            (object(), "20260914_153012", False, None)
        )

        self.assertEqual(events, ["model_saved", "prediction_saved"])
        self.assertEqual(outputs, [order_path, prediction_path])
        predict_run.assert_called_once_with(order_path, jobs=1)
        collect.assert_called_once_with()

    @patch("ps_dare.arima_pipeline.gc.collect")
    @patch("ps_dare.arima_pipeline.predict_arima.run", return_value=[])
    @patch(
        "ps_dare.arima_pipeline.auto_arima._analyze_and_save_model",
        return_value=Path("orders_20260914_153012.csv"),
    )
    def test_passes_faster_update_mode_to_prediction(
        self, select_and_save, predict_run, collect
    ):
        _select_predict_and_save(
            (object(), "20260914_153012", True, None)
        )
        predict_run.assert_called_once_with(
            Path("orders_20260914_153012.csv"),
            jobs=1,
            state_only_updates=True,
        )

        predict_run.reset_mock()
        _select_predict_and_save(
            (object(), "20260914_153012", False, 1)
        )
        predict_run.assert_called_once_with(
            Path("orders_20260914_153012.csv"),
            jobs=1,
            update_maxiter=1,
        )

    @patch("ps_dare.arima_pipeline.run")
    def test_cli_logs_to_the_screen_and_a_file(self, pipeline_run):
        pipeline_run.return_value = [Path("result.csv")]
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            with tempfile.TemporaryDirectory() as directory:
                log_file = Path(directory) / "pipeline.log"
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    patch(
                        "sys.argv",
                        [
                            "ps-dare-arima-pipeline",
                            "configuration.csv",
                            "--log-file",
                            str(log_file),
                        ],
                    ),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    main()
                    LOGGER.info("pipeline logging test")

                self.assertIn("pipeline logging test", stderr.getvalue())
                self.assertIn("pipeline logging test", log_file.read_text())
                self.assertIn("Saved result.csv", stdout.getvalue())
        finally:
            for handler in root_logger.handlers:
                if handler not in original_handlers:
                    handler.close()
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
