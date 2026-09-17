import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ps_dare.recover_arima_pipeline import (
    LoggedConfiguration,
    RecoveryItem,
    RecoveryLog,
    _recover_one,
    read_recovery_log,
    recover,
)


class RecoverArimaPipelineTests(unittest.TestCase):
    def test_reads_legacy_log_and_infers_state_only_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "arima_pipeline_20260915_163546.log"
            log.write_text(
                "2026-09-15 16:35:46 INFO pipeline: Starting ARIMA pipeline "
                "for relative.csv\n"
                "2026-09-15 16:35:46 INFO selector: Loaded 1 analysis "
                "specification(s) from /project/configuration.csv\n"
                "2026-09-15 16:36:31 INFO selector: Saved completed model "
                "order to /data/orders/20260915_163546/model/orders_"
                "20260915_163546.csv\n"
                "2026-09-15 16:36:32 INFO predictor: Using model_001 trained "
                "on 10 observations; forecasting 2 one-step periods with "
                "state-only updates\n",
                encoding="utf-8",
            )

            recovered = read_recovery_log(log)

        self.assertEqual(
            recovered.configurations,
            (LoggedConfiguration("/project/configuration.csv", "20260915_163546"),),
        )
        self.assertEqual(recovered.jobs, 1)
        self.assertTrue(recovered.state_only_updates)
        self.assertIsNone(recovered.update_maxiter)

    def test_prefers_invocation_metadata_in_new_logs(self):
        metadata = {
            "configurations": ["first.csv", "second.csv"],
            "jobs": 4,
            "state_only_updates": False,
            "update_maxiter": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "arima_pipeline_20260915_120000.log"
            log.write_text(
                "2026-09-15 12:00:00 INFO pipeline: Pipeline invocation: "
                + json.dumps(metadata)
                + "\n",
                encoding="utf-8",
            )

            recovered = read_recovery_log(log)

        self.assertEqual(
            recovered.configurations,
            (
                LoggedConfiguration("first.csv", "20260915_120000"),
                LoggedConfiguration("second.csv", "20260915_120000"),
            ),
        )
        self.assertEqual(recovered.jobs, 4)
        self.assertFalse(recovered.state_only_updates)
        self.assertEqual(recovered.update_maxiter, 2)

    @patch("ps_dare.recover_arima_pipeline.predict_arima.run")
    def test_prediction_recovery_reuses_saved_order(self, prediction_run):
        prediction_run.return_value = [Path("prediction.csv"), Path("metrics.csv")]
        item = RecoveryItem(
            "predict",
            pd.Series(dtype="object"),
            "20260915_120000",
            Path("orders_20260915_120000.csv"),
        )

        outputs = _recover_one((item, True, None))

        self.assertEqual(outputs, [Path("prediction.csv"), Path("metrics.csv")])
        prediction_run.assert_called_once_with(
            Path("orders_20260915_120000.csv"),
            jobs=1,
            state_only_updates=True,
        )

    @patch("ps_dare.recover_arima_pipeline.auto_arima.save_skipped", return_value=[])
    @patch("ps_dare.recover_arima_pipeline._run_recovery_tasks", return_value=[])
    @patch("ps_dare.recover_arima_pipeline._recovery_item")
    @patch("ps_dare.recover_arima_pipeline.auto_arima._expanded_model_rows")
    @patch("ps_dare.recover_arima_pipeline.auto_arima.read_configuration")
    @patch("ps_dare.recover_arima_pipeline.read_recovery_log")
    def test_recover_only_schedules_incomplete_models(
        self,
        read_log,
        read_configuration,
        expanded_rows,
        recovery_item,
        run_tasks,
        save_skipped,
    ):
        read_log.return_value = RecoveryLog(
            (LoggedConfiguration("configuration.csv", "20260915_120000"),),
            3,
            True,
            None,
        )
        read_configuration.return_value = pd.DataFrame()
        first = pd.Series({"model_number": 1})
        second = pd.Series({"model_number": 2})
        expanded_rows.return_value = ([first, second], [])
        incomplete = RecoveryItem(
            "select", second, "20260915_120000", Path("orders.csv")
        )
        recovery_item.side_effect = [None, incomplete]

        result = recover("pipeline.log")

        self.assertEqual(result.already_complete, 1)
        self.assertEqual(result.selected, 1)
        run_tasks.assert_called_once()
        tasks, jobs, queue = run_tasks.call_args.args
        self.assertEqual(tasks, [(incomplete, True, None)])
        self.assertEqual(jobs, 3)
        self.assertIsNone(queue)
        save_skipped.assert_not_called()


if __name__ == "__main__":
    unittest.main()
