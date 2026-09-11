import unittest
from pathlib import Path
from unittest.mock import patch

from ps_dare.arima_pipeline import run


class ArimaPipelineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
