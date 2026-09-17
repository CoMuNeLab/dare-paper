import unittest
import warnings

import pandas as pd

from ps_dare.assembly import _fill_internal_case_gaps, merge_driver


class MergeDriverTests(unittest.TestCase):
    def test_fills_only_case_gaps_inside_hospital_timeline(self):
        data = pd.DataFrame(
            {
                "valid_time": pd.to_datetime(
                    ["2023-12-25", "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]
                ),
                "n_accesses": [None, 3, None, 4, None],
            }
        )

        result = _fill_internal_case_gaps(data)

        self.assertTrue(pd.isna(result.loc[0, "n_accesses"]))
        self.assertEqual(result.loc[2, "n_accesses"], 0)
        self.assertTrue(pd.isna(result.loc[4, "n_accesses"]))

    def test_weekly_merge_aligns_driver_end_date_with_next_week_start(self):
        cases = pd.DataFrame({"week": ["2024-01-01", "2024-01-08"], "cases": [2, 3]})
        driver = pd.DataFrame({"valid_time": ["2024-01-07"], "temperature": [10]})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = merge_driver(cases, driver, "weekly")

        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.loc[0, "temperature"]))
        self.assertEqual(result.loc[1, "temperature"], 10)

    def test_merge_preserves_driver_only_dates_with_missing_case_values(self):
        cases = pd.DataFrame({"month": ["2024-01-01"], "cases": [2]})
        driver = pd.DataFrame(
            {
                "valid_time": ["2024-01-31", "2024-02-29"],
                "temperature": [10, 12],
            }
        )

        result = merge_driver(cases, driver, "monthly")
        driver_only = result.loc[result["valid_time"] == pd.Timestamp("2024-02-29")]

        self.assertEqual(len(driver_only), 1)
        self.assertEqual(driver_only.iloc[0]["temperature"], 12)
        self.assertTrue(pd.isna(driver_only.iloc[0]["cases"]))

    def test_later_merge_preserves_earlier_driver_only_rows(self):
        cases = pd.DataFrame({"week": ["2024-01-08"], "cases": [2]})
        temperature = pd.DataFrame(
            {"valid_time": ["2024-01-07", "2024-01-14"], "temperature": [10, 12]}
        )
        humidity = pd.DataFrame(
            {"valid_time": ["2024-01-07", "2024-01-14"], "humidity": [70, 75]}
        )

        result = merge_driver(cases, temperature, "weekly")
        result = merge_driver(result, humidity, "weekly")
        driver_only = result.loc[result["valid_time"] == pd.Timestamp("2024-01-15")]

        self.assertEqual(len(driver_only), 1)
        self.assertEqual(driver_only.iloc[0]["temperature"], 12)
        self.assertEqual(driver_only.iloc[0]["humidity"], 75)
        self.assertTrue(pd.isna(driver_only.iloc[0]["cases"]))

    def test_merge_removes_rows_without_valid_time(self):
        cases = pd.DataFrame(
            {"week": ["2024-01-08", None, "invalid"], "cases": [2, 3, 4]}
        )
        driver = pd.DataFrame(
            {"valid_time": ["2024-01-07", None], "temperature": [10, 12]}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = merge_driver(cases, driver, "weekly")

        self.assertFalse(result["valid_time"].isna().any())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["cases"], 2)
        self.assertEqual(result.iloc[0]["temperature"], 10)

    def test_rejects_unknown_frequency(self):
        with self.assertRaises(ValueError):
            merge_driver(pd.DataFrame(), pd.DataFrame(), "daily")


if __name__ == "__main__":
    unittest.main()
