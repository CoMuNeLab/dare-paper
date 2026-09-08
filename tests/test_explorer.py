import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ps_dare.explorer import (
    ALL_CATEGORIES,
    DatasetChoice,
    categories,
    discover_datasets,
    draw_timeseries,
    driver_columns,
    load_dataset,
    missing_dates,
    prepare_timeseries,
)


class ExplorerTests(unittest.TestCase):
    def test_discovers_supported_aggregated_files_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "weekly" / "aggregated" / "continuous"
            folder.mkdir(parents=True)
            (folder / "general.csv").touch()
            (folder / "general.pkl").touch()
            (folder / "notes.txt").touch()

            choices = discover_datasets(root)

        self.assertEqual([choice.file_format for choice in choices], ["CSV", "Pickle"])
        self.assertTrue(all(choice.frequency == "weekly" for choice in choices))
        self.assertTrue(all(choice.kind == "continuous" for choice in choices))

    def test_load_csv_removes_saved_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.csv"
            pd.DataFrame({"n_accesses": [2], "temperature": [8]}).to_csv(path)
            choice = DatasetChoice("weekly", "continuous", "sample", "CSV", path)

            result = load_dataset(choice)

        self.assertEqual(result.columns.tolist(), ["n_accesses", "temperature"])

    def test_categorized_data_sums_cases_but_not_repeated_drivers(self):
        frame = pd.DataFrame(
            {
                "month": ["2000-01", "2000-01", "2000-01", "2000-01"],
                "valid_time": [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-02-29",
                    "2024-02-29",
                ],
                "diagnosis_category": ["A", "B", "A", "B"],
                "n_accesses": [2, 3, 5, 7],
                "temperature": [8.0, 8.0, 9.0, 9.0],
                "month_index": ["January", "January", "February", "February"],
            }
        )

        all_data = prepare_timeseries(frame, "monthly")
        category_a = prepare_timeseries(frame, "monthly", "A")

        self.assertEqual(all_data["n_accesses"].tolist(), [5, 12])
        self.assertEqual(all_data["temperature"].tolist(), [8.0, 9.0])
        self.assertEqual(category_a["n_accesses"].tolist(), [2, 5])
        self.assertEqual(driver_columns(frame), ["temperature"])
        self.assertEqual(categories(frame), [ALL_CATEGORIES, "A", "B"])

    def test_draws_cases_and_drivers_in_separate_panels(self):
        from matplotlib.figure import Figure

        data = pd.DataFrame(
            {"n_accesses": [2, 3], "temperature": [8.0, 9.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-08"]),
        )

        cases_axis, drivers_axis = draw_timeseries(
            Figure(), data, ["temperature"], "Example", "weekly"
        )

        self.assertEqual(len(cases_axis.lines), 1)
        self.assertEqual(len(drivers_axis.lines), 1)
        self.assertEqual(cases_axis.get_ylabel(), "Cases")
        self.assertEqual(drivers_axis.get_ylabel(), "Driver value")

    def test_marks_absent_periods_and_null_displayed_values(self):
        data = pd.DataFrame(
            {
                "n_accesses": [2.0, 3.0, 4.0],
                "temperature": [8.0, float("nan"), 10.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-22"]),
        )

        result = missing_dates(data, "weekly", ["n_accesses", "temperature"])

        self.assertEqual(
            result.tolist(),
            pd.to_datetime(["2024-01-08", "2024-01-15"]).tolist(),
        )

    def test_plot_adds_dashed_lines_to_both_panels(self):
        from matplotlib.figure import Figure

        data = pd.DataFrame(
            {"n_accesses": [2, 4], "temperature": [8.0, 10.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-15"]),
        )

        cases_axis, drivers_axis = draw_timeseries(
            Figure(), data, ["temperature"], "Example", "weekly"
        )

        self.assertEqual(cases_axis.lines[-1].get_linestyle(), "--")
        self.assertEqual(cases_axis.lines[-1].get_color(), "black")
        self.assertEqual(drivers_axis.lines[-1].get_linestyle(), "--")

    def test_missing_line_is_emphasized_only_in_affected_panel(self):
        from matplotlib.figure import Figure

        data = pd.DataFrame(
            {
                "n_accesses": [float("nan"), 3.0, 4.0],
                "temperature": [8.0, float("nan"), 10.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        )

        cases_axis, drivers_axis = draw_timeseries(
            Figure(), data, ["temperature"], "Example", "weekly"
        )
        case_missing_case, case_missing_driver = cases_axis.lines[-2:]
        driver_missing_case, driver_missing_driver = drivers_axis.lines[-2:]

        self.assertEqual(case_missing_case.get_alpha(), 1)
        self.assertEqual(driver_missing_case.get_alpha(), 0.2)
        self.assertEqual(case_missing_driver.get_alpha(), 0.2)
        self.assertEqual(driver_missing_driver.get_alpha(), 1)
        self.assertEqual(case_missing_case.get_color(), "black")
        self.assertEqual(driver_missing_case.get_color(), "grey")
        self.assertEqual(case_missing_driver.get_color(), "grey")
        self.assertEqual(driver_missing_driver.get_color(), "black")
        self.assertLess(
            driver_missing_case.get_linewidth(), case_missing_case.get_linewidth()
        )
        self.assertLess(
            case_missing_driver.get_linewidth(), driver_missing_driver.get_linewidth()
        )


if __name__ == "__main__":
    unittest.main()
