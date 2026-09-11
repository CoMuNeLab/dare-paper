"""Interactive explorer for aggregated PS-DARE case and driver data.

From a repository checkout, run the interface with
``python -m src.ps_dare.explorer``.  The data-selection and plotting functions
are kept independent from Tk so they can also be reused in notebooks and tested
on machines without a display.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import solve_path

CASE_COLUMN = "n_accesses"
CATEGORY_COLUMN = "diagnosis_category"
ALL_CATEGORIES = "All categories"
SUPPORTED_FORMATS = {".csv": "CSV", ".pkl": "Pickle"}
NON_DRIVER_COLUMNS = {
    CASE_COLUMN,
    CATEGORY_COLUMN,
    "month",
    "week",
    "month_index",
    "valid_time",
}


@dataclass(frozen=True, order=True)
class DatasetChoice:
    """One file found below ``data/<frequency>/aggregated/<kind>``."""

    frequency: str
    kind: str
    name: str
    file_format: str
    path: Path


def discover_datasets(data_dir: str | Path | None = None) -> list[DatasetChoice]:
    """Return the CSV and pickle datasets available in aggregated folders."""
    root = Path(data_dir) if data_dir is not None else solve_path("data")
    choices: list[DatasetChoice] = []
    if not root.is_dir():
        return choices

    for frequency_dir in root.iterdir():
        aggregated = frequency_dir / "aggregated"
        if not aggregated.is_dir():
            continue
        for kind_dir in aggregated.iterdir():
            if not kind_dir.is_dir():
                continue
            for path in kind_dir.iterdir():
                file_format = SUPPORTED_FORMATS.get(path.suffix.lower())
                if file_format:
                    choices.append(
                        DatasetChoice(
                            frequency=frequency_dir.name,
                            kind=kind_dir.name,
                            name=path.stem,
                            file_format=file_format,
                            path=path,
                        )
                    )
    return sorted(choices)


def load_dataset(choice: DatasetChoice) -> pd.DataFrame:
    """Load one dataset and discard CSV index columns."""
    if choice.file_format == "CSV":
        frame = pd.read_csv(choice.path)
    elif choice.file_format == "Pickle":
        frame = pd.read_pickle(choice.path)
    else:
        raise ValueError(f"Unsupported file format: {choice.file_format!r}")

    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]
    if CASE_COLUMN not in frame:
        raise ValueError(f"{choice.path} has no {CASE_COLUMN!r} column")
    return frame


def date_column(frame: pd.DataFrame, frequency: str) -> str:
    """Return the normalized date column used as the shared x axis."""
    if "valid_time" not in frame:
        raise ValueError("Dataset has no 'valid_time' column")
    return "valid_time"


def driver_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric data columns that represent selectable drivers."""
    drivers: list[str] = []
    for column in frame.columns:
        if column in NON_DRIVER_COLUMNS or str(column).startswith("Unnamed:"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            drivers.append(str(column))
    return drivers


def categories(frame: pd.DataFrame) -> list[str]:
    """Return the category choices applicable to *frame*."""
    if CATEGORY_COLUMN not in frame:
        return [ALL_CATEGORIES]
    values = sorted(frame[CATEGORY_COLUMN].dropna().astype(str).unique())
    return [ALL_CATEGORIES, *values]


def prepare_timeseries(
    frame: pd.DataFrame,
    frequency: str,
    category: str = ALL_CATEGORIES,
) -> pd.DataFrame:
    """Reduce an aggregated file to one case/driver row per date.

    Case counts are summed when all diagnosis categories are shown.  Drivers
    use the first value at each date because categorized files repeat the same
    environmental measurement for every category.
    """
    period_column = date_column(frame, frequency)
    work = frame.copy()
    work["__date__"] = pd.to_datetime(work[period_column], errors="coerce")
    work = work.dropna(subset=["__date__"])

    if category != ALL_CATEGORIES and CATEGORY_COLUMN in work:
        work = work.loc[work[CATEGORY_COLUMN].astype(str).eq(category)]

    drivers = driver_columns(work)
    grouped = work.groupby("__date__", sort=True, as_index=True)
    cases = grouped[CASE_COLUMN].sum(min_count=1)
    result = pd.concat(
        [cases, grouped[drivers].first()] if drivers else [cases],
        axis="columns",
    )
    result.index.name = "date"
    return result


def missing_dates(
    data: pd.DataFrame,
    frequency: str,
    columns: Iterable[str],
) -> pd.DatetimeIndex:
    """Find absent periods and null values in the displayed series."""
    dates = pd.DatetimeIndex(data.index).dropna().sort_values().unique()
    if dates.empty:
        return dates

    if frequency == "monthly":
        actual_periods = dates.to_period("M")
        expected_periods = pd.period_range(
            actual_periods.min(), actual_periods.max(), freq="M"
        )
        absent = expected_periods.difference(actual_periods).to_timestamp()
    elif frequency == "weekly":
        expected = pd.date_range(dates.min(), dates.max(), freq="7D")
        absent = expected.difference(dates)
    else:
        raise ValueError(f"Unsupported frequency: {frequency!r}")

    displayed = [column for column in columns if column in data]
    null_dates = pd.DatetimeIndex([])
    if displayed:
        null_dates = pd.DatetimeIndex(data.index[data[displayed].isna().any(axis=1)])
    return absent.union(null_dates).sort_values()


def draw_timeseries(
    figure,
    data: pd.DataFrame,
    selected_drivers: Iterable[str],
    title: str,
    frequency: str,
) -> tuple[object, object]:
    """Draw cases and selected drivers as two stacked horizontal panels."""
    figure.clear()
    cases_axis, drivers_axis = figure.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1.35), "hspace": 0.08},
    )

    cases_axis.plot(data.index, data[CASE_COLUMN], color="#b42318", linewidth=1.8)
    cases_axis.fill_between(
        data.index, data[CASE_COLUMN], color="#f97066", alpha=0.18, linewidth=0
    )
    cases_axis.set_ylabel("Cases")
    cases_axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    cases_axis.grid(axis="y", alpha=0.25)

    valid_drivers = [column for column in selected_drivers if column in data]
    for column in valid_drivers:
        drivers_axis.plot(data.index, data[column], linewidth=1.35, label=column)

    missing_cases = missing_dates(data, frequency, [CASE_COLUMN])
    missing_drivers = missing_dates(data, frequency, valid_drivers)
    missing = missing_cases.union(missing_drivers).sort_values()
    missing_case_set = set(missing_cases)
    missing_driver_set = set(missing_drivers)
    for index, date in enumerate(missing):
        case_is_affected = date in missing_case_set
        driver_is_affected = date in missing_driver_set
        cases_axis.axvline(
            date,
            color="black" if case_is_affected else "grey",
            linestyle="--",
            linewidth=0.9 if case_is_affected else 0.5,
            alpha=1.0 if case_is_affected else 0.2,
        )
        drivers_axis.axvline(
            date,
            color="black" if driver_is_affected else "grey",
            linestyle="--",
            linewidth=0.9 if driver_is_affected else 0.5,
            alpha=1.0 if driver_is_affected else 0.2,
            label="Missing data" if index == 0 else None,
        )
    drivers_axis.set_ylabel("Driver value")
    drivers_axis.set_xlabel("Date")
    drivers_axis.grid(alpha=0.25)
    if valid_drivers or len(missing):
        legend_columns = min(3, len(valid_drivers) + bool(len(missing)))
        drivers_axis.legend(loc="upper left", ncols=legend_columns, fontsize=8)
    else:
        drivers_axis.text(
            0.5,
            0.5,
            "Select one or more drivers from the list",
            transform=drivers_axis.transAxes,
            ha="center",
            va="center",
            color="#667085",
        )
    for label in drivers_axis.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")
    return cases_axis, drivers_axis


class DatasetExplorer:
    """Tk interface that updates the Matplotlib panels as selections change."""

    def __init__(self, root, choices: list[DatasetChoice]):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.root = root
        self.choices = choices
        self.frame: pd.DataFrame | None = None
        self.current_choice: DatasetChoice | None = None
        self._refreshing = False

        root.title("PS-DARE case and driver explorer")
        root.geometry("1280x780")
        root.minsize(940, 620)

        self.frequency_var = tk.StringVar()
        self.kind_var = tk.StringVar()
        self.dataset_var = tk.StringVar()
        self.format_var = tk.StringVar()
        self.category_var = tk.StringVar(value=ALL_CATEGORIES)
        self.status_var = tk.StringVar(value="Loading data…")

        controls = ttk.Frame(root, padding=(12, 10, 12, 8))
        controls.pack(fill="x")
        controls.columnconfigure(2, weight=1)

        self.frequency_box = self._combobox(controls, "Frequency", self.frequency_var, 0, 14)
        self.kind_box = self._combobox(controls, "Driver kind", self.kind_var, 1, 14)
        self.dataset_box = self._combobox(controls, "Dataset", self.dataset_var, 2, 28)
        self.format_box = self._combobox(controls, "Format", self.format_var, 3, 10)
        self.category_box = self._combobox(controls, "Category", self.category_var, 4, 22)

        body = ttk.Frame(root, padding=(12, 0, 12, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar = ttk.LabelFrame(body, text="Drivers", padding=8)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        sidebar.rowconfigure(0, weight=1)

        self.driver_list = tk.Listbox(
            sidebar,
            selectmode=tk.EXTENDED,
            exportselection=False,
            width=27,
            activestyle="dotbox",
        )
        driver_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.driver_list.yview)
        self.driver_list.configure(yscrollcommand=driver_scroll.set)
        self.driver_list.grid(row=0, column=0, sticky="ns")
        driver_scroll.grid(row=0, column=1, sticky="ns")
        self.driver_list.bind("<<ListboxSelect>>", self._redraw)

        buttons = ttk.Frame(sidebar)
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Select all", command=self._select_all).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self._clear_drivers).pack(side="left", padx=5)

        plot_frame = ttk.Frame(body)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        self.figure = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

        ttk.Label(root, textvariable=self.status_var, anchor="w", padding=(12, 3, 12, 8)).pack(
            fill="x"
        )

        for box in (self.frequency_box, self.kind_box, self.dataset_box, self.format_box):
            box.bind("<<ComboboxSelected>>", self._dataset_selection_changed)
        self.category_box.bind("<<ComboboxSelected>>", self._category_changed)

        self._initialise_selection()

    def _combobox(self, parent, label: str, variable, column: int, width: int):
        from tkinter import ttk

        wrapper = ttk.Frame(parent)
        wrapper.grid(row=0, column=column, sticky="ew", padx=(0, 10))
        ttk.Label(wrapper, text=label).pack(anchor="w")
        box = ttk.Combobox(wrapper, textvariable=variable, state="readonly", width=width)
        box.pack(fill="x")
        return box

    @staticmethod
    def _set_options(box, variable, values: Iterable[str], preferred: str = "") -> None:
        values = list(values)
        box["values"] = values
        variable.set(preferred if preferred in values else (values[0] if values else ""))

    def _initialise_selection(self) -> None:
        frequencies = sorted({choice.frequency for choice in self.choices})
        self._set_options(self.frequency_box, self.frequency_var, frequencies)
        self._refresh_cascading_controls()
        self._load_selected_dataset()

    def _dataset_selection_changed(self, _event=None) -> None:
        if self._refreshing:
            return
        self._refresh_cascading_controls()
        self._load_selected_dataset()

    def _refresh_cascading_controls(self) -> None:
        self._refreshing = True
        try:
            frequency = self.frequency_var.get()
            matching = [choice for choice in self.choices if choice.frequency == frequency]
            self._set_options(
                self.kind_box,
                self.kind_var,
                sorted({choice.kind for choice in matching}),
                self.kind_var.get(),
            )
            matching = [choice for choice in matching if choice.kind == self.kind_var.get()]
            self._set_options(
                self.dataset_box,
                self.dataset_var,
                sorted({choice.name for choice in matching}),
                self.dataset_var.get(),
            )
            matching = [choice for choice in matching if choice.name == self.dataset_var.get()]
            self._set_options(
                self.format_box,
                self.format_var,
                sorted({choice.file_format for choice in matching}),
                self.format_var.get(),
            )
        finally:
            self._refreshing = False

    def _selected_choice(self) -> DatasetChoice | None:
        key = (
            self.frequency_var.get(),
            self.kind_var.get(),
            self.dataset_var.get(),
            self.format_var.get(),
        )
        return next(
            (
                choice
                for choice in self.choices
                if (choice.frequency, choice.kind, choice.name, choice.file_format) == key
            ),
            None,
        )

    def _load_selected_dataset(self) -> None:
        choice = self._selected_choice()
        if choice is None:
            self.status_var.set("No matching aggregated dataset")
            return
        try:
            self.frame = load_dataset(choice)
            self.current_choice = choice
        except Exception as exc:  # Keep the GUI alive and show a useful error.
            self.frame = None
            self.current_choice = choice
            self.status_var.set(f"Could not load {choice.path.name}: {exc}")
            self.figure.clear()
            self.canvas.draw_idle()
            return

        self._set_options(
            self.category_box,
            self.category_var,
            categories(self.frame),
            self.category_var.get(),
        )
        category_state = "readonly" if CATEGORY_COLUMN in self.frame else "disabled"
        self.category_box.configure(state=category_state)

        previous = self.selected_drivers()
        available = driver_columns(self.frame)
        self.driver_list.delete(0, "end")
        for driver in available:
            self.driver_list.insert("end", driver)
        selected = [index for index, driver in enumerate(available) if driver in previous]
        if not selected:
            selected = list(range(min(3, len(available))))
        for index in selected:
            self.driver_list.selection_set(index)
        self._redraw()

    def selected_drivers(self) -> list[str]:
        return [self.driver_list.get(index) for index in self.driver_list.curselection()]

    def _category_changed(self, _event=None) -> None:
        self._redraw()

    def _redraw(self, _event=None) -> None:
        if self.frame is None or self.current_choice is None:
            return
        try:
            data = prepare_timeseries(
                self.frame, self.current_choice.frequency, self.category_var.get()
            )
            title = (
                f"{self.current_choice.name} · {self.current_choice.frequency} · "
                f"{self.current_choice.kind}"
            )
            draw_timeseries(
                self.figure,
                data,
                self.selected_drivers(),
                title,
                self.current_choice.frequency,
            )
            self.canvas.draw_idle()
            self.status_var.set(
                f"{len(data):,} dates · {len(self.frame):,} source rows · "
                f"{self.current_choice.path}"
            )
        except Exception as exc:
            self.status_var.set(f"Could not plot dataset: {exc}")

    def _select_all(self) -> None:
        self.driver_list.selection_set(0, "end")
        self._redraw()

    def _clear_drivers(self) -> None:
        self.driver_list.selection_clear(0, "end")
        self._redraw()


def launch(data_dir: str | Path | None = None) -> None:
    """Discover datasets and start the desktop interface."""
    import tkinter as tk
    from tkinter import messagebox

    choices = discover_datasets(data_dir)
    if not choices:
        location = Path(data_dir) if data_dir is not None else solve_path("data")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "PS-DARE explorer",
            f"No .csv or .pkl files were found below\n{location}/*/aggregated/*",
        )
        root.destroy()
        return

    root = tk.Tk()
    DatasetExplorer(root, choices)
    root.mainloop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="data directory containing monthly/ and weekly/ (defaults to DATA_BASE_DIR/data)",
    )
    args = parser.parse_args(argv)
    launch(args.data_dir)


if __name__ == "__main__":
    main()
