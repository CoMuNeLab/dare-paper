"""Find ARIMA orders for datasets described by a CSV configuration file.

Run this module from the repository root with::

    python -m ps_dare.auto_arima path/to/datasets.csv

The configuration accepts the following columns (spaces may be used instead of
underscores): ``dataset``, ``aggregation``, ``driver_kind``,
``discretization``, ``train_start``, ``train_end``, ``test_end``, and ``lag``.
``test_end`` is retained in the output but is not used yet.  ``forced`` uses
the predefined model ``(1, 1, 1)(2, 0, 0)[m]`` without exogenous drivers.
Datasets are fitted only when every training timestep has more than the
configured weekly or monthly case threshold.  Rejected datasets are recorded
in ``data/<weekly|monthly>/orders/skipped.csv`` instead of an order table.
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .explorer import ALL_CATEGORIES, driver_columns, prepare_timeseries
from .paths import solve_path

LOGGER = logging.getLogger(__name__)

CASE_COLUMN = "n_accesses"
# Every training timestep must be strictly above its aggregation's threshold.
MIN_CASES_PER_TIMESTEP = {"weekly": 1, "monthly": 10}
FREQUENCIES = {
    "week": "weekly",
    "weekly": "weekly",
    "month": "monthly",
    "moth": "monthly",
    "monthly": "monthly",
}
SEASONAL_PERIODS = {"weekly": 52, "monthly": 12}
DRIVER_KINDS = {"pollution", "weather", "none", "forced"}
DISCRETIZATIONS = {"discrete", "continuous"}
SELECTED_DRIVER_COLUMN = "_selected_driver"
MODEL_NUMBER_COLUMN = "model_number"
FORCED_ORDER = (1, 1, 1)
FORCED_SEASONAL_ORDER = (2, 0, 0)
FORCED_INTERCEPT = True

# Only drivers listed here are calibrated.  Each available driver produces a
# separate model; it is never combined with another driver from the list.
POLLUTION_DRIVERS = [
    "NO2_mean",
    "NO2_peak",
    "SO2_mean",
    "SO2_peak",
    "CO_mean",
    "CO_peak",
    "pm10_mean",
    "pm10_peak",
    "pm2.5_mean",
    "pm2.5_peak",
]
WEATHER_DRIVERS = [
    "humidex",
    "indoor_wet_bulb",
    "tropical_night",
    "summer_days",
    "frost",
    "heating_degree_days",
    "heat_wave_days",
    "cold_spell_days",
    "frost_days",
    "abs_hu",
    "hu",
]
DRIVERS_BY_KIND = {
    "pollution": POLLUTION_DRIVERS,
    "weather": WEATHER_DRIVERS,
}

COLUMN_ALIASES = {
    "dataset": ("dataset", "path", "file", "training_file", "path_to_training_file"),
    "aggregation": ("aggregation", "frequency", "temporal_aggregation"),
    "driver_kind": ("driver_kind", "kind_of_driver", "driver_type"),
    "discretization": (
        "discretization",
        "kind_of_discretization",
        "driver_discretization",
    ),
    "train_start": (
        "train_start",
        "training_start",
        "start_training",
        "start_date_training",
        "date_start_training",
    ),
    "train_end": (
        "train_end",
        "training_end",
        "end_training",
        "end_date_training",
        "date_end_training",
    ),
    "test_end": (
        "test_end",
        "testing_end",
        "end_testing",
        "end_date_testing",
        "date_end_testing",
    ),
    "lag": ("lag", "driver_lag", "lag_of_drivers"),
    "category": ("category", "diagnosis_category"),
}


def _column_name(value: object) -> str:
    """Normalize a human-readable configuration header."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_configuration(path: str | Path) -> pd.DataFrame:
    """Read and normalize the dataset configuration at *path*."""
    configuration = pd.read_csv(solve_path(path))
    configuration.columns = [_column_name(column) for column in configuration.columns]

    renames: dict[str, str] = {}
    for standard, aliases in COLUMN_ALIASES.items():
        match = next((alias for alias in aliases if alias in configuration), None)
        if match is not None:
            renames[match] = standard
    configuration = configuration.rename(columns=renames)

    required = {
        "dataset",
        "aggregation",
        "driver_kind",
        "discretization",
        "train_start",
        "train_end",
        "lag",
    }
    missing = sorted(required.difference(configuration.columns))
    if missing:
        raise ValueError(f"Missing configuration columns: {', '.join(missing)}")
    return configuration


def _frequency(value: object) -> str:
    key = str(value).strip().lower()
    if key not in FREQUENCIES:
        raise ValueError(f"Unsupported aggregation: {value!r}")
    return FREQUENCIES[key]


def _discretization(value: object) -> str:
    result = str(value).strip().lower()
    if result not in DISCRETIZATIONS:
        raise ValueError(f"Unsupported discretization: {value!r}")
    return result


def _driver_kind(value: object) -> str:
    result = str(value).strip().lower()
    if result not in DRIVER_KINDS:
        raise ValueError(f"Unsupported driver kind: {value!r}")
    return result


def _dataset_path(value: object, frequency: str, discretization: str) -> Path:
    """Resolve a configured path or a dataset name through ``solve_path``."""
    configured = Path(str(value).strip()).expanduser()
    if configured.is_absolute() or configured.parent != Path("."):
        return solve_path(configured)

    if not configured.suffix:
        configured = configured.with_suffix(".csv")
    return solve_path(
        Path("data") / frequency / "aggregated" / discretization / configured
    )


def load_training_file(path: Path) -> pd.DataFrame:
    """Load one CSV or pickle aggregated dataset."""
    if path.suffix.lower() == ".pkl":
        frame = pd.read_pickle(path)
    else:
        frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]


def select_drivers(frame: pd.DataFrame, kind: str) -> list[str]:
    """Select available exogenous columns from the configured driver lists."""
    available = driver_columns(frame)
    if kind in {"none", "forced"}:
        return []
    return [column for column in DRIVERS_BY_KIND[kind] if column in available]


def apply_driver_lag(
    data: pd.DataFrame, drivers: list[str], lag: object
) -> pd.DataFrame:
    """Return driver columns at lag 0, lag 1, or both lags."""
    normalized = str(lag).strip().replace(" ", "")
    if normalized in {"0", "0.0"}:
        lags = (0,)
    elif normalized in {"1", "1.0"}:
        lags = (1,)
    elif normalized in {"0-1", "0,1", "0/1"}:
        lags = (0, 1)
    else:
        raise ValueError(f"Unsupported driver lag: {lag!r}")

    parts = []
    for driver_lag in lags:
        part = data[drivers].shift(driver_lag)
        part.columns = [f"{column}_lag{driver_lag}" for column in drivers]
        parts.append(part)
    return pd.concat(parts, axis="columns")


def continuous_training_period(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> pd.DataFrame:
    """Select training data and insert any absent weekly or monthly periods."""
    training = data.loc[start:end]
    if training.empty:
        return training

    if frequency == "weekly":
        expected = pd.date_range(training.index.min(), training.index.max(), freq="7D")
    else:
        expected = pd.period_range(
            training.index.min().to_period("M"),
            training.index.max().to_period("M"),
            freq="M",
        ).to_timestamp("M")
    return training.reindex(expected)


def checked_driver_data(
    data: pd.DataFrame,
    training_index: pd.DatetimeIndex,
    drivers: list[str],
    lag: object,
    path: Path,
) -> pd.DataFrame:
    """Build lagged drivers and reject missing values in the training period."""
    X = apply_driver_lag(data, drivers, lag).reindex(training_index)
    X = X.apply(pd.to_numeric, errors="coerce")
    missing = X.isna().sum()
    missing = missing.loc[missing.gt(0)]
    if not missing.empty:
        details = ", ".join(f"{column}: {count}" for column, count in missing.items())
        raise ValueError(f"{path}: missing driver values ({details})")
    return X


def analyze_dataset(
    row: pd.Series,
    skipped: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Fit the models requested by one eligible configuration row."""
    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    kind = _driver_kind(row["driver_kind"])
    path = _dataset_path(row["dataset"], frequency, discretization)
    LOGGER.info("Loading dataset %s", path)
    frame = load_training_file(path)

    category = row.get("category", ALL_CATEGORIES)
    if pd.isna(category) or not str(category).strip():
        category = ALL_CATEGORIES
    data = prepare_timeseries(frame, frequency, str(category))

    start = pd.to_datetime(row["train_start"])
    end = pd.to_datetime(row["train_end"])
    training = continuous_training_period(data, start, end, frequency)
    LOGGER.info(
        "Prepared %s observations from %s to %s; driver mode: %s",
        len(training),
        start.date(),
        end.date(),
        kind,
    )
    y = (
        pd.to_numeric(training[CASE_COLUMN], errors="coerce")
        .fillna(0)
        .rename(CASE_COLUMN)
    )
    case_threshold = MIN_CASES_PER_TIMESTEP[frequency]
    below_threshold = y.le(case_threshold)
    if y.empty or below_threshold.any():
        minimum_cases = y.min() if not y.empty else None
        skipped_record = {
            "dataset": str(row["dataset"]),
            "aggregation": frequency,
            "discretization": discretization,
            "category": str(category),
            "driver_kind": kind,
            "lag": str(row["lag"]),
            "path_to_training_file": str(path),
            "date_start_training": start.date().isoformat(),
            "date_end_training": end.date().isoformat(),
            "date_end_testing": (
                "" if pd.isna(row.get("test_end")) else str(row.get("test_end"))
            ),
            "minimum_cases": minimum_cases,
            "case_threshold": case_threshold,
            "timesteps_at_or_below_threshold": int(below_threshold.sum()),
            "reason": (
                "no observations in the training period"
                if y.empty
                else f"not all training timesteps have more than {case_threshold} cases"
            ),
        }
        if skipped is not None:
            skipped.append(skipped_record)
        LOGGER.warning(
            "Skipping %s: %s",
            path,
            skipped_record["reason"],
        )
        return []

    drivers = select_drivers(data, kind)
    if kind in DRIVERS_BY_KIND and not drivers:
        raise ValueError(f"No listed {kind} drivers found in {path}")
    selected_driver = row.get(SELECTED_DRIVER_COLUMN)
    if selected_driver is None or pd.isna(selected_driver):
        driver_sets = [[driver] for driver in drivers] if drivers else [[]]
    elif str(selected_driver) == "":
        driver_sets = [[]]
    elif str(selected_driver) not in drivers:
        raise ValueError(f"Driver {selected_driver!r} is unavailable in {path}")
    else:
        driver_sets = [[str(selected_driver)]]

    from pmdarima import ARIMA, auto_arima as pmd_auto_arima

    seasonal_period = SEASONAL_PERIODS[frequency]
    test_end = row.get("test_end")
    results: list[dict[str, object]] = []

    for model_drivers in driver_sets:
        X = None
        if model_drivers:
            X = checked_driver_data(
                data,
                y.index,
                model_drivers,
                row["lag"],
                path,
            )

        driver_label = model_drivers[0] if model_drivers else kind
        LOGGER.info(
            "Fitting %s model for %s with lag %s (%s observations)",
            "predefined" if kind == "forced" else "auto_arima",
            driver_label,
            row["lag"],
            len(y),
        )
        if kind == "forced":
            model = ARIMA(
                order=FORCED_ORDER,
                seasonal_order=(*FORCED_SEASONAL_ORDER, seasonal_period),
                with_intercept=FORCED_INTERCEPT,
                suppress_warnings=True,
            ).fit(y)
        else:
            model = pmd_auto_arima(
                y,
                X=X,
                seasonal=True,
                m=seasonal_period,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
        p, d, q = model.order
        P, D, Q, m = model.seasonal_order
        LOGGER.info(
            "Selected order %s, seasonal order %s, intercept=%s for %s",
            model.order,
            model.seasonal_order,
            model.with_intercept,
            driver_label,
        )

        result = {
                "aggregation": frequency,
                "discretization": discretization,
                "category": str(category),
                "driver_kind": kind,
                "drivers_used": (
                    "forced"
                    if kind == "forced"
                    else model_drivers[0] if model_drivers else "none"
                ),
                "lag": str(row["lag"]),
                "path_to_training_file": str(path),
                "date_start_training": start.date().isoformat(),
                "date_end_training": end.date().isoformat(),
                "date_end_testing": "" if pd.isna(test_end) else str(test_end),
                "pdq": str((p, d, q)),
                "PDQ": str((P, D, Q)),
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "seasonal_period": m,
                "intercept": bool(model.with_intercept),
            }
        if MODEL_NUMBER_COLUMN in row:
            result[MODEL_NUMBER_COLUMN] = int(row[MODEL_NUMBER_COLUMN])
        results.append(result)
    return results


def _expanded_model_rows(
    configuration: pd.DataFrame,
) -> tuple[list[pd.Series], list[dict[str, object]]]:
    """Expand configuration rows into one independently runnable row per model."""
    model_rows: list[pd.Series] = []
    skipped_records: list[dict[str, object]] = []
    model_number = 0

    for _, row in configuration.iterrows():
        frequency = _frequency(row["aggregation"])
        discretization = _discretization(row["discretization"])
        kind = _driver_kind(row["driver_kind"])
        path = _dataset_path(row["dataset"], frequency, discretization)
        frame = load_training_file(path)
        category = row.get("category", ALL_CATEGORIES)
        if pd.isna(category) or not str(category).strip():
            category = ALL_CATEGORIES
        data = prepare_timeseries(frame, frequency, str(category))
        start = pd.to_datetime(row["train_start"])
        end = pd.to_datetime(row["train_end"])
        training = continuous_training_period(data, start, end, frequency)
        y = pd.to_numeric(training[CASE_COLUMN], errors="coerce").fillna(0)
        case_threshold = MIN_CASES_PER_TIMESTEP[frequency]
        below_threshold = y.le(case_threshold)

        if y.empty or below_threshold.any():
            skipped_records.append(
                {
                    "dataset": str(row["dataset"]),
                    "aggregation": frequency,
                    "discretization": discretization,
                    "category": str(category),
                    "driver_kind": kind,
                    "lag": str(row["lag"]),
                    "path_to_training_file": str(path),
                    "date_start_training": start.date().isoformat(),
                    "date_end_training": end.date().isoformat(),
                    "date_end_testing": (
                        "" if pd.isna(row.get("test_end")) else str(row.get("test_end"))
                    ),
                    "minimum_cases": y.min() if not y.empty else None,
                    "case_threshold": case_threshold,
                    "timesteps_at_or_below_threshold": int(below_threshold.sum()),
                    "reason": (
                        "no observations in the training period"
                        if y.empty
                        else "not all training timesteps have more than "
                        f"{case_threshold} cases"
                    ),
                }
            )
            continue

        drivers = select_drivers(data, kind)
        if kind in DRIVERS_BY_KIND and not drivers:
            raise ValueError(f"No listed {kind} drivers found in {path}")
        selected_drivers: list[str | None] = drivers if drivers else [None]
        for driver in selected_drivers:
            model_number += 1
            model_row = row.copy()
            model_row[SELECTED_DRIVER_COLUMN] = "" if driver is None else driver
            model_row[MODEL_NUMBER_COLUMN] = model_number
            model_rows.append(model_row)

    return model_rows, skipped_records


def _model_slug(record: dict[str, object]) -> str:
    """Build the stable identifier shared by one-row order artifacts."""
    dataset = Path(str(record["path_to_training_file"])).stem
    driver = str(record["drivers_used"])
    label = re.sub(r"[^a-zA-Z0-9]+", "_", f"{dataset}_{driver}").strip("_").lower()
    return f"model_{int(record[MODEL_NUMBER_COLUMN]):03d}_{label}"


def _save_model_order(record: dict[str, object], timestamp: str) -> Path:
    """Immediately save one completed order search to its own CSV."""
    output = solve_path(
        Path("data")
        / str(record["aggregation"])
        / "orders"
        / timestamp
        / _model_slug(record)
        / f"orders_{timestamp}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(output, index=False)
    LOGGER.info("Saved completed model order to %s", output)
    return output


def _analyze_and_save_model(task: tuple[pd.Series, str]) -> Path:
    """Select and persist exactly one model without returning fitted state."""
    row, timestamp = task
    results = analyze_dataset(row)
    if len(results) != 1:
        raise RuntimeError(f"Expected exactly one model result, got {len(results)}")
    return _save_model_order(results[0], timestamp)


def iter_run(configuration_path: str | Path, jobs: int = 1):
    """Yield durable per-model order files as soon as their searches finish."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    configuration = read_configuration(configuration_path)
    LOGGER.info(
        "Loaded %s analysis specification(s) from %s",
        len(configuration),
        solve_path(configuration_path),
    )
    model_rows, skipped_records = _expanded_model_rows(configuration)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    tasks = [(row, timestamp) for row in model_rows]

    if jobs == 1:
        for task in tasks:
            yield _analyze_and_save_model(task)
    else:
        LOGGER.info("Selecting orders with %s worker processes", jobs)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_analyze_and_save_model, task) for task in tasks]
            for future in as_completed(futures):
                # Each worker has already written its one-row order file.
                yield future.result()

    yield from save_skipped(pd.DataFrame(skipped_records))


def save_results(results: pd.DataFrame) -> list[Path]:
    """Save one order table below each frequency's ``orders`` directory."""
    outputs: list[Path] = []
    if results.empty:
        return outputs

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    for frequency, rows in results.groupby("aggregation", sort=False):
        output = solve_path(
            Path("data") / frequency / "orders" / f"orders_{timestamp}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output, index=False)
        LOGGER.info("Saved %s model result(s) to %s", len(rows), output)
        outputs.append(output)
    return outputs


def save_skipped(skipped: pd.DataFrame) -> list[Path]:
    """Save skipped dataset details beside each frequency's order tables."""
    outputs: list[Path] = []
    if skipped.empty:
        return outputs

    for frequency, rows in skipped.groupby("aggregation", sort=False):
        output = solve_path(Path("data") / frequency / "orders" / "skipped.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output, index=False)
        LOGGER.info("Saved %s skipped dataset(s) to %s", len(rows), output)
        outputs.append(output)
    return outputs


def _analyze_dataset_isolated(
    row: pd.Series,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Analyze one row without sharing mutable state between worker processes."""
    skipped: list[dict[str, object]] = []
    return analyze_dataset(row, skipped), skipped


def run(configuration_path: str | Path, jobs: int = 1) -> list[Path]:
    """Analyze eligible rows and save orders plus details of skipped datasets."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    configuration = read_configuration(configuration_path)
    LOGGER.info(
        "Loaded %s analysis specification(s) from %s",
        len(configuration),
        solve_path(configuration_path),
    )
    result_records: list[dict[str, object]] = []
    skipped_records: list[dict[str, object]] = []
    rows = [row for _, row in configuration.iterrows()]
    if jobs == 1:
        analyses = (_analyze_dataset_isolated(row) for row in rows)
        for results, skipped in analyses:
            result_records.extend(results)
            skipped_records.extend(skipped)
    else:
        LOGGER.info("Analyzing configuration rows with %s worker processes", jobs)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for results, skipped in executor.map(_analyze_dataset_isolated, rows):
                result_records.extend(results)
                skipped_records.extend(skipped)

    outputs = save_results(pd.DataFrame(result_records))
    outputs.extend(save_skipped(pd.DataFrame(skipped_records)))
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", help="CSV file describing datasets to analyze")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="configuration rows to process concurrently (default: %(default)s)",
    )
    args = parser.parse_args()
    for output in run(args.configuration, jobs=args.jobs):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
