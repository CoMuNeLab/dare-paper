"""Combine case counts with weather and pollution drivers."""

import warnings
from itertools import product
from pathlib import Path

import pandas as pd

from .paths import solve_path

FREQUENCIES = ("monthly", "weekly")
HOSPITALS = ("giustiniani", "osa", "pediatrico")

# (output diagnosis name, categorized input)
CASE_VARIANTS = {
    "general": ("general", False),
    "general_ili": ("ili", False),
    "general_hw": ("hw", False),
    "ili": ("ili", True),
    "hw": ("hw", True),
}

DRIVER_FILES = {
    "continuous": (
        ("humidex_wet_bulb_padova_era5.dat", 0),
        ("hot_index_padova.dat", None),
        ("cold_index_padova.dat", None),
    ),
    "discrete": (
        ("humidex_wet_bulb_padova_era5.dat", 0),
        ("heatwave_frequency_padova.dat", None),
        ("cold_spell_days_padova.dat", None),
        ("tropical_night_padova.dat", None),
        ("frost_days_padova.dat", 0),
    ),
}


def warn_about_missing_periods(cases: pd.DataFrame, frequency: str, source: Path) -> None:
    """Warn if case dates are invalid or do not form a continuous timeline."""
    column = "month" if frequency == "monthly" else "week"
    dates = pd.to_datetime(cases[column], errors="coerce")
    invalid = int(dates.isna().sum())
    if invalid:
        warnings.warn(f"{source}: {invalid} invalid {column!r} value(s).", stacklevel=2)

    dates = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    if len(dates) < 2:
        return

    if frequency == "monthly":
        dates = dates.to_period("M").to_timestamp("M")
        expected = pd.date_range(dates.min(), dates.max(), freq="ME")
    else:
        expected = pd.date_range(dates.min(), dates.max(), freq="7D")

    missing = expected.difference(dates)
    if len(missing):
        preview = ", ".join(value.strftime("%Y-%m-%d") for value in missing[:10])
        suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        warnings.warn(
            f"{source}: missing {len(missing)} {frequency} period(s): {preview}{suffix}.",
            stacklevel=2,
        )


def merge_driver(
    cases: pd.DataFrame,
    driver: pd.DataFrame,
    frequency: str,
    source: str = "driver dataset",
) -> pd.DataFrame:
    """Outer-join one driver so dates from both cases and the driver are preserved."""
    if frequency not in FREQUENCIES:
        raise ValueError(f"Unsupported frequency: {frequency!r}")

    cases = cases.copy()
    driver = driver.copy()
    driver["valid_time"] = pd.to_datetime(driver["valid_time"], errors="coerce")
    date_column = "month" if frequency == "monthly" else "week"
    case_dates = pd.to_datetime(cases[date_column], errors="coerce")
    if frequency == "monthly":
        case_dates += pd.offsets.MonthEnd(0)
    else:
        driver["valid_time"] += pd.Timedelta(days=1)

    # After the first outer join, driver-only rows have no case date but do
    # have a valid_time.  Preserve that timestamp during subsequent merges.
    if "valid_time" in cases:
        existing_dates = pd.to_datetime(cases["valid_time"], errors="coerce")
        cases["valid_time"] = existing_dates.combine_first(case_dates)
    else:
        cases["valid_time"] = case_dates

    invalid_cases = int(cases["valid_time"].isna().sum())
    if invalid_cases:
        warnings.warn(
            f"case dataset: ignoring {invalid_cases} row(s) with invalid dates.",
            stacklevel=2,
        )
        cases = cases.dropna(subset=["valid_time"])

    invalid = int(driver["valid_time"].isna().sum())
    if invalid:
        warnings.warn(f"{source}: ignoring {invalid} invalid date(s).", stacklevel=2)
        driver = driver.dropna(subset=["valid_time"])

    duplicates = driver["valid_time"].duplicated(keep=False)
    if duplicates.any():
        count = int(driver.loc[duplicates, "valid_time"].nunique())
        warnings.warn(
            f"{source}: {count} duplicate date(s); keeping the first.", stacklevel=2
        )
        driver = driver.drop_duplicates("valid_time")

    marker = "__driver_match__"
    merged = cases.merge(
        driver,
        on="valid_time",
        how="outer",
        sort=False,
        validate="many_to_one",
        indicator=marker,
    )
    unmatched = int(merged[marker].eq("left_only").sum())
    if unmatched:
        warnings.warn(
            f"{source}: no match for {unmatched}/{len(cases)} case row(s).",
            stacklevel=2,
        )
    return merged.drop(columns=marker)


def _case_paths(
    variant: str, frequency: str, hospital: str, mode: str
) -> tuple[Path, Path]:
    diagnosis, categorized = CASE_VARIANTS[variant]
    category = "_cat" if categorized else ""
    source = solve_path(f"data/{frequency}/raw/{diagnosis}_{hospital}{category}.pkl")
    output = solve_path(
        f"data/{frequency}/aggregated/{mode}/{diagnosis}_{hospital}{category}"
    )
    return source, output


def _load_drivers(frequency: str, mode: str) -> list[tuple[pd.DataFrame, Path]]:
    specs = [
        (Path(f"data/{frequency}/drivers/{mode}/{name}"), index_col)
        for name, index_col in DRIVER_FILES[mode]
    ]
    specs.append((Path(f"data/{frequency}/drivers/shared/humidity_padova.dat"), None))
    if mode == "continuous":
        specs.append((Path(f"data/{frequency}/drivers/continuous/pollution.dat"), 0))
    return [(pd.read_csv(solve_path(path), index_col=index), solve_path(path)) for path, index in specs]


def _fill_internal_case_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """Set missing case counts to zero only within the observed case timeline."""
    data = data.copy()
    observed = data["n_accesses"].notna() & data["valid_time"].notna()
    if not observed.any():
        return data

    first_case = data.loc[observed, "valid_time"].min()
    last_case = data.loc[observed, "valid_time"].max()
    internal_gap = (
        data["n_accesses"].isna()
        & data["valid_time"].between(first_case, last_case)
    )
    data.loc[internal_gap, "n_accesses"] = 0
    return data


def build_dataset(variant: str, frequency: str, hospital: str, mode: str) -> Path:
    """Build one aggregated dataset and return its output path without a suffix."""
    source, output = _case_paths(variant, frequency, hospital, mode)
    cases = pd.read_pickle(source)
    warn_about_missing_periods(cases, frequency, source)
    for driver, path in _load_drivers(frequency, mode):
        cases = merge_driver(cases, driver, frequency, str(path))
    cases = _fill_internal_case_gaps(cases)

    output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_pickle(output.with_suffix(".pkl"))
    cases.to_csv(output.with_suffix(".csv"))
    return output


def build_all_datasets() -> None:
    """Build every diagnosis, frequency, hospital, and driver-mode combination."""
    combinations = product(CASE_VARIANTS, FREQUENCIES, HOSPITALS, DRIVER_FILES)
    for combination in combinations:
        output = build_dataset(*combination)
        print(f"Saved {output}.pkl and {output}.csv")
