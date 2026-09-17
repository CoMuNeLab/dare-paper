"""Score a persistence baseline with rolling one-step-ahead predictions.

The baseline predicts that the next period's case count equals the most
recently observed count.  It has no fitted likelihood or uncertainty model,
so AIC and the prediction bounds are deliberately left missing.

From the repository root, run::

    python -m ps_dare.predict_baseline configuration.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .auto_arima import (
    CASE_COLUMN,
    _dataset_path,
    _discretization,
    _frequency,
    continuous_training_period,
    load_training_file,
    read_configuration,
)
from .explorer import ALL_CATEGORIES, prepare_timeseries
from .paths import solve_path
from .predict_arima import (
    CONFIDENCE_ALPHA,
    _metrics_output_path,
    _testing_period,
    model_metrics,
)

LOGGER = logging.getLogger(__name__)
METHOD = "baseline"


def _category(row: pd.Series) -> str:
    value = row.get("category", ALL_CATEGORIES)
    if pd.isna(value) or not str(value).strip():
        return ALL_CATEGORIES
    return str(value)


def forecast_baseline(row: pd.Series) -> pd.DataFrame:
    """Return persistence forecasts for one configuration row."""
    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    path = _dataset_path(row["dataset"], frequency, discretization)
    category = _category(row)
    data = prepare_timeseries(load_training_file(path), frequency, category)

    training_start = pd.to_datetime(row["train_start"])
    training_end = pd.to_datetime(row["train_end"])
    testing_end = pd.to_datetime(row.get("test_end"))
    if pd.isna(training_start) or pd.isna(training_end) or pd.isna(testing_end):
        raise ValueError("Training and testing limits must all be valid dates")
    if training_start > training_end:
        raise ValueError("Training start must not be after training end")
    if testing_end <= training_end:
        raise ValueError("Testing end must be after training end")

    training = continuous_training_period(
        data, training_start, training_end, frequency
    )
    if training.empty:
        raise ValueError(f"{path}: no observations in the configured training period")
    testing = _testing_period(data, training_end, testing_end, frequency)
    if testing.empty:
        raise ValueError(f"{path}: no observations in the configured testing period")

    observed = pd.to_numeric(data[CASE_COLUMN], errors="coerce")
    y_testing = observed.reindex(testing.index)
    missing_testing = y_testing.index[y_testing.isna()]
    if len(missing_testing):
        dates = ", ".join(date.date().isoformat() for date in missing_testing[:5])
        suffix = "..." if len(missing_testing) > 5 else ""
        raise ValueError(
            f"{path}: missing observed accesses in testing ({dates}{suffix})"
        )

    predictions = observed.shift(1).where(observed.index.isin(testing.index))
    missing_predictions = testing.index[predictions.reindex(testing.index).isna()]
    if len(missing_predictions):
        raise ValueError(
            f"{path}: the first testing period has no preceding observed count"
        )

    phase = pd.Series("before_training", index=data.index, dtype="object")
    phase.loc[
        (data.index >= training_start) & (data.index <= training_end)
    ] = "training"
    phase.loc[testing.index] = "testing"
    phase.loc[data.index > pd.Timestamp(testing.index.max())] = "after_testing"

    missing_values = pd.Series(index=data.index, dtype="float64")
    result = pd.DataFrame(
        {
            "model_id": METHOD,
            "model_number": row.get("model_number", pd.NA),
            "method": METHOD,
            "aggregation": frequency,
            "discretization": discretization,
            "path_to_training_file": str(path),
            "category": category,
            "driver_kind": METHOD,
            "date_start_training": training_start.date().isoformat(),
            "date_end_training": training_end.date().isoformat(),
            "date_end_testing": testing_end.date().isoformat(),
            "date": data.index,
            "phase": phase,
            CASE_COLUMN: observed,
            "drivers_used": METHOD,
            "driver_value": missing_values,
            "lag": pd.NA,
            "prediction_horizon": 1,
            "confidence_alpha": CONFIDENCE_ALPHA,
            "update_mode": METHOD,
            "update_maxiter": pd.NA,
            "prediction": predictions,
            "prediction_lower_bound": missing_values,
            "prediction_upper_bound": missing_values,
            "p": pd.NA,
            "d": pd.NA,
            "q": pd.NA,
            "P": pd.NA,
            "D": pd.NA,
            "Q": pd.NA,
            "seasonal_period": pd.NA,
            "intercept": pd.NA,
            "training_aic": float("nan"),
            "fitted_parameters": "",
            "model_file": "",
        }
    )
    result.index = pd.RangeIndex(len(result))
    return result


def _baseline_key(row: pd.Series) -> tuple[str, ...]:
    """Identify duplicate baseline specifications across driver variants."""
    return tuple(
        str(row.get(column, ""))
        for column in (
            "dataset",
            "aggregation",
            "discretization",
            "category",
            "train_start",
            "train_end",
            "test_end",
        )
    )


def _output_path(predictions: pd.DataFrame, timestamp: str) -> Path:
    frequency = str(predictions["aggregation"].iloc[0])
    dataset = Path(str(predictions["path_to_training_file"].iloc[0])).stem
    model_number = predictions["model_number"].iloc[0]
    model_suffix = (
        ""
        if pd.isna(model_number)
        else f"_model_{int(model_number):03d}"
    )
    return solve_path(
        Path("data")
        / frequency
        / "predictions"
        / dataset
        / f"{dataset}_baseline_{frequency}{model_suffix}_{timestamp}.csv"
    )


def save_baseline(row: pd.Series, timestamp: str) -> list[Path]:
    """Forecast, score, and save one configured baseline."""
    predictions = forecast_baseline(row)
    prediction_output = _output_path(predictions, timestamp)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_output, index=False, date_format="%Y-%m-%d")
    metrics_output = _metrics_output_path(prediction_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model_metrics(predictions).to_csv(metrics_output, index=False)
    LOGGER.info("Saved baseline predictions to %s", prediction_output)
    LOGGER.info("Saved baseline metrics to %s", metrics_output)
    return [prediction_output, metrics_output]


def run(
    configurations: Iterable[str | Path], timestamp: str | None = None
) -> list[Path]:
    """Save one prediction and metrics artifact per unique baseline setup."""
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[pd.Series] = []
    seen: set[tuple[str, ...]] = set()
    for configuration in configurations:
        for _, row in read_configuration(configuration).iterrows():
            key = _baseline_key(row)
            if key not in seen:
                seen.add(key)
                rows.append(row)

    outputs: list[Path] = []
    for row in rows:
        outputs.extend(save_baseline(row, run_timestamp))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configurations",
        nargs="+",
        help="one or more ARIMA configuration CSVs defining evaluation windows",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for output in run(args.configurations):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
