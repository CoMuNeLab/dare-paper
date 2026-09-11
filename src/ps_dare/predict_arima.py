"""Run rolling one-step-ahead forecasts from an ``orders.csv`` file.

The order table is the complete forecasting specification: it identifies the
dataset, training and testing limits, exogenous driver, lag, ARIMA orders, and
intercept choice.  Each model is fitted once on its configured training
window.  After every one-step-ahead prediction, ``pmdarima.ARIMA.update`` adds
the newly observed access count before the next prediction is made.

From the repository root, run::

    python -m ps_dare.predict_arima data/weekly/orders/orders_20240101_120000.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .auto_arima import (
    CASE_COLUMN,
    _discretization,
    _frequency,
    apply_driver_lag,
    continuous_training_period,
    load_training_file,
)
from .explorer import ALL_CATEGORIES, prepare_timeseries
from .paths import solve_path

LOGGER = logging.getLogger(__name__)

CONFIDENCE_ALPHA = 0.05

REQUIRED_ORDER_COLUMNS = {
    "aggregation",
    "discretization",
    "drivers_used",
    "lag",
    "path_to_training_file",
    "date_start_training",
    "date_end_training",
    "date_end_testing",
    "p",
    "d",
    "q",
    "P",
    "D",
    "Q",
    "seasonal_period",
    "intercept",
}

ORDERS_FILENAME_PATTERN = re.compile(r"orders_(\d{8}_\d{6})\.csv")


def read_orders(path: str | Path) -> pd.DataFrame:
    """Load and validate a model-order table produced by ``auto_arima``."""
    orders_path = solve_path(path)
    orders = pd.read_csv(orders_path)
    orders = orders.loc[:, ~orders.columns.astype(str).str.startswith("Unnamed:")]
    missing = sorted(REQUIRED_ORDER_COLUMNS.difference(orders.columns))
    if missing:
        raise ValueError(f"Missing order columns: {', '.join(missing)}")
    if orders.empty:
        raise ValueError(f"No model specifications found in {orders_path}")
    return orders


def orders_timestamp(path: str | Path) -> str:
    """Extract the run timestamp written by ``auto_arima`` from a filename."""
    match = ORDERS_FILENAME_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(
            "Orders filename must have the form "
            "orders_YYYYMMDD_HHMMSS.csv so predictions can reuse its timestamp"
        )
    return match.group(1)


def _integer(value: object, name: str) -> int:
    """Parse one integer-valued order-table field."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if float(value) != number:
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def _boolean(value: object) -> bool:
    """Parse an explicit CSV boolean without treating non-empty text as true."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid intercept value: {value!r}")


def _dataset_path(row: pd.Series, frequency: str, discretization: str) -> Path:
    """Resolve a saved dataset path, with a repository-relative fallback."""
    configured = Path(str(row["path_to_training_file"]).strip()).expanduser()
    candidate = solve_path(configured)
    if candidate.is_file():
        return candidate

    fallback = solve_path(
        Path("data") / frequency / "aggregated" / discretization / configured.name
    )
    if fallback.is_file():
        LOGGER.warning(
            "Using %s because the saved dataset path is unavailable", fallback
        )
        return fallback
    raise FileNotFoundError(f"Training dataset not found: {configured}")


def _driver_name(row: pd.Series) -> str | None:
    """Return the single exogenous driver saved for this model, if any."""
    value = row["drivers_used"]
    if pd.isna(value):
        return None
    driver = str(value).strip()
    if driver.lower() in {"", "none", "forced"}:
        return None
    return driver


def _category(row: pd.Series) -> str:
    """Return the saved diagnosis category or the legacy all-category default."""
    value = row.get("category", ALL_CATEGORIES)
    if pd.isna(value) or not str(value).strip():
        return ALL_CATEGORIES
    return str(value)


def _numeric_features(
    data: pd.DataFrame,
    index: pd.DatetimeIndex,
    driver: str,
    lag: object,
    path: Path,
) -> pd.DataFrame:
    """Build and validate the exact lagged exogenous inputs for an index."""
    if driver not in data:
        raise ValueError(f"{path}: driver column not found: {driver}")
    features = apply_driver_lag(data, [driver], lag).reindex(index)
    features = features.apply(pd.to_numeric, errors="coerce")
    missing = features.isna().sum()
    missing = missing.loc[missing.gt(0)]
    if not missing.empty:
        details = ", ".join(f"{column}: {count}" for column, count in missing.items())
        raise ValueError(f"{path}: missing driver values ({details})")
    return features


def _testing_features(
    data: pd.DataFrame,
    testing: pd.DataFrame,
    driver: str,
    lag: object,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stop testing before the first period with an unavailable driver."""
    if driver not in data:
        raise ValueError(f"{path}: driver column not found: {driver}")

    features = apply_driver_lag(data, [driver], lag).reindex(testing.index)
    features = features.apply(pd.to_numeric, errors="coerce")
    incomplete = features.isna().any(axis="columns")
    if not incomplete.any():
        return testing, features

    first_missing = incomplete.index[incomplete][0]
    available = testing.index < first_missing
    omitted = len(testing) - int(available.sum())
    LOGGER.warning(
        "%s: ending testing before %s because %s is unavailable; "
        "marking %s period(s) as after_testing",
        path,
        first_missing.date().isoformat(),
        driver,
        omitted,
    )
    return testing.loc[available], features.loc[available]


def _testing_period(
    data: pd.DataFrame,
    training_end: pd.Timestamp,
    testing_end: pd.Timestamp,
    frequency: str,
) -> pd.DataFrame:
    """Select a continuous observed period after training through testing end."""
    available = data.loc[(data.index > training_end) & (data.index <= testing_end)]
    if available.empty:
        return available
    return continuous_training_period(
        data,
        pd.Timestamp(available.index.min()),
        pd.Timestamp(available.index.max()),
        frequency,
    )


def _model_id(row_number: int, path: Path, driver: str | None) -> str:
    """Build a stable, CSV-friendly identifier for one order-table row."""
    label = f"{path.stem}_{driver or 'none'}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    return f"model_{row_number + 1:03d}_{slug}"


def _training_fit_statistics(model: Any) -> tuple[float, str]:
    """Extract AIC and named coefficients before rolling updates modify the fit."""
    try:
        aic = float(model.aic())
    except (AttributeError, TypeError, ValueError):
        aic = float("nan")

    result = getattr(model, "arima_res_", None)
    names = list(getattr(result, "param_names", []))
    try:
        values = list(model.params())
    except (AttributeError, TypeError, ValueError):
        values = list(getattr(result, "params", []))

    if not names and values:
        names = [f"parameter_{number + 1}" for number in range(len(values))]
    parameters = {
        str(name): float(value) for name, value in zip(names, values, strict=False)
    }
    return aic, json.dumps(parameters, sort_keys=True)


def _save_fitted_model(model: Any, output: Path) -> None:
    """Serialize the initial training fit before any test-period updates."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Saved fitted model to %s", output)


def _model_output_path(
    row: pd.Series, row_number: int, timestamp: str
) -> Path:
    """Return a unique, dataset-specific path for one fitted model."""
    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    dataset = _dataset_path(row, frequency, discretization)
    model_id = _model_id(row_number, dataset, _driver_name(row))
    return solve_path(
        Path("data")
        / frequency
        / "models"
        / dataset.stem
        / f"{model_id}_{timestamp}.pkl"
    )


def _prediction_output_path(
    row: pd.Series, row_number: int, timestamp: str
) -> Path:
    """Return a unique per-model path for one completed test simulation."""
    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    dataset = _dataset_path(row, frequency, discretization)
    model_id = _model_id(row_number, dataset, _driver_name(row))
    return solve_path(
        Path("data")
        / frequency
        / "predictions"
        / dataset.stem
        / f"{model_id}_{timestamp}.csv"
    )


def forecast_order(
    row: pd.Series,
    row_number: int = 0,
    model_class: Any | None = None,
    model_output_path: Path | None = None,
) -> pd.DataFrame:
    """Fit one saved model and return its full input series and predictions."""
    if model_class is None:
        from pmdarima import ARIMA

        model_class = ARIMA

    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    path = _dataset_path(row, frequency, discretization)
    frame = load_training_file(path)
    category = _category(row)
    data = prepare_timeseries(frame, frequency, category)
    driver = _driver_name(row)

    training_start = pd.to_datetime(row["date_start_training"])
    training_end = pd.to_datetime(row["date_end_training"])
    testing_end = pd.to_datetime(row["date_end_testing"])
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

    y_training = pd.to_numeric(training[CASE_COLUMN], errors="coerce").fillna(0)
    X_training = None
    X_testing = None
    if driver is not None:
        X_training = _numeric_features(
            data, y_training.index, driver, row["lag"], path
        )
        testing, X_testing = _testing_features(
            data, testing, driver, row["lag"], path
        )

    y_testing = pd.to_numeric(testing[CASE_COLUMN], errors="coerce")
    missing_testing = y_testing.index[y_testing.isna()]
    if len(missing_testing):
        dates = ", ".join(date.date().isoformat() for date in missing_testing[:5])
        suffix = "..." if len(missing_testing) > 5 else ""
        raise ValueError(
            f"{path}: missing observed accesses in testing ({dates}{suffix})"
        )

    order = tuple(_integer(row[name], name) for name in ("p", "d", "q"))
    seasonal_order = tuple(
        _integer(row[name], name) for name in ("P", "D", "Q", "seasonal_period")
    )
    model = model_class(
        order=order,
        seasonal_order=seasonal_order,
        with_intercept=_boolean(row["intercept"]),
        suppress_warnings=True,
    ).fit(y_training, X=X_training)
    training_aic, fitted_parameters = _training_fit_statistics(model)
    if model_output_path is not None:
        _save_fitted_model(model, model_output_path)

    LOGGER.info(
        "Fitted %s on %s observations; forecasting %s one-step periods",
        _model_id(row_number, path, driver),
        len(y_training),
        len(y_testing),
    )
    predictions = pd.Series(index=data.index, dtype="float64", name="prediction")
    prediction_lower_bound = pd.Series(
        index=data.index, dtype="float64", name="prediction_lower_bound"
    )
    prediction_upper_bound = pd.Series(
        index=data.index, dtype="float64", name="prediction_upper_bound"
    )
    for date, actual in y_testing.items():
        X_step = None if X_testing is None else X_testing.loc[[date]]
        predicted, confidence_interval = model.predict(
            n_periods=1,
            X=X_step,
            return_conf_int=True,
            alpha=CONFIDENCE_ALPHA,
        )
        predictions.loc[date] = float(pd.Series(predicted).iloc[0])
        interval = pd.DataFrame(confidence_interval)
        if interval.shape != (1, 2):
            raise ValueError(
                "Expected one lower and one upper confidence bound for "
                f"{date.date().isoformat()}, got shape {interval.shape}"
            )
        prediction_lower_bound.loc[date] = float(interval.iloc[0, 0])
        prediction_upper_bound.loc[date] = float(interval.iloc[0, 1])
        model.update([float(actual)], X=X_step)

    phase = pd.Series("before_training", index=data.index, dtype="object")
    phase.loc[
        (data.index >= training_start) & (data.index <= training_end)
    ] = "training"
    if testing.empty:
        effective_testing_end = training_end
    else:
        effective_testing_end = pd.Timestamp(testing.index.max())
        phase.loc[testing.index] = "testing"
    phase.loc[data.index > effective_testing_end] = "after_testing"

    driver_values = pd.Series(index=data.index, dtype="float64")
    if driver is not None:
        driver_values = pd.to_numeric(data[driver], errors="coerce")

    result = pd.DataFrame(
        {
            "model_id": _model_id(row_number, path, driver),
            "aggregation": frequency,
            "discretization": discretization,
            "path_to_training_file": str(path),
            "category": category,
            "date_start_training": training_start.date().isoformat(),
            "date_end_training": training_end.date().isoformat(),
            "date_end_testing": testing_end.date().isoformat(),
            "date": data.index,
            "phase": phase,
            CASE_COLUMN: pd.to_numeric(data[CASE_COLUMN], errors="coerce"),
            "drivers_used": "none" if driver is None else driver,
            "driver_value": driver_values,
            "lag": str(row["lag"]),
            "prediction_horizon": 1,
            "prediction": predictions,
            "prediction_lower_bound": prediction_lower_bound,
            "prediction_upper_bound": prediction_upper_bound,
            "p": order[0],
            "d": order[1],
            "q": order[2],
            "P": seasonal_order[0],
            "D": seasonal_order[1],
            "Q": seasonal_order[2],
            "seasonal_period": seasonal_order[3],
            "intercept": _boolean(row["intercept"]),
            "training_aic": training_aic,
            "fitted_parameters": fitted_parameters,
            "model_file": "" if model_output_path is None else str(model_output_path),
        }
    )
    result.index = pd.RangeIndex(len(result))
    return result


def save_predictions(predictions: pd.DataFrame, timestamp: str) -> list[Path]:
    """Save one timestamped long-form CSV per distinct training file."""
    outputs: list[Path] = []
    grouped = predictions.groupby(
        ["aggregation", "path_to_training_file"], sort=False
    )
    for (frequency, training_file), rows in grouped:
        output = solve_path(
            Path("data")
            / str(frequency)
            / "predictions"
            / f"{Path(str(training_file)).stem}_{timestamp}.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(output, index=False, date_format="%Y-%m-%d")
        LOGGER.info("Saved %s rows to %s", len(rows), output)
        outputs.append(output)
    return outputs


def _save_model_prediction(prediction: pd.DataFrame, output: Path) -> Path:
    """Persist one completed model simulation immediately."""
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output, index=False, date_format="%Y-%m-%d")
    LOGGER.info("Saved completed model simulation to %s", output)
    return output


def _forecast_order_isolated(
    task: tuple[pd.Series, int, Path, Path],
) -> Path:
    """Fit, persist, test, and immediately save one order-table row."""
    row, row_number, model_output_path, prediction_output_path = task
    prediction = forecast_order(
        row, row_number, model_output_path=model_output_path
    )
    return _save_model_prediction(prediction, prediction_output_path)


def run(orders_path: str | Path, jobs: int = 1) -> list[Path]:
    """Forecast every order row and immediately save one file per model."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    resolved_orders = solve_path(orders_path)
    timestamp = orders_timestamp(resolved_orders)
    orders = read_orders(resolved_orders)
    tasks = []
    for fallback_number, (_, row) in enumerate(orders.iterrows()):
        saved_number = row.get("model_number")
        row_number = (
            fallback_number
            if saved_number is None or pd.isna(saved_number)
            else _integer(saved_number, "model_number") - 1
        )
        tasks.append(
            (
                row,
                row_number,
                _model_output_path(row, row_number, timestamp),
                _prediction_output_path(row, row_number, timestamp),
            )
        )
    if jobs == 1:
        return [_forecast_order_isolated(task) for task in tasks]

    LOGGER.info("Calibrating and testing models with %s worker processes", jobs)
    outputs: list[Path] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_forecast_order_isolated, task) for task in tasks]
        for future in as_completed(futures):
            # The worker writes before returning, so completed DataFrames never
            # accumulate in the parent process while other simulations finish.
            outputs.append(future.result())
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("orders", help="orders.csv produced by ps_dare.auto_arima")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="models to calibrate and test concurrently (default: %(default)s)",
    )
    args = parser.parse_args()
    for output in run(args.orders, jobs=args.jobs):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
