"""Run rolling one-step-ahead forecasts from an ``orders.csv`` file.

The order table identifies the fitted model produced by ``auto_arima`` along
with the dataset, testing limits, exogenous driver, lag, and ARIMA orders.
Prediction loads that exact training fit. After every one-step-ahead
prediction, the newly observed access count is added before the next forecast.
By default this uses ``pmdarima.ARIMA.update``; optional CLI modes can append
state without refitting or limit the update's MLE iterations.

From the repository root, run::

    python -m ps_dare.predict_arima \
        --orders data/output/20240101_120000/orders/orders_20240101_120000.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import pickle
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from .paths import output_path, solve_path

LOGGER = logging.getLogger(__name__)

CONFIDENCE_ALPHA = 0.05
HOSPITAL_NAMES = ("giustiniani", "osa", "pediatrico")

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
    "model_file",
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


def _prediction_file_id(
    row_number: int,
    path: Path,
    driver: str | None,
    aggregation: str,
    lag: object,
) -> str:
    """Include aggregation and lag before the progressive model label."""
    label = f"{path.stem}_{driver or 'none'}_{aggregation}_lag_{lag}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    return f"{slug}_model_{row_number + 1:03d}"


def _training_fit_statistics(model: Any) -> tuple[float, str]:
    """Extract AIC, coefficients, and p-values before rolling updates."""
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

    try:
        pvalues = list(model.pvalues())
    except (AttributeError, TypeError, ValueError):
        pvalues = list(getattr(result, "pvalues", []))

    if not names and values:
        names = [f"parameter_{number + 1}" for number in range(len(values))]
    parameters = {
        str(name): float(value) for name, value in zip(names, values, strict=False)
    }
    named_pvalues = {
        str(name): float(pvalue)
        for name, pvalue in zip(names, pvalues, strict=False)
    }
    statistics = {"parameters": parameters, "pvalues": named_pvalues}
    return aic, json.dumps(statistics, sort_keys=True)


def _save_fitted_model(model: Any, output: Path) -> None:
    """Serialize the initial training fit before any test-period updates."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Saved fitted model to %s", output)


def _load_fitted_model(path: Path) -> Any:
    """Load the exact training fit saved by the order-selection stage."""
    if not path.is_file():
        raise FileNotFoundError(f"Fitted model not found: {path}")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    LOGGER.info("Loaded fitted Auto-ARIMA model from %s", path)
    return model


def _validate_update_options(
    state_only_updates: bool, update_maxiter: int | None
) -> None:
    """Validate the mutually exclusive rolling-update strategies."""
    if state_only_updates and update_maxiter is not None:
        raise ValueError(
            "state_only_updates and update_maxiter cannot be used together"
        )
    if update_maxiter is not None and update_maxiter < 1:
        raise ValueError("update_maxiter must be at least 1")


def _update_model(
    model: Any,
    actual: float,
    date: pd.Timestamp,
    X_step: pd.DataFrame | None,
    *,
    state_only_updates: bool,
    update_maxiter: int | None,
) -> None:
    """Append one observation, optionally avoiding or limiting MLE fitting."""
    if state_only_updates:
        results = getattr(model, "arima_res_", None)
        append = getattr(results, "append", None)
        if not callable(append):
            raise TypeError(
                "The fitted model does not support state-only append updates"
            )
        model_data = getattr(getattr(results, "model", None), "data", None)
        original_endog = getattr(model_data, "orig_endog", None)
        observation_index = pd.DatetimeIndex([date])
        if isinstance(original_endog, pd.DataFrame):
            observation = pd.DataFrame(
                [[float(actual)]],
                index=observation_index,
                columns=original_endog.columns,
            )
        elif isinstance(original_endog, pd.Series):
            observation = pd.Series(
                [float(actual)],
                index=observation_index,
                name=original_endog.name,
            )
        else:
            observation = [float(actual)]

        original_exog = getattr(model_data, "orig_exog", None)
        if X_step is None:
            appended_X = None
        elif isinstance(original_exog, pd.DataFrame):
            appended_X = X_step.copy()
            if len(appended_X.columns) != len(original_exog.columns):
                raise ValueError(
                    "State-only update exogenous columns do not match the fitted model"
                )
            appended_X.columns = original_exog.columns
        elif isinstance(original_exog, pd.Series):
            if X_step.shape[1] != 1:
                raise ValueError(
                    "State-only update expected one exogenous model column"
                )
            appended_X = X_step.iloc[:, 0].rename(original_exog.name)
        else:
            appended_X = X_step.to_numpy()

        model.arima_res_ = append(
            observation,
            exog=appended_X,
            refit=False,
        )
        if hasattr(model, "nobs_"):
            model.nobs_ = model.arima_res_.nobs
        return

    if update_maxiter is None:
        # Preserve pmdarima's existing default behavior exactly.
        model.update([float(actual)], X=X_step)
    else:
        model.update([float(actual)], X=X_step, maxiter=update_maxiter)


def _model_output_path(
    row: pd.Series, row_number: int, timestamp: str
) -> Path:
    """Resolve the saved fit, with a legacy generated-path fallback."""
    saved_model = row.get("model_file")
    if (
        saved_model is not None
        and not pd.isna(saved_model)
        and str(saved_model).strip()
    ):
        configured = Path(str(saved_model).strip()).expanduser()
        return configured if configured.is_absolute() else solve_path(configured)

    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    dataset = _dataset_path(row, frequency, discretization)
    model_id = _model_id(row_number, dataset, _driver_name(row))
    return output_path(
        timestamp,
        frequency,
        "models",
        dataset.stem,
        f"{model_id}_{timestamp}.pkl",
    )


def _prediction_output_path(
    row: pd.Series, row_number: int, timestamp: str
) -> Path:
    """Return a unique per-model path for one completed test simulation."""
    frequency = _frequency(row["aggregation"])
    discretization = _discretization(row["discretization"])
    dataset = _dataset_path(row, frequency, discretization)
    prediction_file_id = _prediction_file_id(
        row_number,
        dataset,
        _driver_name(row),
        frequency,
        row["lag"],
    )
    return output_path(
        timestamp,
        frequency,
        "predictions",
        dataset.stem,
        f"{prediction_file_id}_{timestamp}.csv",
    )


def _metrics_output_path(prediction_output: Path) -> Path:
    """Mirror a prediction path under the frequency's ``metrics`` folder."""
    predictions_directory = prediction_output.parent
    while (
        predictions_directory.name != "predictions"
        and predictions_directory != predictions_directory.parent
    ):
        predictions_directory = predictions_directory.parent
    if predictions_directory.name != "predictions":
        raise ValueError(
            f"Prediction output is not below a predictions folder: {prediction_output}"
        )
    relative_output = prediction_output.relative_to(predictions_directory)
    return predictions_directory.parent / "metrics" / relative_output


def _dataset_metadata(path: str | Path) -> dict[str, object]:
    """Extract hospital and subset metadata from an assembled dataset name."""
    dataset_name = Path(path).stem
    categorized = dataset_name.endswith("_cat")
    base_name = dataset_name.removesuffix("_cat")
    hospital = next(
        (
            name
            for name in HOSPITAL_NAMES
            if base_name == name or base_name.endswith(f"_{name}")
        ),
        "",
    )
    subset = (
        base_name.removesuffix(f"_{hospital}") if hospital else base_name
    )
    return {
        "dataset_name": dataset_name,
        "hospital": hospital,
        "subset": subset,
        "categorized_dataset": categorized,
    }


def forecast_order(
    row: pd.Series,
    row_number: int = 0,
    model_class: Any | None = None,
    model_output_path: Path | None = None,
    *,
    state_only_updates: bool = False,
    update_maxiter: int | None = None,
) -> pd.DataFrame:
    """Load one saved fit and return its full input series and predictions."""
    _validate_update_options(state_only_updates, update_maxiter)

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
    if model_class is None:
        if model_output_path is None:
            saved_model = row.get("model_file")
            if (
                saved_model is None
                or pd.isna(saved_model)
                or not str(saved_model).strip()
            ):
                raise ValueError("Order row does not identify a fitted model file")
            model_output_path = _model_output_path(row, row_number, "")
        model = _load_fitted_model(model_output_path)
        if tuple(model.order) != order or tuple(model.seasonal_order) != seasonal_order:
            raise ValueError(
                f"{model_output_path}: fitted model orders do not match its order table"
            )
    else:
        # Explicit model classes remain available for isolated tests and callers
        # constructing a model without an Auto-ARIMA artifact.
        model = model_class(
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=_boolean(row["intercept"]),
            suppress_warnings=True,
        ).fit(y_training, X=X_training)
        if model_output_path is not None:
            _save_fitted_model(model, model_output_path)

    training_aic, fitted_parameters = _training_fit_statistics(model)

    LOGGER.info(
        "Using %s trained on %s observations; forecasting %s one-step periods "
        "with %s updates",
        _model_id(row_number, path, driver),
        len(y_training),
        len(y_testing),
        (
            "state-only"
            if state_only_updates
            else "pmdarima default"
            if update_maxiter is None
            else f"pmdarima maxiter={update_maxiter}"
        ),
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
        _update_model(
            model,
            float(actual),
            pd.Timestamp(date),
            X_step,
            state_only_updates=state_only_updates,
            update_maxiter=update_maxiter,
        )

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
            "model_number": row_number + 1,
            "aggregation": frequency,
            "discretization": discretization,
            "path_to_training_file": str(path),
            "category": category,
            "driver_kind": str(row.get("driver_kind", "")),
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
            "confidence_alpha": CONFIDENCE_ALPHA,
            "update_mode": (
                "state_only"
                if state_only_updates
                else "pmdarima_default"
                if update_maxiter is None
                else "pmdarima_limited"
            ),
            "update_maxiter": update_maxiter,
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


def _constant_value(
    predictions: pd.DataFrame, column: str, default: object = ""
) -> object:
    """Return a model-level value repeated across a prediction artifact."""
    if column not in predictions or predictions.empty:
        return default
    value = predictions[column].iloc[0]
    return default if pd.isna(value) else value


def model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build one metadata and evaluation row for a completed model."""
    training_file = str(_constant_value(predictions, "path_to_training_file"))
    metadata = _dataset_metadata(training_file)
    testing = predictions.loc[predictions["phase"].eq("testing")].copy()
    actual = pd.to_numeric(testing[CASE_COLUMN], errors="coerce")
    forecast = pd.to_numeric(testing["prediction"], errors="coerce")
    scored = actual.notna() & forecast.notna()
    actual = actual.loc[scored]
    forecast = forecast.loc[scored]
    absolute_errors = (actual - forecast).abs()
    nonzero = actual.ne(0)

    mean_absolute_error = (
        float(absolute_errors.mean()) if not absolute_errors.empty else float("nan")
    )
    total_absolute_error = (
        float(absolute_errors.sum()) if not absolute_errors.empty else float("nan")
    )
    mape = (
        float((absolute_errors.loc[nonzero] / actual.loc[nonzero].abs()).mean() * 100)
        if nonzero.any()
        else float("nan")
    )
    testing_dates = (
        pd.to_datetime(testing["date"], errors="coerce").dropna()
        if "date" in testing
        else pd.Series(dtype="datetime64[ns]")
    )

    row = {
        "model_id": _constant_value(predictions, "model_id"),
        "model_number": _constant_value(predictions, "model_number"),
        **metadata,
        "path_to_training_file": training_file,
        "temporal_aggregation": _constant_value(predictions, "aggregation"),
        "driver_kind": _constant_value(predictions, "driver_kind"),
        "driver": _constant_value(predictions, "drivers_used"),
        "lag": _constant_value(predictions, "lag"),
        "discretization": _constant_value(predictions, "discretization"),
        "category": _constant_value(predictions, "category"),
        "training_start": _constant_value(predictions, "date_start_training"),
        "training_end": _constant_value(predictions, "date_end_training"),
        "testing_start": (
            "" if testing_dates.empty else testing_dates.min().date().isoformat()
        ),
        "testing_end": _constant_value(predictions, "date_end_testing"),
        "training_observations": int(predictions["phase"].eq("training").sum()),
        "testing_observations": int(len(testing)),
        "scored_predictions": int(scored.sum()),
        "p": _constant_value(predictions, "p"),
        "d": _constant_value(predictions, "d"),
        "q": _constant_value(predictions, "q"),
        "P": _constant_value(predictions, "P"),
        "D": _constant_value(predictions, "D"),
        "Q": _constant_value(predictions, "Q"),
        "seasonal_period": _constant_value(predictions, "seasonal_period"),
        "intercept": _constant_value(predictions, "intercept"),
        "update_mode": _constant_value(predictions, "update_mode"),
        "update_maxiter": _constant_value(predictions, "update_maxiter"),
        "prediction_horizon": _constant_value(predictions, "prediction_horizon"),
        "confidence_alpha": _constant_value(predictions, "confidence_alpha"),
        "model_file": _constant_value(predictions, "model_file"),
        "fitted_parameters": _constant_value(predictions, "fitted_parameters"),
        "aic": _constant_value(predictions, "training_aic", float("nan")),
        # ``ae`` is the mean absolute error so models with different testing
        # lengths remain directly comparable. The total is retained separately.
        "ae": mean_absolute_error,
        "total_absolute_error": total_absolute_error,
        "mape": mape,
        "mape_zero_actuals_excluded": int((~nonzero).sum()),
    }
    return pd.DataFrame([row])


def save_model_metrics(predictions: pd.DataFrame, output: Path) -> Path:
    """Save one model's metadata, AIC, AE, and MAPE immediately."""
    output.parent.mkdir(parents=True, exist_ok=True)
    model_metrics(predictions).to_csv(output, index=False)
    LOGGER.info("Saved completed model metrics to %s", output)
    return output


def save_predictions(predictions: pd.DataFrame, timestamp: str) -> list[Path]:
    """Save one timestamped long-form CSV per distinct training file."""
    outputs: list[Path] = []
    grouped = predictions.groupby(
        ["aggregation", "path_to_training_file"], sort=False
    )
    for (frequency, training_file), rows in grouped:
        output = output_path(
            timestamp,
            str(frequency),
            "predictions",
            f"{Path(str(training_file)).stem}_{timestamp}.csv",
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
    task: tuple[pd.Series, int, Path, Path, bool, int | None],
) -> list[Path]:
    """Load, test, and immediately save predictions plus model metrics."""
    (
        row,
        row_number,
        model_output_path,
        prediction_output_path,
        state_only_updates,
        update_maxiter,
    ) = task
    prediction = forecast_order(
        row,
        row_number,
        model_output_path=model_output_path,
        state_only_updates=state_only_updates,
        update_maxiter=update_maxiter,
    )
    saved_prediction = _save_model_prediction(prediction, prediction_output_path)
    saved_metrics = save_model_metrics(
        prediction, _metrics_output_path(prediction_output_path)
    )
    return [saved_prediction, saved_metrics]


def run(
    orders_path: str | Path,
    jobs: int = 1,
    *,
    state_only_updates: bool = False,
    update_maxiter: int | None = None,
) -> list[Path]:
    """Forecast every order row and immediately save one file per model."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    _validate_update_options(state_only_updates, update_maxiter)
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
                state_only_updates,
                update_maxiter,
            )
        )
    if jobs == 1:
        outputs: list[Path] = []
        for task in tasks:
            outputs.extend(_forecast_order_isolated(task))
        return outputs

    LOGGER.info("Testing models with %s worker processes", jobs)
    outputs: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=jobs,
        max_tasks_per_child=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        task_iterator = iter(tasks)
        pending = {
            executor.submit(_forecast_order_isolated, task)
            for task in (next(task_iterator, None) for _ in range(jobs))
            if task is not None
        }
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                # Each fresh worker saves predictions and metrics before exit,
                # returning native-library memory to the operating system.
                outputs.extend(future.result())
                next_task = next(task_iterator, None)
                if next_task is not None:
                    pending.add(
                        executor.submit(_forecast_order_isolated, next_task)
                    )
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "orders_path",
        nargs="?",
        help="orders manifest produced by ps_dare.auto_arima (legacy positional form)",
    )
    parser.add_argument(
        "--orders",
        dest="orders_option",
        metavar="PATH",
        help="orders manifest produced by ps_dare.auto_arima",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="models to test concurrently (default: %(default)s)",
    )
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "--state-only-updates",
        action="store_true",
        help="append observations without recalibrating model coefficients",
    )
    update_group.add_argument(
        "--update-maxiter",
        type=int,
        metavar="N",
        help="limit each pmdarima coefficient update to N MLE iterations",
    )
    args = parser.parse_args()
    if args.orders_path and args.orders_option:
        parser.error(
            "specify the orders file either positionally or with --orders, not both"
        )
    orders_path = args.orders_option or args.orders_path
    if orders_path is None:
        parser.error("an orders manifest is required; pass --orders PATH")
    if args.update_maxiter is not None and args.update_maxiter < 1:
        parser.error("--update-maxiter must be at least 1")
    for output in run(
        orders_path,
        jobs=args.jobs,
        state_only_updates=args.state_only_updates,
        update_maxiter=args.update_maxiter,
    ):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
