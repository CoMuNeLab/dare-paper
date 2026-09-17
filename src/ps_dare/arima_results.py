# %%
import argparse
import numpy as np
import warnings
from pmdarima.arima import ARIMA
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.path_solver import solve_path
from src.arima_with_weather.utils import (
    apply_driver_lag,
    apply_driver_lags,
    cover_empty_dates,
    extract_ts_and_validate,
    filter_by_date,
    format_driver_lags,
    get_info_from_filename,
    get_model_inputs,
    get_result_info_from_filename,
    get_whole_time_series,
    resolve_driver_lags,
)

from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ConvergenceWarning)

# %% Function definitions


# Fit the ARIMA model on the initial training data,
# then iteratively predict and update the model for each subsequent time point,
def return_prediction_and_ae(model, n_cases, exogenous, n_initial_points):
    n_cases = np.asarray(n_cases).reshape(-1)
    if not 0 < n_initial_points < len(n_cases):
        raise ValueError(
            "Initial training size must be greater than 0 and smaller than "
            "the time-series length"
        )
    if exogenous is not None:
        exogenous = np.asarray(exogenous)
        if exogenous.ndim == 1:
            exogenous = exogenous.reshape(-1, 1)
        if exogenous.ndim != 2:
            raise ValueError("Exogenous variables must be a 1-D or 2-D array")
        if len(exogenous) != len(n_cases):
            raise ValueError("Exogenous variables and target must have equal lengths")

    print(f"    Fitting model with {n_initial_points} initial points...")
    model.fit(
        n_cases[:n_initial_points],
        X=exogenous[:n_initial_points] if exogenous is not None else None,
    )

    preds = []
    actuals = []

    for i in range(n_initial_points, len(n_cases)):
        print(f"Predicting for index {i} / {len(n_cases)}", end="\r")
        pred = model.predict(
            n_periods=1,
            X=exogenous[i : i + 1] if exogenous is not None else None,
        )[0]
        preds.append(pred)
        actuals.append(n_cases[i])
        model.update(
            n_cases[i : i + 1],
            X=exogenous[i : i + 1] if exogenous is not None else None,
        )  # Update the model with the new data point

    return np.array(preds), np.abs(np.array(actuals) - np.array(preds))


# %% The main loop to process all result files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved ARIMA orders with rolling one-step forecasts."
    )
    parser.add_argument(
        "--orders-dir",
        default="data/arima-orders-stationary",
        help="Directory containing saved ARIMA order pickle files.",
    )
    parser.add_argument(
        "--results-dir",
        default="data/arima-results",
        help="Directory in which prediction result pickle files are written.",
    )
    parser.add_argument(
        "--time-threshold",
        default="2018-01-01",
        help="Discard observations before this date (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    orders_dir = solve_path(args.orders_dir)
    results_dir = solve_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    files = os.listdir(orders_dir)
    files = [f for f in files if f.endswith(".pkl")]

    for file in files:
        try:
            print(f"Processing file: {file}")
            path = orders_dir / file
            with path.open("rb") as handle:
                df = pickle.load(handle)
            disease_category, hospital, kind, disc, temporal, filename_driver_lags = (
                get_info_from_filename(file)
            )
            model_driver_lags = resolve_driver_lags(df, filename_driver_lags)

            if temporal == "weekly":
                date_col = "week"
                initial_train_size = 52 * 3  # 3 years of weekly data
            elif temporal == "monthly":
                date_col = "month"
                initial_train_size = 12 * 3  # 3 years of monthly data
            else:
                raise ValueError(f"Invalid temporal aggregation: {temporal}")

            disease_category = (
                None if disease_category.startswith("no-category") else disease_category
            )
            valid_drivers = df["result"].keys()
            df_whole_ts = get_whole_time_series(kind, hospital, disc, temporal)
            fill_date = cover_empty_dates(df_whole_ts, temporal)
            df_whole_ts_by_cat = extract_ts_and_validate(
                df_whole_ts,
                category=disease_category,
                fill_date=fill_date,
                temporal=temporal,
            )

            df_whole_ts_by_cat = filter_by_date(
                df_whole_ts_by_cat, args.time_threshold, temporal
            )
            lagged_drivers = [
                driver
                for driver in valid_drivers
                if driver in df_whole_ts_by_cat.columns
            ]
            if len(model_driver_lags) == 1:
                df_whole_ts_by_cat = apply_driver_lag(
                    df_whole_ts_by_cat,
                    lagged_drivers,
                    lag=model_driver_lags[0],
                )
            else:
                df_whole_ts_by_cat = apply_driver_lags(
                    df_whole_ts_by_cat, lagged_drivers, lags=model_driver_lags
                )

            res_arima = {
                "predictions": {},
                "real_values": df_whole_ts_by_cat[[date_col, "n_accesses"]],
                "driver_lags": model_driver_lags,
            }
            for driver in valid_drivers:
                n_cases, exogenous = get_model_inputs(
                    df_whole_ts_by_cat, driver, model_driver_lags
                )
                model = ARIMA(
                    order=df["result"][driver]["order"],
                    seasonal_order=df["result"][driver]["seasonal_order"],
                    with_intercept=df["result"][driver]["with_intercept"],
                )
                res, ae = return_prediction_and_ae(
                    model,
                    n_cases,
                    exogenous,
                    n_initial_points=initial_train_size,
                )
                res_arima["predictions"][driver] = {
                    "predictions": res,
                    "absolute_errors": ae,
                    "mae": ae.mean(),
                    "n_observations": len(n_cases),
                    "n_predictions": len(res),
                    "has_driver": exogenous is not None,
                    "model_summary": model.arima_res_.summary(),
                }
                print(f"    Driver: {driver}, MAE: {ae.mean():.2f}")
            result_path = results_dir / f"results_stationary_{file}"
            with result_path.open("wb") as handle:
                pickle.dump(res_arima, handle)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
