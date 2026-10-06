from src.ps_dare import auto_arima
from src.ps_dare.recover_arima_pipeline import read_recovery_log, _recovery_item

LOGS = [
    "logs/arima_pipeline_20260921_125305.log",
    "logs/arima_pipeline_20260921_150717.log",
    "logs/arima_pipeline_20260921_180755.log",
    "logs/arima_pipeline_20260921_231038.log",
    "logs/arima_pipeline_20260922_005629.log",
    "logs/arima_pipeline_20260922_030448.log",
]
for log in LOGS:
    run = read_recovery_log(log)

    for configured in run.configurations:
        config = auto_arima.read_configuration(configured.path)
        rows, _ = auto_arima._expanded_model_rows(config)

        for row in rows:
            # Baselines do not undergo ARIMA calibration.
            if auto_arima._driver_kind(row["driver_kind"]) == "baseline":
                continue

            item = _recovery_item(row, configured.timestamp)
            if item is not None and item.action == "select":
                number = int(row[auto_arima.MODEL_NUMBER_COLUMN])
                driver = row.get(auto_arima.SELECTED_DRIVER_COLUMN, "") or "none"
                lag = row["lag"]
                print(
                    f"model_{number:03d}: "
                    f"{row['dataset']}, {driver}, {row['aggregation']}, lag {lag}"
                )
