"""Recover an interrupted ARIMA pipeline from its log file.

The log identifies the configuration, run timestamp, and rolling-update mode.
Recovery verifies the durable artifacts for every configured model and resumes
from the latest safe boundary: model selection, prediction, or metrics.

From the repository root, run::

    python -m ps_dare.recover_arima_pipeline logs/arima_pipeline_<timestamp>.log
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from . import arima_pipeline, auto_arima, predict_arima
from .paths import output_path, solve_path

LOGGER = logging.getLogger(__name__)

_START_PATTERN = re.compile(r"Starting ARIMA pipeline for (.+)$")
_LOADED_PATTERN = re.compile(
    r"Loaded \d+ analysis specification\(s\) from (.+)$"
)
_ORDERS_TIMESTAMP_PATTERN = re.compile(r"orders_(\d{8}_\d{6})\.csv")
_LOG_TIMESTAMP_PATTERN = re.compile(r"arima_pipeline_(\d{8}_\d{6})\.log$")
_INVOCATION_MARKER = "Pipeline invocation: "

Action = Literal["select", "predict", "metrics"]


@dataclass(frozen=True)
class LoggedConfiguration:
    """A configuration file and the artifact timestamp used for its models."""

    path: str
    timestamp: str


@dataclass(frozen=True)
class RecoveryLog:
    """Pipeline settings reconstructed from a log."""

    configurations: tuple[LoggedConfiguration, ...]
    jobs: int
    state_only_updates: bool
    update_maxiter: int | None


@dataclass(frozen=True)
class RecoveryItem:
    """One incomplete model and the durable stage from which it can resume."""

    action: Action
    row: pd.Series
    timestamp: str
    order_path: Path
    prediction_path: Path | None = None
    metrics_path: Path | None = None


@dataclass(frozen=True)
class RecoveryResult:
    """Counts and files produced by a recovery run."""

    already_complete: int
    selected: int
    predicted: int
    metrics_rebuilt: int
    outputs: tuple[Path, ...]


def _log_fallback_timestamp(log_path: Path, text: str) -> str:
    match = _LOG_TIMESTAMP_PATTERN.search(log_path.name)
    if match is not None:
        return match.group(1)

    match = re.search(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})", text)
    if match is None:
        raise ValueError(
            "Could not determine the pipeline timestamp from the log filename "
            "or its first record"
        )
    return "".join(match.groups()[:3]) + "_" + "".join(match.groups()[3:])


def _invocation_metadata(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        marker_position = line.find(_INVOCATION_MARKER)
        if marker_position < 0:
            continue
        payload = line[marker_position + len(_INVOCATION_MARKER) :]
        try:
            metadata = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid pipeline invocation metadata in log") from error
        if not isinstance(metadata, dict):
            raise ValueError("Pipeline invocation metadata must be a JSON object")
        return metadata
    return None


def _inferred_update_options(text: str) -> tuple[bool, int | None]:
    modes: set[tuple[bool, int | None]] = set()
    if "with state-only updates" in text:
        modes.add((True, None))
    if "with pmdarima default updates" in text:
        modes.add((False, None))
    for match in re.finditer(r"with pmdarima maxiter=(\d+) updates", text):
        modes.add((False, int(match.group(1))))

    if len(modes) > 1:
        raise ValueError("The log contains more than one prediction update mode")
    if modes:
        return modes.pop()

    LOGGER.warning(
        "The log ended before its prediction mode was recorded; using the "
        "pipeline default (pmdarima recalibration)"
    )
    return False, None


def read_recovery_log(log_file: str | Path) -> RecoveryLog:
    """Reconstruct configurations and execution options from *log_file*."""
    log_path = Path(log_file).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Pipeline log not found: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    fallback_timestamp = _log_fallback_timestamp(log_path, text)
    metadata = _invocation_metadata(lines)

    started: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for line in lines:
        start_match = _START_PATTERN.search(line)
        if start_match is not None:
            active = {"path": start_match.group(1).strip(), "timestamps": []}
            started.append(active)
            continue
        if active is None:
            continue
        loaded_match = _LOADED_PATTERN.search(line)
        if loaded_match is not None:
            active["path"] = loaded_match.group(1).strip()
        for timestamp in _ORDERS_TIMESTAMP_PATTERN.findall(line):
            if timestamp not in active["timestamps"]:
                active["timestamps"].append(timestamp)

    configured_paths: list[str]
    if metadata is not None and isinstance(metadata.get("configurations"), list):
        configured_paths = [str(path) for path in metadata["configurations"]]
    else:
        configured_paths = [str(item["path"]) for item in started]
    if not configured_paths:
        raise ValueError("The log does not identify any pipeline configuration")

    configurations: list[LoggedConfiguration] = []
    for position, configured_path in enumerate(configured_paths):
        corresponding = started[position] if position < len(started) else None
        if corresponding is not None:
            configured_path = str(corresponding["path"])
            timestamps = corresponding["timestamps"]
        else:
            timestamps = []
        if len(timestamps) > 1:
            raise ValueError(
                f"Configuration {configured_path} has artifacts from multiple "
                f"timestamps in the same log: {', '.join(timestamps)}"
            )
        configurations.append(
            LoggedConfiguration(
                configured_path,
                timestamps[0] if timestamps else fallback_timestamp,
            )
        )

    if metadata is not None:
        jobs = int(metadata.get("jobs", 1))
        state_only_updates = bool(metadata.get("state_only_updates", False))
        raw_maxiter = metadata.get("update_maxiter")
        update_maxiter = None if raw_maxiter is None else int(raw_maxiter)
    else:
        jobs = 1
        state_only_updates, update_maxiter = _inferred_update_options(text)
    if jobs < 1:
        raise ValueError("The logged jobs setting must be at least 1")
    predict_arima._validate_update_options(state_only_updates, update_maxiter)

    return RecoveryLog(
        tuple(configurations), jobs, state_only_updates, update_maxiter
    )


def _expected_order_path(row: pd.Series, timestamp: str) -> Path:
    frequency = auto_arima._frequency(row["aggregation"])
    discretization = auto_arima._discretization(row["discretization"])
    dataset = auto_arima._dataset_path(row["dataset"], frequency, discretization)
    kind = auto_arima._driver_kind(row["driver_kind"])
    selected_driver = row.get(auto_arima.SELECTED_DRIVER_COLUMN)
    driver = (
        kind
        if kind == "forced"
        else "none"
        if selected_driver is None
        or pd.isna(selected_driver)
        or not str(selected_driver)
        else str(selected_driver)
    )
    record = {
        "aggregation": frequency,
        "path_to_training_file": str(dataset),
        "drivers_used": driver,
        auto_arima.MODEL_NUMBER_COLUMN: int(row[auto_arima.MODEL_NUMBER_COLUMN]),
    }
    return output_path(
        timestamp,
        frequency,
        "orders",
        auto_arima._model_slug(record),
        f"orders_{timestamp}.csv",
    )


def _read_valid_order(path: Path) -> pd.Series | None:
    try:
        orders = predict_arima.read_orders(path)
        if len(orders) != 1:
            return None
        row = orders.iloc[0]
        model_path = predict_arima._model_output_path(
            row, int(row["model_number"]) - 1, predict_arima.orders_timestamp(path)
        )
        if not model_path.is_file() or model_path.stat().st_size == 0:
            return None
        return row
    except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError):
        return None


def _valid_prediction(path: Path) -> bool:
    required = {"model_id", "model_number", "phase", "prediction"}
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return not frame.empty and required.issubset(frame.columns)


def _valid_metrics(path: Path) -> bool:
    required = {"model_id", "model_number", "ae", "mape", "directional_accuracy"}
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return len(frame) == 1 and required.issubset(frame.columns)


def _recovery_item(row: pd.Series, timestamp: str) -> RecoveryItem | None:
    order_path = _expected_order_path(row, timestamp)
    order_row = _read_valid_order(order_path)
    if order_row is None:
        return RecoveryItem("select", row, timestamp, order_path)

    model_number = int(order_row["model_number"]) - 1
    prediction_path = predict_arima._prediction_output_path(
        order_row, model_number, timestamp
    )
    metrics_path = predict_arima._metrics_output_path(prediction_path)
    if not _valid_prediction(prediction_path):
        return RecoveryItem(
            "predict", row, timestamp, order_path, prediction_path, metrics_path
        )
    if not _valid_metrics(metrics_path):
        return RecoveryItem(
            "metrics", row, timestamp, order_path, prediction_path, metrics_path
        )
    return None


def _recover_one(
    task: tuple[RecoveryItem, bool, int | None],
) -> list[Path]:
    item, state_only_updates, update_maxiter = task
    if item.action == "select":
        return arima_pipeline._select_predict_and_save(
            (item.row, item.timestamp, state_only_updates, update_maxiter)
        )
    if item.action == "predict":
        options: dict[str, Any] = {"jobs": 1}
        if state_only_updates:
            options["state_only_updates"] = True
        elif update_maxiter is not None:
            options["update_maxiter"] = update_maxiter
        return predict_arima.run(item.order_path, **options)

    if item.prediction_path is None or item.metrics_path is None:
        raise RuntimeError("Metrics recovery is missing its artifact paths")
    predictions = pd.read_csv(item.prediction_path)
    return [predict_arima.save_model_metrics(predictions, item.metrics_path)]


def _run_recovery_tasks(
    tasks: list[tuple[RecoveryItem, bool, int | None]],
    jobs: int,
    worker_log_queue: Any | None,
) -> list[Path]:
    if not tasks:
        return []
    if jobs == 1:
        outputs: list[Path] = []
        for task in tasks:
            outputs.extend(_recover_one(task))
        return outputs

    options: dict[str, Any] = {
        "max_workers": jobs,
        "max_tasks_per_child": 1,
        "mp_context": multiprocessing.get_context("spawn"),
    }
    if worker_log_queue is not None:
        options.update(
            initializer=auto_arima._configure_worker_logging,
            initargs=(worker_log_queue,),
        )

    outputs = []
    task_iterator = iter(tasks)
    with ProcessPoolExecutor(**options) as executor:
        pending = {
            executor.submit(_recover_one, task)
            for task in (next(task_iterator, None) for _ in range(jobs))
            if task is not None
        }
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                outputs.extend(future.result())
                next_task = next(task_iterator, None)
                if next_task is not None:
                    pending.add(executor.submit(_recover_one, next_task))
    return outputs


def recover(
    log_file: str | Path,
    jobs: int | None = None,
    *,
    state_only_updates: bool | None = None,
    update_maxiter: int | None = None,
    default_updates: bool = False,
    _worker_log_queue: Any | None = None,
) -> RecoveryResult:
    """Verify durable artifacts and continue the interrupted logged run."""
    logged = read_recovery_log(log_file)
    selected_jobs = logged.jobs if jobs is None else jobs
    if selected_jobs < 1:
        raise ValueError("jobs must be at least 1")
    override_count = sum(
        (state_only_updates is True, update_maxiter is not None, default_updates)
    )
    if override_count > 1:
        raise ValueError("Only one prediction update override may be supplied")
    if state_only_updates is True:
        selected_state_only, selected_maxiter = True, None
    elif update_maxiter is not None:
        selected_state_only, selected_maxiter = False, update_maxiter
    elif default_updates:
        selected_state_only, selected_maxiter = False, None
    else:
        selected_state_only = logged.state_only_updates
        selected_maxiter = logged.update_maxiter
    predict_arima._validate_update_options(selected_state_only, selected_maxiter)

    items: list[RecoveryItem] = []
    skipped_frames: list[tuple[pd.DataFrame, str]] = []
    total_models = 0
    for configured in logged.configurations:
        LOGGER.info(
            "Inspecting recovery artifacts for %s (run %s)",
            configured.path,
            configured.timestamp,
        )
        configuration = auto_arima.read_configuration(configured.path)
        model_rows, skipped_records = auto_arima._expanded_model_rows(configuration)
        total_models += len(model_rows)
        items.extend(
            item
            for row in model_rows
            if (item := _recovery_item(row, configured.timestamp)) is not None
        )
        if skipped_records:
            skipped_frames.append(
                (pd.DataFrame(skipped_records), configured.timestamp)
            )

    counts = {
        action: sum(item.action == action for item in items)
        for action in ("select", "predict", "metrics")
    }
    already_complete = total_models - len(items)
    LOGGER.info(
        "Recovery plan: %s complete, %s need selection, %s need prediction, "
        "%s need metrics",
        already_complete,
        counts["select"],
        counts["predict"],
        counts["metrics"],
    )
    tasks = [
        (item, selected_state_only, selected_maxiter)
        for item in items
    ]
    outputs = _run_recovery_tasks(tasks, selected_jobs, _worker_log_queue)
    for skipped, timestamp in skipped_frames:
        outputs.extend(auto_arima.save_skipped(skipped, timestamp))
    LOGGER.info("Pipeline recovery completed successfully")
    return RecoveryResult(
        already_complete=already_complete,
        selected=counts["select"],
        predicted=counts["predict"],
        metrics_rebuilt=counts["metrics"],
        outputs=tuple(outputs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_file", help="log file written by the interrupted pipeline"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="models to recover concurrently (default: reuse logged value or 1)",
    )
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument(
        "--state-only-updates",
        action="store_true",
        help="override the log and use state-only prediction updates",
    )
    update_group.add_argument(
        "--update-maxiter",
        type=int,
        metavar="N",
        help="override the log and limit pmdarima updates to N iterations",
    )
    update_group.add_argument(
        "--default-updates",
        action="store_true",
        help="override the log and use pmdarima's default recalibration",
    )
    args = parser.parse_args()
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.update_maxiter is not None and args.update_maxiter < 1:
        parser.error("--update-maxiter must be at least 1")

    # FileHandler appends by default, keeping the original run and recovery in
    # one audit trail as requested by the supplied log file.
    arima_pipeline.configure_logging(args.log_file)
    log_queue = multiprocessing.get_context("spawn").Queue()
    listener = QueueListener(
        log_queue,
        *logging.getLogger().handlers,
        respect_handler_level=True,
    )
    listener.start()
    try:
        result = recover(
            args.log_file,
            jobs=args.jobs,
            state_only_updates=True if args.state_only_updates else None,
            update_maxiter=args.update_maxiter,
            default_updates=args.default_updates,
            _worker_log_queue=log_queue,
        )
    finally:
        listener.stop()
        log_queue.close()
        log_queue.join_thread()

    for output in result.outputs:
        print(f"Saved {output}")
    print(
        "Recovery complete: "
        f"{result.already_complete} already complete, "
        f"{result.selected} selected and predicted, "
        f"{result.predicted} predicted from saved fits, "
        f"{result.metrics_rebuilt} metrics files rebuilt"
    )


if __name__ == "__main__":
    main()
