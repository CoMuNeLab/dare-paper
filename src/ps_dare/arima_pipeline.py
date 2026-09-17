"""Run ARIMA order selection, calibration, persistence, and testing.

Each model is processed as a durable pipeline: ``auto_arima`` immediately
serializes its fitted training model and writes a one-row order table, then
that exact fit is loaded and tested with one-step-ahead forecasts. Completed
simulations are immediately saved to separate CSV files rather than
accumulated in memory.

From the repository root, run::

    python -m ps_dare.arima_pipeline configuration.csv [configuration2.csv ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from logging.handlers import QueueListener
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import auto_arima, predict_arima, predict_baseline

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(log_file: str | Path | None = None) -> Path:
    """Send pipeline logs to both the terminal and a timestamped log file."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / f"arima_pipeline_{timestamp}.log"
    else:
        path = Path(log_file).expanduser()

    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(path, encoding="utf-8"),
        ],
        force=True,
    )
    LOGGER.info("Saving pipeline log to %s", path.resolve())
    return path


def _order_tables(outputs: Iterable[Path]) -> list[Path]:
    """Return only timestamped order tables from ``auto_arima`` outputs."""
    return [
        output
        for output in outputs
        if predict_arima.ORDERS_FILENAME_PATTERN.fullmatch(output.name) is not None
    ]


def _select_predict_and_save(
    task: tuple[pd.Series, str, bool, int | None]
) -> list[Path]:
    """Complete one model before allowing this worker to receive another."""
    row, timestamp, state_only_updates, update_maxiter = task
    driver_kind = row.get("driver_kind") if hasattr(row, "get") else None
    if driver_kind is not None and auto_arima._driver_kind(driver_kind) == "baseline":
        LOGGER.info("Generating and scoring baseline predictions")
        return predict_baseline.save_baseline(row, timestamp)
    order_output = auto_arima._analyze_and_save_model((row, timestamp))
    # Release the in-memory selection fit before loading its saved copy.
    gc.collect()
    LOGGER.info("Loading, testing, and saving predictions for %s", order_output)
    prediction_options: dict[str, Any] = {"jobs": 1}
    if state_only_updates:
        prediction_options["state_only_updates"] = True
    elif update_maxiter is not None:
        prediction_options["update_maxiter"] = update_maxiter
    prediction_outputs = predict_arima.run(order_output, **prediction_options)
    return [order_output, *prediction_outputs]


def _run_parallel_models(
    tasks: Iterable[tuple[pd.Series, str, bool, int | None]],
    jobs: int,
    worker_log_queue: Any | None,
) -> list[Path]:
    """Run at most *jobs* complete model pipelines at once without a backlog."""
    process_context = multiprocessing.get_context("spawn")
    executor_options: dict[str, Any] = {
        "max_workers": jobs,
        # A fresh process per model guarantees native-library memory is returned
        # to the operating system after that model's prediction has been saved.
        "max_tasks_per_child": 1,
        "mp_context": process_context,
    }
    if worker_log_queue is not None:
        executor_options.update(
            initializer=auto_arima._configure_worker_logging,
            initargs=(worker_log_queue,),
        )

    outputs: list[Path] = []
    task_iterator = iter(tasks)
    with ProcessPoolExecutor(**executor_options) as executor:
        pending = {
            executor.submit(_select_predict_and_save, task)
            for task in (next(task_iterator, None) for _ in range(jobs))
            if task is not None
        }
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                outputs.extend(future.result())
                next_task = next(task_iterator, None)
                if next_task is not None:
                    pending.add(executor.submit(_select_predict_and_save, next_task))
    return outputs


def run(
    configurations: Iterable[str | Path],
    jobs: int = 1,
    *,
    _worker_log_queue: Any | None = None,
    state_only_updates: bool = False,
    update_maxiter: int | None = None,
) -> list[Path]:
    """Run the complete ARIMA workflow for every configuration in order."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    configuration_paths = list(configurations)
    if not configuration_paths:
        raise ValueError("At least one configuration file is required")
    predict_arima._validate_update_options(state_only_updates, update_maxiter)

    outputs: list[Path] = []
    for configuration in configuration_paths:
        LOGGER.info("Starting ARIMA pipeline for %s", configuration)
        try:
            configuration_frame = auto_arima.read_configuration(configuration)
        except FileNotFoundError:
            # Retain compatibility with callers that provide configuration
            # through a mocked/custom ``iter_run`` implementation.
            configuration_frame = None
        has_baseline = (
            configuration_frame is not None
            and configuration_frame["driver_kind"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("baseline")
            .any()
        )
        if not has_baseline and (jobs == 1 or configuration_frame is None):
            found_order = False
            for selection_output in auto_arima.iter_run(configuration, jobs=1):
                outputs.append(selection_output)
                if not _order_tables([selection_output]):
                    continue
                found_order = True
                LOGGER.info("Calibrating and testing %s", selection_output)
                prediction_options: dict[str, Any] = {"jobs": 1}
                if state_only_updates:
                    prediction_options["state_only_updates"] = True
                elif update_maxiter is not None:
                    prediction_options["update_maxiter"] = update_maxiter
                outputs.extend(
                    predict_arima.run(selection_output, **prediction_options)
                )
                gc.collect()
            if not found_order:
                LOGGER.warning(
                    "%s produced no eligible model orders; skipping calibration and testing",
                    configuration,
                )
            continue

        assert configuration_frame is not None
        LOGGER.info(
            "Loaded %s analysis specification(s) from %s",
            len(configuration_frame),
            configuration,
        )
        model_rows, skipped_records = auto_arima._expanded_model_rows(
            configuration_frame
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks = [
            (row, timestamp, state_only_updates, update_maxiter)
            for row in model_rows
        ]
        if jobs == 1:
            for task in tasks:
                outputs.extend(_select_predict_and_save(task))
        else:
            outputs.extend(_run_parallel_models(tasks, jobs, _worker_log_queue))
        outputs.extend(auto_arima.save_skipped(pd.DataFrame(skipped_records)))

        if not model_rows:
            LOGGER.warning(
                "%s produced no eligible models; skipping prediction and scoring",
                configuration,
            )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configurations",
        nargs="+",
        help="one or more CSV files describing datasets and model variants",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="complete model pipelines to run in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        help=(
            "path for the pipeline log "
            "(default: logs/arima_pipeline_<timestamp>.log)"
        ),
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
    if args.update_maxiter is not None and args.update_maxiter < 1:
        parser.error("--update-maxiter must be at least 1")
    configure_logging(args.log_file)
    LOGGER.info(
        "Pipeline invocation: %s",
        json.dumps(
            {
                "configurations": args.configurations,
                "jobs": args.jobs,
                "state_only_updates": args.state_only_updates,
                "update_maxiter": args.update_maxiter,
            },
            sort_keys=True,
        ),
    )
    log_queue = multiprocessing.get_context("spawn").Queue()
    listener = QueueListener(
        log_queue,
        *logging.getLogger().handlers,
        respect_handler_level=True,
    )
    listener.start()
    try:
        for output in run(
            args.configurations,
            jobs=args.jobs,
            _worker_log_queue=log_queue,
            state_only_updates=args.state_only_updates,
            update_maxiter=args.update_maxiter,
        ):
            print(f"Saved {output}")
    finally:
        listener.stop()
        log_queue.close()
        log_queue.join_thread()


if __name__ == "__main__":
    main()
