"""Run ARIMA order selection, calibration, persistence, and testing.

Each model is processed as a durable pipeline: ``auto_arima`` writes its own
one-row order table as soon as selection finishes, then it is calibrated on
its training window, serialized before any rolling updates, and tested with
one-step-ahead forecasts.  Completed simulations are immediately saved to
separate CSV files rather than accumulated in memory.

From the repository root, run::

    python -m ps_dare.arima_pipeline configuration.csv [configuration2.csv ...]
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from . import auto_arima, predict_arima

LOGGER = logging.getLogger(__name__)


def _order_tables(outputs: Iterable[Path]) -> list[Path]:
    """Return only timestamped order tables from ``auto_arima`` outputs."""
    return [
        output
        for output in outputs
        if predict_arima.ORDERS_FILENAME_PATTERN.fullmatch(output.name) is not None
    ]


def run(configurations: Iterable[str | Path], jobs: int = 1) -> list[Path]:
    """Run the complete ARIMA workflow for every configuration in order."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    configuration_paths = list(configurations)
    if not configuration_paths:
        raise ValueError("At least one configuration file is required")

    outputs: list[Path] = []
    for configuration in configuration_paths:
        LOGGER.info("Starting ARIMA pipeline for %s", configuration)
        found_order = False
        if jobs == 1:
            for selection_output in auto_arima.iter_run(configuration, jobs=1):
                outputs.append(selection_output)
                if not _order_tables([selection_output]):
                    continue
                found_order = True
                LOGGER.info("Calibrating and testing %s", selection_output)
                outputs.extend(predict_arima.run(selection_output, jobs=1))
        else:
            selection_jobs = max(1, jobs // 2)
            testing_jobs = jobs - selection_jobs
            LOGGER.info(
                "Sharing worker budget: %s order selectors, %s model testers",
                selection_jobs,
                testing_jobs,
            )
            pending_tests: list[Future[list[Path]]] = []
            with ThreadPoolExecutor(max_workers=testing_jobs) as test_executor:
                for selection_output in auto_arima.iter_run(
                    configuration, jobs=selection_jobs
                ):
                    outputs.append(selection_output)
                    if not _order_tables([selection_output]):
                        continue
                    found_order = True
                    LOGGER.info("Queueing calibration and testing for %s", selection_output)
                    pending_tests.append(
                        test_executor.submit(
                            predict_arima.run, selection_output, jobs=1
                        )
                    )

                    unfinished: list[Future[list[Path]]] = []
                    for future in pending_tests:
                        if future.done():
                            outputs.extend(future.result())
                        else:
                            unfinished.append(future)
                    pending_tests = unfinished

                for future in as_completed(pending_tests):
                    outputs.extend(future.result())

        if not found_order:
            LOGGER.warning(
                "%s produced no eligible model orders; skipping calibration and testing",
                configuration,
            )

    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
        help="models/configuration rows to process concurrently (default: %(default)s)",
    )
    args = parser.parse_args()
    for output in run(args.configurations, jobs=args.jobs):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
