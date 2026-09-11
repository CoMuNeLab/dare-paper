## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate ps-dare-paper
```

The environment installs this repository in editable mode, so source changes
are available immediately without reinstalling it.

## Repository layout

```text
data/
  monthly/
    raw/                          Original monthly case datasets
    drivers/{discrete,continuous,shared}/
    aggregated/{discrete,continuous}/
  weekly/
    raw/                          Original weekly case datasets
    drivers/{discrete,continuous,shared}/
    aggregated/{discrete,continuous}/
src/ps_dare/
  __main__.py    Dataset-building entry point
  assembly.py    Dataset validation, merging, and batch assembly
  explorer.py    Live case and driver plotting interface
  paths.py       Repository and optional external-data path resolution
tests/
```

Build all aggregated datasets from the repository root:

```bash
python -m src.ps_dare
```

Explore cases and drivers interactively:

```bash
python -m src.ps_dare.explorer
```

The selectors switch between monthly/weekly data, continuous/discrete driver
files, datasets, CSV/pickle formats, and (when present) diagnosis categories.
Cases remain visible in the upper panel while the selected drivers are updated
live in the lower panel. The installed `ps-dare-explorer` command starts the
same interface. Use `--data-dir /path/to/data` to select a different data
directory explicitly.


Estimate the model orders based on the discretization , driver, time limits, and lag specified in a CSV file:

```bash
python -m src.ps_dare.auto_arima /Users/tommasobertola/Git/ps-dare-paper/auto_arima_example.csv
```

Run order selection, model calibration/persistence, and rolling testing as one
pipeline for one or more configuration files:

```bash
python -m src.ps_dare.arima_pipeline --jobs 6 auto_arima_example_def_1.csv auto_arima_example_def_2.csv auto_arima_example_def_3.csv
```

The installed equivalent is `ps-dare-arima-pipeline`. Configurations are
processed in the order supplied. For each one, testing starts automatically as
soon as its Auto-ARIMA order table has been saved. Rows whose `driver_kind` is
`forced` retain their explicitly defined order; all other eligible rows use
Auto-ARIMA. `--jobs` controls how many independent rows/models are processed
concurrently; omit it for deterministic single-process execution.

Alternatively, from an existing orders file, run predict-arima directly.


Run rolling one-step-ahead predictions from a model-order table:

```bash
python -m src.ps_dare.predict_arima data/weekly/orders/orders.csv
```

Each saved model is calibrated once on its configured training period. After
each prediction, the observed access count is passed to the model's `update`
method before predicting the next period. Each model's long-form output keeps
the complete access series, the selected driver and its value when applicable,
and predictions with their 95% confidence interval lower and upper bounds for
the configured testing window. A completed simulation is immediately written
to its own file at
`data/<weekly|monthly>/predictions/<dataset>/<model_id>_<timestamp>.csv`.
Completed simulation DataFrames are never retained while other models finish,
so an interrupted batch preserves every finished result file.

The prediction output also stores `training_aic` and `fitted_parameters`. These
are captured immediately after the initial training fit, before rolling test
observations update the model. `fitted_parameters` is a JSON object mapping
coefficient names to values. To calculate out-of-sample MAPE, use only rows
whose `phase` is `testing`, comparing `n_accesses` with `prediction`; define a
policy for zero observed values because ordinary percentage error is undefined
there.

Every initial training fit is also serialized before testing as a pickle file:
`data/<weekly|monthly>/models/<dataset>/<model_id>_<timestamp>.pkl`. The dataset
directory, model ID (including the driver), row number, and orders-file
timestamp uniquely identify the fitted model. The prediction CSV's
`model_file` column records the corresponding path. Only load pickle files
produced by a trusted source, because unpickling can execute arbitrary code.

For models with an exogenous driver, testing stops before the first period
whose required lagged driver value is unavailable. That period and every later
period are labeled `after_testing` instead of causing prediction to fail.

Set `DATA_BASE_DIR` in `.env` when the `data/` directory lives outside this
repository.
