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

## Create the aggregated datasets

After activating the environment, build all monthly and weekly datasets from
the repository root:

```bash
python -m ps_dare
```

The command combines every case dataset in `data/<frequency>/raw/` with the
continuous and discrete driver files in `data/<frequency>/drivers/`. It writes
both CSV and pickle versions to:

```text
data/monthly/aggregated/{continuous,discrete}/
data/weekly/aggregated/{continuous,discrete}/
```

Existing aggregated files are overwritten. Driver-only dates are retained.
Within the interval from the hospital's first case observation through its
last observation, a missing case count is saved as `0`; dates before the case
series starts or after it finishes retain a blank case count. Warnings printed
during assembly identify missing periods, invalid dates, and case dates that
have no matching driver observation.

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


Estimate the model orders based on the discretization, driver, time limits, and
lag specified in a CSV file:

```bash
python -m src.ps_dare.auto_arima --jobs 6 auto_arima_example.csv
```

The standalone selector saves each fitted model immediately and then produces
one combined manifest at
`data/output/<timestamp>/orders/orders_<timestamp>.csv`. Each manifest row contains
the selected orders and the `model_file` path of the exact fitted model. Use
the manifest printed as `Use for prediction` to run the prediction stage
separately:

```bash
python -m src.ps_dare.predict_arima \
  --orders data/output/20260915_114627/orders/orders_20260915_114627.csv \
  --jobs 6
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
Auto-ARIMA. `--jobs` controls how many complete model pipelines are processed
concurrently. Each worker selects and saves one fitted model, saves all of its
rolling predictions, and exits before another model is assigned; this bounds
live model memory to the requested process count. Omit `--jobs` for
deterministic single-process execution. Pipeline
messages are shown in the terminal and saved to a timestamped file under
`logs/`. Pass `--log-file path/to/pipeline.log` to choose a different file.
The fitted model produced during order selection is immediately serialized;
rolling prediction loads that exact training fit instead of fitting a second
model from the selected orders.

If a pipeline is interrupted, continue it by passing its log file to the
recovery command:

```bash
ps-dare-arima-recover logs/arima_pipeline_20260915_163546.log
```

The equivalent module command is
`python -m src.ps_dare.recover_arima_pipeline LOG_FILE`. Recovery reuses the
configuration, artifact timestamp, and prediction update mode recorded in the
log. It verifies every model's files, skips completed models, rebuilds missing
metrics from completed prediction CSVs, resumes prediction from saved fitted
models, and refits only models that have no usable order/model artifact.
Recovery messages are appended to the same log. Use `--jobs N` to override the
original concurrency.

Rolling updates use pmdarima's default MLE recalibration when no update option
is supplied. Two mutually exclusive faster modes are available:

```bash
# Keep fitted coefficients fixed and update only the forecasting state.
python -m src.ps_dare.arima_pipeline --state-only-updates --jobs 6 auto_arima_example.csv

# Keep recalibration, but limit every observation update to one MLE iteration.
python -m src.ps_dare.arima_pipeline --update-maxiter 1 --jobs 6 auto_arima_example.csv
```

The same `--state-only-updates` and `--update-maxiter N` options are available
on `ps-dare-predict`. They cannot be combined. Prediction CSVs record the
selected strategy in `update_mode` and `update_maxiter`.

Alternatively, the positional form remains supported for an existing orders
file:

```bash
python -m src.ps_dare.predict_arima data/output/20260915_114627/orders/orders_20260915_114627.csv
```

Each saved model is calibrated once on its configured training period. After
each prediction, the observed access count is passed to the model's `update`
method before predicting the next period. Each model's long-form output keeps
the complete access series, the selected driver and its value when applicable,
and predictions with their 95% confidence interval lower and upper bounds for
the configured testing window. A completed simulation is immediately written
to its own file at
`data/output/<timestamp>/<weekly|monthly>/predictions/<dataset>/<dataset>_<driver>_<aggregation>_lag_<lag>_model_<NNN>_<timestamp>.csv`.
For example:
`general_no2_mean_weekly_lag_1_model_007_20260914_153012.csv`.
Completed simulation DataFrames are never retained while other models finish,
so an interrupted batch preserves every finished result file.

After each prediction file is saved, the same worker writes a one-row model
summary with the identical filename under
`data/output/<timestamp>/<weekly|monthly>/metrics/<dataset>/`. This metrics CSV includes the model
number and ID, dataset name and path, hospital, subset, category, temporal
aggregation, driver and lag, discretization, training/testing dates and sample
counts, ARIMA orders, intercept, update strategy, fitted parameters, model
path, training AIC, AE, total absolute error, and MAPE. `ae` is the mean
absolute error over scored testing predictions. `mape` is expressed as a
percentage; observations whose actual value is zero are excluded and counted
in `mape_zero_actuals_excluded`. The worker does not proceed to another model
until both its prediction and metrics files have been written.

The prediction output also stores `training_aic` and `fitted_parameters`. These
are captured immediately after the initial training fit, before rolling test
observations update the model. `fitted_parameters` is a JSON object containing
parallel `parameters` and `pvalues` mappings from coefficient names to values.
To calculate out-of-sample MAPE, use only rows
whose `phase` is `testing`, comparing `n_accesses` with `prediction`. The
metrics stage applies the zero-actual policy described above.

Every pipeline execution writes its artifacts below
`data/output/<timestamp>/` and copies each analysis-directives CSV into that
directory. Inside it, the existing `weekly`/`monthly` hierarchy is preserved
for predictions, metrics, models, and orders.

Every initial training fit is serialized by Auto-ARIMA, before its order table
is published, as a pickle file under
`data/output/<timestamp>/<weekly|monthly>/models/<dataset>/`. The dataset
directory, model ID (including the driver), row number, and orders-file
timestamp uniquely identify the fitted model. Both the order table and the
prediction CSV record its path in `model_file`; prediction loads this artifact
without refitting it. Only load pickle files produced by a trusted source,
because unpickling can execute arbitrary code.

For models with an exogenous driver, testing stops before the first period
whose required lagged driver value is unavailable. That period and every later
period are labeled `after_testing` instead of causing prediction to fail.

Set `DATA_BASE_DIR` in `.env` when the `data/` directory lives outside this
repository.
