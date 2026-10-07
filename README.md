# PS-DARE paper analysis

This repository builds the monthly and weekly PS-DARE analysis datasets,
explores their case and environmental-driver series, and evaluates seasonal
ARIMA and persistence-baseline forecasts. The forecasting workflow is designed
to be restartable: each fitted model, order table, prediction series, and
metrics row is saved as soon as it is completed.

## Environment

The project requires Python 3.13. The supplied Conda environment also installs
the R packages used by the publication-figure scripts.

From the repository root:

```bash
conda env create -f environment.yml
conda activate ps-dare-paper
```

The repository is installed in editable mode, so source changes are available
without reinstalling it.

## Repository layout

```text
data/
  monthly/
    raw/                          Monthly case datasets
    drivers/{continuous,discrete,shared}/
    aggregated/{continuous,discrete}/
  weekly/
    raw/                          Weekly case datasets
    drivers/{continuous,discrete,shared}/
    aggregated/{continuous,discrete}/
  output/<timestamp>/             Durable model-run artifacts
graphs/                            Generated figures and figure data
src/ps_dare/
  __main__.py                     Dataset-building entry point
  assembly.py                     Validation, merging, and batch assembly
  explorer.py                     Interactive case/driver explorer
  auto_arima.py                   Model expansion and order selection
  predict_arima.py                Rolling ARIMA forecasts and metrics
  predict_baseline.py             Rolling persistence baseline and metrics
  arima_pipeline.py               End-to-end forecasting pipeline
  recover_arima_pipeline.py       Interrupted-run recovery
  fig1.ipynb, fig2.ipynb          Figure notebooks
  fig3_prot1.R, fig3_prot2.R      Protocol-comparison figures
tests/                             Unit and integration tests
```

Set `DATA_BASE_DIR` in a repository-root `.env` file when `data/` lives outside
the repository. Relative paths are resolved below that directory; without the
variable they are resolved from the repository root.

## Build the aggregated datasets

```bash
python -m ps_dare
```

This combines every case dataset in `data/<frequency>/raw/` with the continuous
and discrete driver files in `data/<frequency>/drivers/`, writing CSV and
pickle versions to:

```text
data/monthly/aggregated/{continuous,discrete}/
data/weekly/aggregated/{continuous,discrete}/
```

Existing aggregated files are overwritten. Driver-only dates are retained.
Within the interval from a hospital's first case observation through its last,
a missing case count is stored as `0`; dates outside that interval keep a blank
case count. Assembly warnings report invalid or missing periods, duplicate
driver dates, and case dates without matching driver observations.

## Explore the datasets

```bash
ps-dare-explorer
# Equivalent module command:
python -m ps_dare.explorer
```

The explorer switches among monthly and weekly data, continuous and discrete
drivers, datasets, CSV and pickle formats, and diagnosis categories when they
are present. Cases remain in the upper panel while selected drivers are drawn
in the lower panel. Use `--data-dir /path/to/data` to inspect another data
directory.

## Analysis configuration

Forecasting commands consume a CSV such as `auto_arima_example.csv` or one of
the `protocol-*.csv` files. A minimal file is:

```csv
dataset,aggregation,driver_kind,discretization,train_start,train_end,test_end,lag
ili_giustiniani,weekly,none,continuous,2017-01-01,2022-12-31,2025-12-31,0
ili_giustiniani,weekly,weather,continuous,2017-01-01,2022-12-31,2025-12-31,0-1
ili_giustiniani,weekly,baseline,continuous,2017-01-01,2022-12-31,2025-12-31,0
```

The columns have the following meaning:

| Column | Accepted values and behavior |
| --- | --- |
| `dataset` | An absolute/relative CSV or pickle path, or a dataset name resolved as `data/<aggregation>/aggregated/<discretization>/<dataset>.csv` |
| `aggregation` | `weekly`/`week` or `monthly`/`month` (`moth` is also accepted for compatibility) |
| `driver_kind` | `none`, `weather`, `pollution`, `forced`, or `baseline` |
| `discretization` | `continuous` or `discrete` |
| `train_start`, `train_end` | Inclusive initial fitting window |
| `test_end` | Inclusive end of the rolling evaluation window; required for prediction and baseline evaluation |
| `lag` | `0`, `1`, or both (`0-1`, `0,1`, and `0/1` are equivalent) |
| `category` | Optional diagnosis category; if omitted, categories are aggregated |

Headers are normalized, so spaces may replace underscores and common aliases
such as `frequency`, `training_start`, and `testing_end` are accepted.

`weather` and `pollution` rows expand to one independent model per recognized
driver found in the dataset. `none` fits seasonal ARIMA without an exogenous
driver. `forced` fits `(1,1,1)(2,0,0)[m]` with an intercept and no exogenous
driver, where `m` is 52 for weekly and 12 for monthly data. `baseline` uses the
previous observed case count as the next one-step forecast; it has no fitted
likelihood or confidence interval, so AIC, fitted parameters, and prediction
bounds are blank.

## Run the forecasting pipeline

The usual entry point runs model expansion, Auto-ARIMA order selection, model
persistence, rolling one-step prediction, and metric calculation:

```bash
ps-dare-arima-pipeline --jobs 6 protocol-1-lag0.csv protocol-1-lag1.csv
# Equivalent module command:
python -m ps_dare.arima_pipeline --jobs 6 protocol-1-lag0.csv protocol-1-lag1.csv
```

Configuration files are processed in the supplied order. Within each file,
`--jobs N` permits up to `N` complete model pipelines to run concurrently.
Each worker finishes and saves one model before accepting another, bounding
live model memory. Omit `--jobs` for deterministic single-process execution.
Rows with `driver_kind=baseline` are scored directly without fitting ARIMA.

Auto-ARIMA uses seasonal stepwise selection. If its first seasonal-differencing
attempt exhausts the available training samples, selection retries that model
with `D=0`; other fitting errors are logged and the pipeline continues with the
next model. `forced` rows bypass order selection.

Pipeline messages appear in the terminal and are also written to
`logs/arima_pipeline_<timestamp>.log`. Use `--log-file PATH` to choose another
location.

### Rolling update modes

By default, each observed testing value is passed to pmdarima's `update`
method with its normal maximum-likelihood recalibration before the next
forecast. Two mutually exclusive faster modes are available:

```bash
# Keep fitted coefficients fixed and update only the forecasting state.
ps-dare-arima-pipeline --state-only-updates --jobs 6 protocol-1-lag0.csv

# Recalibrate, but limit each observation update to one MLE iteration.
ps-dare-arima-pipeline --update-maxiter 1 --jobs 6 protocol-1-lag0.csv
```

Prediction CSVs record the choice in `update_mode` and `update_maxiter`.

### Recover an interrupted run

Pass the original pipeline log to the recovery command:

```bash
ps-dare-arima-recover logs/arima_pipeline_20260921_125305.log
# Equivalent module command:
python -m ps_dare.recover_arima_pipeline logs/arima_pipeline_20260921_125305.log
```

Recovery reuses the configurations, artifact timestamp, concurrency, and
update mode recorded in the log. It validates the durable files, skips
completed ARIMA models, rebuilds missing metrics from completed prediction
CSVs, resumes prediction from saved fitted models, and refits models only when
no usable order/model artifact exists. Messages are appended to the same log.
Use `--jobs N` to override concurrency, or one of `--state-only-updates`,
`--update-maxiter N`, and `--default-updates` to override the recorded update
mode.

## Run individual stages

Select orders and persist fitted models without running predictions:

```bash
python -m ps_dare.auto_arima --jobs 6 auto_arima_example.csv
```

The selector writes a one-row order artifact for every completed model and a
combined manifest at
`data/output/<timestamp>/orders/orders_<timestamp>.csv`. The manifest includes
the selected orders and the exact fitted model's `model_file` path. Use the
path printed after `Use for prediction`:

```bash
ps-dare-predict \
  --orders data/output/<timestamp>/orders/orders_<timestamp>.csv \
  --jobs 6
```

The orders path may instead be passed positionally. `ps-dare-predict` supports
the same `--state-only-updates` and `--update-maxiter N` options as the full
pipeline.

Evaluate persistence baselines separately for the unique evaluation windows
described by one or more configuration files:

```bash
ps-dare-baseline protocol-1-lag0.csv protocol-1-lag1.csv
# Equivalent module command:
python -m ps_dare.predict_baseline protocol-1-lag0.csv protocol-1-lag1.csv
```

## Outputs and metrics

Every pipeline run copies its configuration CSVs and saves artifacts below a
single run directory:

```text
data/output/<timestamp>/
  <configuration>.csv
  <weekly|monthly>/
    models/<dataset>/              Serialized fitted ARIMA models
    orders/<model-id>/             One-row order tables
    predictions/<dataset>/         Full long-form series and rolling forecasts
    metrics/<dataset>/             One-row model summaries
    orders/skipped.csv             Ineligible specifications, when present
```

Each ARIMA fit is serialized before its order table is published. Prediction
loads that exact fit rather than fitting a second model from the selected
orders. Only unpickle model files from a trusted source, because loading a
pickle can execute arbitrary code.

Prediction files retain the complete series and label rows as
`before_training`, `training`, `testing`, or `after_testing`. Testing rows
contain the one-step forecast and its 95% confidence bounds. For an exogenous
model, testing stops before the first period whose required lagged driver value
is unavailable; that period and all later periods are labeled `after_testing`.
Training AIC and fitted coefficient estimates/p-values are captured before
rolling updates begin.

Each metrics CSV includes model and dataset metadata, fitting and evaluation
windows, ARIMA orders, update strategy, fitted parameters, model path,
training AIC, and these out-of-sample measures:

- `ae`: mean absolute error over scored testing forecasts.
- `total_absolute_error`: sum of absolute errors.
- `mape`: mean absolute percentage error, in percent. Zero actuals are omitted
  and counted in `mape_zero_actuals_excluded`.
- `directional_accuracy`: percentage of consecutive testing steps for which
  the forecast and actual series move in the same direction (including ties).

Baseline artifacts use the same prediction and metrics layout, with the
model-specific fields left blank where they do not apply.

## Figures

`src/ps_dare/fig1.ipynb` and `fig2.ipynb` contain the Python figure workflows.
`fig3_prot1.R` and `fig3_prot2.R` generate the protocol-comparison figures and
use `reticulate` with the `ps-dare-paper` Conda environment to inspect fitted
pmdarima models. Generated PDFs and supporting CSVs are stored in `graphs/`.
Review the input paths near the top of each notebook or R script before
rerunning it for a different output run.

## Tests

Run the test suite from the activated environment:

```bash
pytest
```
