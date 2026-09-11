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

From the orders file, run the predict-arima


Run rolling one-step-ahead predictions from a model-order table:

```bash
python -m ps_dare.predict_arima data/weekly/orders/orders_YYYYMMDD_HHMMSS.csv
```

Each saved model is calibrated once on its configured training period. After
each prediction, the observed access count is passed to the model's `update`
method before predicting the next period. The combined long-form output keeps
the complete access series, the selected driver and its value when applicable,
and predictions with their 95% confidence interval lower and upper bounds for
the configured testing window. It is written to
`data/<weekly|monthly>/predictions/`, with one CSV named after each distinct
training file and suffixed with the timestamp from the input orders file (for
example, `general_YYYYMMDD_HHMMSS.csv`). Multiple driver models for the same
training file remain in the same output and are distinguished by `model_id`.

For models with an exogenous driver, testing stops before the first period
whose required lagged driver value is unavailable. That period and every later
period are labeled `after_testing` instead of causing prediction to fail.

Set `DATA_BASE_DIR` in `.env` when the `data/` directory lives outside this
repository.
