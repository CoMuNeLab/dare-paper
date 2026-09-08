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

Set `DATA_BASE_DIR` in `.env` when the `data/` directory lives outside this
repository.
