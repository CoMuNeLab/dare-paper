# PS-DARE data assembly

This document describes how `src/ps_dare/assembly.py` combines prepared
hospital case counts with weather, humidity, and pollution drivers. The
hospital case pickles can be generated with
`src/ps_dare/from_raw_to_pickle.ipynb`; the environmental driver `.dat` files
described below must already exist because their upstream preparation is not
implemented in this repository.

## Build the raw case pickles

Run `src/ps_dare/from_raw_to_pickle.ipynb` from this repository's environment.
The notebook reads the legacy weekly hospitalization JSON exports and the ILI
and heat-wave ICD-9 ontologies, retains the first diagnosis associated with
each encounter, and constructs weekly and monthly case counts.

By default, the notebook looks for the source data first in this repository
and then in a sibling repository named `ps-dare`. To use another location, set
the legacy repository root before running the notebook:

```dotenv
PS_DARE_SOURCE_ROOT=/absolute/path/to/legacy/ps-dare
```

The source root must contain:

```text
data/
  hospitalization/weekly-aggregation/json-dump/
    ps-adulti-giustiniani-srigato-*.json
    ps-adulti-giustiniani-dettaglio-diagnosi-*.json
    ps-adulti-osa-srigato-*.json
    ps-adulti-osa-dettaglio-diagnosi-*.json
    ps-pediatrico-srigato-*.json
    ps-pediatrico-dettaglio-diagnosi-*.json
  disease_ontology/
    HW/ontology_heatwave.csv
    ILI/sorted_uniqued_full_manualv3.csv
```

The notebook writes exactly 30 case pickles: 5 case series for each of 3
hospitals at both weekly and monthly frequencies. They are placed directly in
the folders consumed by `assembly.py`:

```text
data/weekly/raw/<case input>.pkl
data/monthly/raw/<case input>.pkl
```

The output root honors `DATA_BASE_DIR` as described below. Existing files with
the same names are overwritten. The notebook checks that its written-file
inventory exactly matches the 30 inputs expected by this repository.

## Run the assembly

Create the environment and activate it as described in `README.md`, then run
from the repository root:

```bash
python -m ps_dare
```

The command builds every combination of:

- 2 temporal frequencies: `monthly` and `weekly`;
- 3 hospitals: `giustiniani`, `osa`, and `pediatrico`;
- 5 case variants: general, ILI, heat-wave-related illness, categorized ILI,
  and categorized heat-wave-related illness;
- 2 driver modes: `continuous` and `discrete`.

This produces 60 logical datasets, each written as both CSV and pickle, for a
total of 120 output files. Existing files with the same names are overwritten.

If the `data/` directory is external to the repository, set its parent in the
repository-root `.env` file:

```dotenv
DATA_BASE_DIR=/absolute/path/to/data-parent
```

All paths below are resolved relative to `DATA_BASE_DIR`, or to the repository
root when that variable is unset.

## Directory structure

Assembly expects this layout for both frequencies:

```text
data/
  <monthly|weekly>/
    raw/
      <case input>.pkl
    drivers/
      continuous/
        humidex_wet_bulb_padova_era5.dat
        hot_index_padova.dat
        cold_index_padova.dat
        pollution.dat
      discrete/
        humidex_wet_bulb_padova_era5.dat
        heatwave_frequency_padova.dat
        cold_spell_days_padova.dat
        tropical_night_padova.dat
        frost_days_padova.dat
      shared/
        humidity_padova.dat
    aggregated/
      continuous/
      discrete/
```

The `.dat` files are comma-separated text files and are loaded with
`pandas.read_csv`.

## Case inputs

For each hospital, the following raw pickle files are required at both monthly
and weekly frequencies:

| Case series | Input filename | Output stem |
| --- | --- | --- |
| General emergency-department accesses | `general_<hospital>.pkl` | `general_<hospital>` |
| ILI accesses, all categories combined | `ili_<hospital>.pkl` | `ili_<hospital>` |
| Heat-wave-related illness, all categories combined | `hw_<hospital>.pkl` | `hw_<hospital>` |
| ILI accesses by diagnosis category | `ili_<hospital>_cat.pkl` | `ili_<hospital>_cat` |
| Heat-wave-related illness by diagnosis category | `hw_<hospital>_cat.pkl` | `hw_<hospital>_cat` |

Here, `<hospital>` is one of `giustiniani`, `osa`, or `pediatrico`.

Monthly case frames must contain:

- `month`: a date-like value identifying the month;
- `n_accesses`: the case count;
- `diagnosis_category` for `_cat` datasets.

The current monthly inputs also contain `month_index`; assembly preserves it
but does not use it for matching.

Weekly case frames must contain:

- `week`: a date-like value identifying the week start;
- `n_accesses`: the case count;
- `diagnosis_category` for `_cat` datasets.

Additional case columns are preserved. Categorized inputs may contain several
rows for the same date, one per diagnosis category. The join therefore permits
many case rows to match one driver row and repeats that date's driver values
across the category rows.

## Driver inputs

Every driver frame must contain a parseable `valid_time` column. Additional
columns are preserved in the assembled output.

### Continuous mode

Continuous datasets merge these files in order:

| Directory | File | Current driver columns |
| --- | --- | --- |
| `continuous/` | `humidex_wet_bulb_padova_era5.dat` | `humidex`, `indoor_wet_bulb` |
| `continuous/` | `hot_index_padova.dat` | `tropical_night`, `tropical_night_threshold`, `summer_days`, `summer_days_threshold`, `excess_minimal`, `excess_maximal` |
| `continuous/` | `cold_index_padova.dat` | `frost`, `frost_threshold`, `heating_degree_days`, `heating_degree_days_threshold`, `default_minimal`, `default_maximal` |
| `shared/` | `humidity_padova.dat` | `abs_hu`, `hu` |
| `continuous/` | `pollution.dat` | `NO2_mean`, `NO2_peak`, `SO2_mean`, `SO2_peak`, `CO_mean`, `CO_peak`, `pm10_mean`, `pm10_peak`, `pm2.5_mean`, `pm2.5_peak` |

For the humidex/wet-bulb and pollution files, the first CSV column is treated
as an index and is not merged as data.

### Discrete mode

Discrete datasets merge these files in order:

| Directory | File | Current driver columns |
| --- | --- | --- |
| `discrete/` | `humidex_wet_bulb_padova_era5.dat` | `moderate_max`, `intense_max`, `extreme_max`, `moderate_mean`, `intense_mean`, `extreme_mean`, `iwetbulb_mean_30`, `iwetbulb_max_30`, `iwetbulb_mean_25`, `iwetbulb_max_25` |
| `discrete/` | `heatwave_frequency_padova.dat` | `heat_wave_days`, `heat_wave_days_25tx`, `heat_wave_days_30tx` |
| `discrete/` | `cold_spell_days_padova.dat` | `cold_spell_days`, `cold_spell_days_5tn`, `cold_spell_days_0tn` |
| `discrete/` | `tropical_night_padova.dat` | `tropical_night`, `tropical_night_heatwave`, `tropical_night_cum` |
| `discrete/` | `frost_days_padova.dat` | `frost_days`, `max_cum_frost` |
| `shared/` | `humidity_padova.dat` | `abs_hu`, `hu` |

For the discrete humidex/wet-bulb and frost files, the first CSV column is
treated as an index and is not merged as data. Pollution is included only in
continuous datasets.

## Assembly procedure

Each output dataset is constructed independently using the following steps.

### 1. Load and inspect the case series

The raw case pickle is loaded with `pandas.read_pickle`. Before joining,
assembly checks the case date column for invalid values and missing periods:

- monthly dates are compared as calendar month ends;
- weekly dates are expected every seven days from the first date to the last.

Detected problems produce warnings; a gap in the sequence does not by itself
stop assembly.

### 2. Normalize dates

All joins use a shared `valid_time` column.

For monthly data, case `month` values are moved to the end of their calendar
month. Driver `valid_time` values are not shifted, so monthly driver files are
expected to use month-end dates.

For weekly data, case `week` values are used unchanged. One day is added to
every driver `valid_time` before joining. This aligns driver periods ending on
Sunday with case weeks beginning on Monday. For example:

```text
driver valid_time 2024-01-07 -> join valid_time 2024-01-08
case week          2024-01-08 -> join valid_time 2024-01-08
```

On later joins, an already established `valid_time` is retained. This is what
allows a date introduced by an earlier driver file to survive all subsequent
merges.

### 3. Clean invalid and duplicate dates

Case rows whose date cannot be parsed are warned about and removed.

Driver rows whose `valid_time` cannot be parsed are likewise warned about and
removed. If a driver contains duplicate `valid_time` values, assembly warns
and retains the first row for each duplicated date. The resulting driver date
key must be unique.

### 4. Outer-join every driver

Drivers are joined one at a time on `valid_time` using an outer join. This has
two important consequences:

- case dates with no driver match remain in the dataset and have missing
  values for that driver's columns;
- driver-only dates remain in the dataset and initially have missing case
  values.

Assembly warns after each join when rows already present on the left do not
match that driver. After the first join, that count can include dates
introduced by an earlier driver as well as case rows. It counts rows rather
than unique dates, so a categorized dataset can report several unmatched rows
for one date.

### 5. Fill internal case gaps

After every driver has been merged, assembly identifies the first and last
rows with both a valid date and a non-null `n_accesses` value. A missing
`n_accesses` between those two dates is replaced with `0`.

Missing case values before the first observed case date or after the last
observed case date remain null. This distinction prevents environmental-only
coverage outside a hospital's observation window from being interpreted as
zero emergency-department activity.

The operation fills existing rows only. It does not synthesize every possible
date/category combination for categorized datasets.

### 6. Write CSV and pickle outputs

The finished frame is written to both formats:

```text
data/<frequency>/aggregated/<mode>/<dataset>.csv
data/<frequency>/aggregated/<mode>/<dataset>.pkl
```

For example:

```text
data/weekly/aggregated/continuous/general_giustiniani.csv
data/weekly/aggregated/continuous/general_giustiniani.pkl
data/monthly/aggregated/discrete/ili_osa_cat.csv
data/monthly/aggregated/discrete/ili_osa_cat.pkl
```

CSV files are written with the pandas row index. Downstream project loaders
discard columns whose names begin with `Unnamed:`. Pickle files preserve pandas
dtypes more exactly.

## Resulting data model

Every assembled frame contains at least:

- the original `month` or `week` case-date column;
- `n_accesses`;
- normalized `valid_time` used by downstream exploration and forecasting;
- the columns supplied by all driver files for the selected mode.

Categorized outputs also contain `diagnosis_category`. Driver-only rows may
have null original case-date and category fields while still having a valid
`valid_time` and environmental measurements.

The assembly code does not explicitly sort the completed frame after its outer
joins. Downstream time-series preparation parses `valid_time`, groups by date,
and sorts the resulting date index before analysis.

## Warnings and failures

Assembly continues with warnings for:

- invalid case or driver dates;
- missing weekly or monthly case periods;
- duplicate driver dates, after keeping the first row;
- case rows with no matching date in a driver file.

Assembly stops with an exception when, for example:

- an expected input file is absent;
- a required `month`, `week`, `valid_time`, or `n_accesses` column is absent;
- an input file cannot be decoded by pandas;
- a direct call requests a frequency other than `monthly` or `weekly`.

Because the batch command builds combinations sequentially, outputs completed
before an exception remain on disk.

## Relevant implementation and tests

- `src/ps_dare/assembly.py` contains path expansion, validation, joining, gap
  filling, and serialization.
- `src/ps_dare/__main__.py` invokes the complete build matrix.
- `tests/test_assembly.py` covers weekly date alignment, preservation of
  driver-only dates across repeated merges, invalid-date removal, internal-gap
  filling, and unsupported frequencies.

Run the assembly tests with:

```bash
pytest tests/test_assembly.py
```
