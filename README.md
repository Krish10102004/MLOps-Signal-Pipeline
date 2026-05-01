# MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal, production-style batch job that demonstrates **reproducibility**, **observability**, and **deployment readiness** using an OHLCV dataset.

---

## What it does

| Step | Detail |
|---|---|
| Load config | Parses `config.yaml`, validates required fields, sets NumPy random seed |
| Load dataset | Reads `data.csv` (10 000-row BTC OHLCV), validates schema and `close` column |
| Rolling mean | Computes `pandas` rolling mean on `close` with configurable `window` |
| Signal | `signal = 1` if `close > rolling_mean`, else `0` (first `window-1` rows excluded) |
| Metrics | Writes `metrics.json` on both success and error |
| Logs | Structured timestamped log to `run.log` (+ stdout INFO) |

---

## Project structure

```
.
├── run.py           # Main batch job
├── config.yaml      # Seed / window / version config
├── data.csv         # 10 000-row OHLCV dataset
├── requirements.txt # Python dependencies
├── Dockerfile       # One-command containerised run
├── metrics.json     # Sample output — successful run
├── run.log          # Sample log   — successful run
└── README.md
```

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

Final metrics JSON is printed to stdout **and** written to `metrics.json`.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run (uses bundled data.csv + config.yaml)
docker run --rm mlops-task
```

To retrieve output files from the container:

```bash
docker run --rm -v "$(pwd)/output:/app/output" mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   output/metrics.json \
    --log-file output/run.log
```

Exit code `0` = success, non-zero = failure.

---

## Configuration (`config.yaml`)

| Field | Type | Description |
|---|---|---|
| `seed` | int | NumPy random seed for determinism |
| `window` | int | Rolling mean window size (rows) |
| `version` | str | Pipeline version tag written to metrics |

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 18.17,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: ['timestamp', 'open']",
  "latency_ms": 3.42
}
```

---

## Design decisions

- **NaN warm-up rows**: the first `window - 1` rows have no rolling mean. Rather than forward-filling (which would inject a look-ahead bias), those rows receive `NaN` for both `rolling_mean` and `signal`, and are excluded from `signal_rate` computation. This keeps the metric statistically honest.
- **Determinism**: seed is set immediately after config load, before any data processing.
- **Error safety**: `metrics.json` is always written — even on failure — so downstream monitoring never finds a missing file.
- **No hardcoded paths**: all file paths are CLI arguments; the Dockerfile passes them via `CMD`.
