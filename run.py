import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean signal pipeline"
    )
    parser.add_argument("--input",    required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for output log file")
    return parser.parse_args()


# Logging setup

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler — full detail
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Config loading & validation

REQUIRED_CONFIG_FIELDS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value) at top level.")

    missing = REQUIRED_CONFIG_FIELDS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {sorted(missing)}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got: {cfg['version']!r}")

    logger.info(
        "Config loaded — seed=%d | window=%d | version=%s",
        cfg["seed"], cfg["window"], cfg["version"]
    )
    return cfg


# Dataset loading & validation


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    # The provided CSV wraps every row in outer double-quotes.
    # We strip those before handing the text to pandas, so the reader
    # works identically regardless of whether the file has that quirk.
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = []
    for line in raw.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            stripped = stripped[1:-1]
        lines.append(stripped)

    if len(lines) <= 1:
        raise ValueError("Input file contains no data rows (only header or empty).")

    clean_text = "\n".join(lines)

    try:
        df = pd.read_csv(io.StringIO(clean_text))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Dataset is empty after parsing.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Columns present: {df.columns.tolist()}"
        )

    # Coerce close to numeric; raise if it fails
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_invalid = df["close"].isna().sum()
    if n_invalid == len(df):
        raise ValueError("Column 'close' contains no valid numeric values.")
    if n_invalid > 0:
        logger.warning("Found %d non-numeric value(s) in 'close'; those rows will be NaN.", n_invalid)

    logger.info("Dataset loaded — rows=%d | columns=%s", len(df), df.columns.tolist())
    return df



# Signal pipeline

def compute_pipeline(df: pd.DataFrame, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    window = cfg["window"]

    # Rolling mean — first (window-1) rows will be NaN; we exclude them from
    # signal computation rather than forward-filling, keeping the metric honest.
    logger.info("Computing rolling mean (window=%d) on 'close' ...", window)
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    n_nan = df["rolling_mean"].isna().sum()
    logger.debug("Rolling mean computed — NaN warm-up rows excluded from signal: %d", n_nan)

    # Signal: 1 if close > rolling_mean, else 0; NaN where rolling_mean is NaN
    logger.info("Generating binary signal (close > rolling_mean) ...")
    df["signal"] = np.where(
        df["rolling_mean"].isna(),
        np.nan,
        np.where(df["close"] > df["rolling_mean"], 1, 0)
    )

    signal_rows = df["signal"].notna().sum()
    logger.debug("Signal computed for %d rows (%d warm-up rows excluded).", signal_rows, n_nan)

    return df



# Metrics

def compute_metrics(df: pd.DataFrame, cfg: dict, latency_ms: float) -> dict:
    valid_signal = df["signal"].dropna()
    signal_rate = float(round(valid_signal.mean(), 4))
    rows_processed = len(df)

    return {
        "version":        cfg["version"],
        "rows_processed": rows_processed,
        "metric":         "signal_rate",
        "value":          signal_rate,
        "latency_ms":     round(latency_ms, 2),
        "seed":           cfg["seed"],
        "status":         "success",
    }


def write_metrics(metrics: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# Main

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    logger.info("=" * 60)
    logger.info("Job START")
    logger.info("  input  : %s", args.input)
    logger.info("  config : %s", args.config)
    logger.info("  output : %s", args.output)
    logger.info("  log    : %s", args.log_file)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    try:
        # 1. Config
        cfg = load_config(args.config, logger)

        # 2. Seed — set immediately after config load for full reproducibility
        np.random.seed(cfg["seed"])
        logger.debug("NumPy random seed set to %d.", cfg["seed"])

        # 3. Dataset
        df = load_dataset(args.input, logger)

        # 4. Pipeline
        df = compute_pipeline(df, cfg, logger)

        # 5. Metrics
        latency_ms = (time.perf_counter() - t_start) * 1000
        metrics = compute_metrics(df, cfg, latency_ms)

        logger.info("Metrics summary:")
        for k, v in metrics.items():
            logger.info("  %-20s %s", k + ":", v)

        write_metrics(metrics, args.output)
        logger.info("Metrics written to: %s", args.output)

        logger.info("Job END — status=success | latency_ms=%.2f", latency_ms)
        logger.info("=" * 60)

        # Print to stdout for Docker visibility
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.exception("Job FAILED: %s", exc)

        error_metrics = {
            "version":       _safe_version(args),
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    round(latency_ms, 2),
        }
        write_metrics(error_metrics, args.output)
        logger.error("Error metrics written to: %s", args.output)
        logger.info("Job END — status=error")
        logger.info("=" * 60)

        print(json.dumps(error_metrics, indent=2))
        return 1


def _safe_version(args: argparse.Namespace) -> str:
    """Return version string from config if readable, else 'unknown'."""
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("version", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
