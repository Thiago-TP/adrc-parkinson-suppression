import csv
import pickle
import re
from glob import glob
from pathlib import Path
from typing import Any

import blosc
import numpy as np
from scipy.integrate import trapezoid

PayLoad = dict[str, np.ndarray | float]

EPS = 1e-12


def _sorted_run_keys(keys: list[str]) -> list[str]:
    def _order_key(key: str) -> tuple[int, int]:
        if key == "nominal_run":
            return (0, 0)
        match = re.search(r"(\d+)$", key)
        idx = int(match.group(1)) if match else 10**9
        return (1, idx)

    return sorted(keys, key=_order_key)


def run_payloads(data_path: Path) -> dict[str, PayLoad]:
    with open(data_path, "rb") as f:
        compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        data = pickle.loads(depressed_pickle)
    return data


def _entropy(signal: np.ndarray, bins: int = 64) -> float:
    if signal.size == 0:
        return np.nan

    hist, _ = np.histogram(signal, bins=bins)
    total = np.sum(hist)
    if total <= 0:
        return np.nan

    p = hist.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + EPS)))


def _as_float_array(value: np.ndarray | float) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _compute_metrics(
    run_payload: PayLoad,
    baseline_payload: PayLoad | None,
) -> dict[str, float]:
    t = _as_float_array(run_payload["time"])
    theta = _as_float_array(run_payload["theta"][:, 2])
    theta_v = _as_float_array(run_payload["theta_v"][:, 2])
    u = _as_float_array(run_payload["u"][:, 2])

    entropy = float(_entropy(theta))

    err = theta_v - theta
    err_sq = err**2
    err_abs = np.abs(err)

    # Tremor residual is modeled as deviation from voluntary component.
    tremor = err
    tremor_power = float(np.mean(tremor**2))
    tremor_amplitude = float(np.ptp(tremor))

    if baseline_payload is None:
        tpsr = np.nan
        asr = np.nan
    else:
        theta_bl = _as_float_array(baseline_payload["theta"][:, 2])
        theta_v_bl = _as_float_array(baseline_payload["theta_v"][:, 2])
        tremor_bl = theta_bl - theta_v_bl
        tremor_power_bl = float(np.mean(tremor_bl**2))
        tremor_amplitude_bl = float(np.ptp(tremor_bl))

        tpsr = (
            100.0 * (tremor_power_bl - tremor_power) / tremor_power_bl
            if tremor_power_bl > EPS
            else np.nan
        )
        asr = (
            100.0 * (tremor_amplitude_bl - tremor_amplitude) / tremor_amplitude_bl  # noqa: E501
            if tremor_amplitude_bl > EPS
            else np.nan
        )

    # Control signal metrics
    u_sq = u**2
    u_abs = np.abs(u)
    control_power = float(np.mean(u_sq))
    control_rms = float(np.sqrt(control_power))
    u_dot = np.gradient(u, t)
    u_dot_abs = np.abs(u_dot)
    control_tvc = float(trapezoid(u_dot_abs, t))
    control_iac = float(trapezoid(u_abs, t))
    control_isc = float(trapezoid(u_sq, t))

    # R2 score
    ss_res = np.sum(err_sq)
    ss_tot = sum((theta_v - np.mean(theta_v)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + EPS))

    return {
        "tpsr_percent": float(tpsr),
        "asr_percent": float(asr),
        "rmse": float(np.sqrt(np.mean(err_sq))),
        "r2": float(r2),
        "response_entropy": entropy,
        "iae": float(trapezoid(err_abs, t)),
        "ise": float(trapezoid(err_sq, t)),
        "itae": float(trapezoid(t * err_abs, t)),
        "itse": float(trapezoid(t * err_sq, t)),
        "control_power": control_power,
        "control_rms": control_rms,
        "control_tvc": control_tvc,
        "control_iac": control_iac,
        "control_isc": control_isc,
    }


def metrics_table_for_file(
    path: Path,
    baseline: Path,
) -> list[dict[str, str | float]]:
    payloads = run_payloads(path)
    baseline_payloads = run_payloads(baseline)
    rows: list[dict[str, str | float]] = []
    for run_key in _sorted_run_keys(list(payloads.keys())):
        metrics = _compute_metrics(
            run_payload=payloads[run_key],
            baseline_payload=baseline_payloads.get(run_key),
        )
        rows.append({"run_key": run_key, **metrics})

    return rows


def write_csv(output_path: Path, rows: list[dict[str, str | float]]) -> None:

    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"* Saved metrics table: {output_path}")


def summarize_metrics_csv(metrics_csv: str | Path) -> Path:
    """
    Read a metrics CSV and save columnwise mean/std to a "-summary" CSV.

    The summary CSV keeps the same metric columns as the input file, excluding
    ``run_key``. It contains two rows: first row is mean, second row is
    standard deviation.
    """
    input_path = Path(metrics_csv)
    summary_path = input_path.with_name(f"{input_path.stem}-summary{input_path.suffix}")

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file has no header: {input_path}")

        metric_columns = [name for name in fieldnames if name != "run_key"]
        if not metric_columns:
            raise ValueError(
                "No metric columns found in CSV "
                f"(expected columns besides run_key): {input_path}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows to summarize: {input_path}")

    values_by_column: dict[str, list[float]] = {col: [] for col in metric_columns}

    for row in rows:
        for col in metric_columns:
            raw_value: Any = row.get(col, "")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric value in column '{col}' "
                    f"for file {input_path}: "
                    f"{raw_value!r}"
                ) from exc
            values_by_column[col].append(value)

    mean_row = {
        col: float(np.mean(np.asarray(values_by_column[col], dtype=float)))
        for col in metric_columns
    }
    std_row = {
        col: float(np.std(np.asarray(values_by_column[col], dtype=float)))
        for col in metric_columns
    }

    summary_rows = [
        {"statistic": "mean", **mean_row},
        {"statistic": "standard deviation", **std_row},
    ]
    summary_columns = ["statistic", *metric_columns]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"* Saved metrics summary: {summary_path}")
    return summary_path


def merge_summaries_into_latex(
    metrics_dir: str = "results/metrics",
    output_file: str | None = None,
    amplitude: str | None = None,
) -> Path:
    """
    Merge all -summary.csv files into a single LaTeX table.

    Creates a LaTeX tabular environment where each row represents a control
    strategy and columns contain metrics formatted as "mean ± std" with two
    decimal places.

    Parameters
    ----------
    metrics_dir : str
        Directory containing the summary CSV files.
    output_file : str, optional
        Path to the output LaTeX file. If None, defaults to
        "results/metrics/merged_summaries.tex" or
        "results/metrics/merged_summaries_amplitude_{amplitude}.tex" if amplitude
        is provided.
    amplitude : str, optional
        Filter summary files to match a specific amplitude value.
        If provided, only files matching this amplitude are included.

    Returns
    -------
    Path
        Path to the created LaTeX file.
    """
    # Set default output file if not provided
    if output_file is None:
        if amplitude is not None:
            output_file = f"{metrics_dir}/merged_summaries_amplitude_{amplitude}.tex"
        else:
            output_file = f"{metrics_dir}/merged_summaries.tex"

    # Find all summary CSV files
    summary_files = sorted(glob(f"{metrics_dir}/*-summary.csv"))
    if not summary_files:
        raise ValueError(f"No summary CSV files found in {metrics_dir}")

    # Dictionary to store formatted metrics for each control strategy
    formatted_data: dict[str, dict[str, str]] = {}
    all_metric_columns: list[str] | None = None

    for file_path in summary_files:
        file_name = Path(file_path).stem
        # Extract control strategy name (remove -summary suffix and amplitude info)
        # File format: {control_name}_amplitude_{amplitude}_metrics-summary
        parts = file_name.replace("_metrics-summary", "").rsplit("_amplitude_", 1)
        control_name = parts[0] if parts else file_name
        file_amplitude = parts[1] if len(parts) > 1 else None

        # Filter by amplitude if specified
        if amplitude is not None and file_amplitude != amplitude:
            continue

        # Read the summary CSV
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if len(rows) < 2:
                continue

            mean_row = rows[0]  # First row is mean
            std_row = rows[1]  # Second row is std

        if all_metric_columns is None:
            all_metric_columns = [col for col in mean_row.keys() if col != "statistic"]

        # Format metrics as "mean ± std"
        formatted_metrics: dict[str, str] = {}
        for col in all_metric_columns:
            try:
                mean_val = float(mean_row.get(col, ""))
                std_val = float(std_row.get(col, ""))
                formatted_metrics[col] = f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
            except (ValueError, TypeError):
                formatted_metrics[col] = "—"

        formatted_data[control_name] = formatted_metrics

    if all_metric_columns is None:
        raise ValueError("No valid metric columns found in summary files")

    # Create LaTeX table
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        # Write table header
        col_spec = "l" + "r" * len(all_metric_columns)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Column headers
        f.write("Strategy")
        for col in all_metric_columns:
            f.write(f" & {col}")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for control_name in sorted(formatted_data.keys()):
            f.write(control_name)
            for col in all_metric_columns:
                value = formatted_data[control_name].get(col, "—")
                f.write(f" & {value}")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"* Saved merged summaries LaTeX table: {output_path}")
    return output_path
