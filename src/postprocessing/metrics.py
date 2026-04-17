import csv
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

EPS = 1e-12


def _sorted_run_keys(keys: list[str]) -> list[str]:
    def _order_key(key: str) -> tuple[int, int]:
        if key == "nominal_run":
            return (0, 0)
        match = re.search(r"(\d+)$", key)
        idx = int(match.group(1)) if match else 10**9
        return (1, idx)

    return sorted(keys, key=_order_key)


def run_payloads(data_path: Path) -> dict[str, dict[str, np.ndarray | float]]:
    with open(data_path, "rb") as f:
        payloads = pickle.load(f)
    return payloads


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
    run_payload: dict[str, np.ndarray | float],
    baseline_payload: dict[str, np.ndarray | float] | None,
) -> dict[str, float]:
    t = _as_float_array(run_payload["time"])
    theta = _as_float_array(run_payload["theta"])
    theta_v = _as_float_array(run_payload["theta_v"])
    u = _as_float_array(run_payload["u"])

    entropy = float(np.mean([
        _entropy(theta[:, i]) for i in range(theta.shape[1])
    ]))

    err = theta - theta_v
    err_sq = np.sum(err**2, axis=1)
    err_abs = np.sum(np.abs(err), axis=1)

    # Tremor residual is modeled as deviation from voluntary component.
    tremor = err[:, 2]
    tremor_power = float(np.mean(tremor**2))
    tremor_amplitude = float(np.ptp(tremor))

    if baseline_payload is None:
        tpsr = np.nan
        asr = np.nan
    else:
        theta_bl = _as_float_array(baseline_payload["theta"])
        theta_v_bl = _as_float_array(baseline_payload["theta_v"])
        tremor_bl = theta_bl[:, 2] - theta_v_bl[:, 2]
        tremor_power_bl = float(np.mean(tremor_bl**2))
        tremor_amplitude_bl = float(np.ptp(tremor_bl))

        tpsr = (
            100.0 * (tremor_power_bl - tremor_power) / tremor_power_bl
            if tremor_power_bl > EPS else np.nan
        )
        asr = (
            100.0 * (tremor_amplitude_bl - tremor_amplitude)
            / tremor_amplitude_bl
            if tremor_amplitude_bl > EPS else np.nan
        )

    # Control signal metrics
    u_sq = np.sum(u**2, axis=1)
    u_abs = np.sum(np.abs(u), axis=1)
    control_power = float(np.mean(u_sq))
    control_rms = float(np.sqrt(control_power))
    u_dot = np.gradient(u, t, axis=0)
    u_dot_abs = np.sum(np.abs(u_dot), axis=1)
    control_tvc = float(trapezoid(u_dot_abs, t))
    control_iac = float(trapezoid(u_abs, t))
    control_isc = float(trapezoid(u_sq, t))

    return {
        "tpsr_percent": float(tpsr),
        "asr_percent": float(asr),
        "rmse": float(np.sqrt(np.mean(err_sq))),
        "response_entropy": entropy,
        "control_power": control_power,
        "control_rms": control_rms,
        "control_tvc": control_tvc,
        "control_iac": control_iac,
        "control_isc": control_isc,
        "ise": float(trapezoid(err_sq, t)),
        "iae": float(trapezoid(err_abs, t)),
        "itae": float(trapezoid(t * err_abs, t)),
        "itse": float(trapezoid(t * err_sq, t)),
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
