from pathlib import Path

from metrics import run_payloads, metrics_table_for_file, write_csv
from plots import plot_from_npz


def generate_plots(
    results_dir: str = "results/runs",
    run_key: str = "nominal_run",
) -> None:
    """
    Generate plots from previously saved numeric simulation results.

    Parameters
    ----------
    run_key : str, optional
        Simulation run key to plot from each `.npz` file.
        Defaults to "nominal_run".
    """
    results_path = Path(results_dir)

    for amplitude in (0.0, 1.0):
        control_files = {
            "adrc": results_path / f"adrc_amplitude_{amplitude}.npz",
            "pid": results_path / f"pid_amplitude_{amplitude}.npz",
            "open_loop": results_path / f"open_loop_amplitude_{amplitude}.npz",
        }

        for control_name, file_path in control_files.items():
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Expected results file '{file_path}' was not found. "
                    "Run main.py first."
                )
            plot_from_npz(str(file_path), control_name, run_key=run_key)


def generate_metrics_tables(
    results_dir: str = "results/runs",
    metrics_dir: str = "results/metrics",
) -> None:
    """
    Generate per-response run-quality metrics tables from saved NPZ outputs.

    Each generated CSV contains one row per run key in the source NPZ file.
    TPSR and ASR are computed against the corresponding open-loop run for the
    same voluntary-amplitude scenario.
    """
    results_path = Path(results_dir)
    output_path = Path(metrics_dir)

    for amplitude in (0.0, 1.0):
        control_files = {
            "adrc": results_path / f"adrc_amplitude_{amplitude}.npz",
            "pid": results_path / f"pid_amplitude_{amplitude}.npz",
            "open_loop": results_path / f"open_loop_amplitude_{amplitude}.npz",
        }

        missing = [
            str(path)
            for path in control_files.values()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Expected result files were not found: " + ", ".join(missing)
            )

        baseline_payloads = run_payloads(control_files["open_loop"])

        for control_name, npz_path in control_files.items():
            print(f"\nGenerating metrics table for file: {npz_path}")
            rows = metrics_table_for_file(
                npz_path,
                baseline_payloads=baseline_payloads,
            )
            out_csv = (
                output_path
                / f"{control_name}_amplitude_{amplitude}_metrics.csv"
            )
            write_csv(out_csv, rows)


if __name__ == "__main__":
    generate_metrics_tables()
    generate_plots(run_key="nominal_run")
