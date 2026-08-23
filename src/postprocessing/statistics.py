"""
This script runs statistical tests on the results of the tremor suppression simulations.
Control strategies are compared against the baseline open-loop (uncontrolled) case
across metrics, and the results are saved to CSV files in the results/metrics folder.
"""

from pathlib import Path

from metrics import (
    metrics_table_for_file,
    write_csv,
)


def table_metrics_from_blosc(
    control_files: list[str],
    baseline_file: str,
    metrics_dir: str = "results/metrics",
) -> None:
    """
    Generate per-response run-quality metrics tables from saved pickle outputs.

    Each generated CSV contains one row per run key in the source data file.
    Columns are metrics grouped by control strategy.
    """
    output_path = Path(metrics_dir)

    for file in control_files:
        file_name = Path(file).stem

        print(f"\nGenerating metrics table for file: {file}")
        rows = metrics_table_for_file(
            Path(file),
            baseline=Path(baseline_file),
        )
        out_csv = output_path / f"{file_name}_metrics-stats.csv"
        write_csv(out_csv, rows)


def main() -> None:
    """
    Run statistical tests on the results of the tremor suppression simulations.
    Control strategies are compared against the baseline open-loop (uncontrolled) case
    across metrics, and the results are saved to CSV files in the results/metrics folder.
    """

    # Case 1: null amplitude (0.0) voluntary tremor
    control_files = list(Path("results/runs").glob("*_amplitude_0.0.data"))
    baseline_file = Path("results/runs/uncontrolled_amplitude_0.0.data")
    table_metrics_from_blosc(control_files, baseline_file)

    # Case 2: non-null amplitude (1.0) voluntary tremor
    control_files = list(Path("results/runs").glob("*_amplitude_1.0.data"))
    baseline_file = Path("results/runs/uncontrolled_amplitude_1.0.data")
    table_metrics_from_blosc(control_files, baseline_file)


if __name__ == "__main__":
    main()
