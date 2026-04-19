from glob import glob
from pathlib import Path

from metrics import metrics_table_for_file, summarize_metrics_csv, write_csv
from plots import plot_from_data


def generate_plots(
    control_files: list[str],
    baseline_file: str,
    separator: str,
    run_key: str = "nominal_run",
) -> None:
    """
    Generate plots from previously saved numeric simulation results.

    Parameters
    ----------
    run_key : str, optional
        Simulation run key to plot from each `.data` file.
        Defaults to "nominal_run".
    """

    for file_path in control_files:
        control_name = Path(file_path).stem.split(separator)[0]
        plot_from_data(
            file_path,
            baseline_file,
            control_name,
            run_key=run_key
        )


def generate_metrics_tables(
    control_files: list[str],
    baseline_file: str,
    metrics_dir: str = "results/metrics",
) -> None:
    """
    Generate per-response run-quality metrics tables from saved pickle outputs.

    Each generated CSV contains one row per run key in the source data file.
    TPSR and ASR are computed against the corresponding open-loop run for the
    same voluntary-amplitude scenario.
    """
    output_path = Path(metrics_dir)

    for file in control_files:

        file_name = Path(file).stem

        print(f"\nGenerating metrics table for file: {file}")
        rows = metrics_table_for_file(
            Path(file),
            baseline=Path(baseline_file),
        )
        out_csv = (output_path / f"{file_name}_metrics.csv")
        write_csv(out_csv, rows)
        summarize_metrics_csv(out_csv)


def generate_all(
    results_dir: str = "results/runs",
    extension: str = "data",
    separator: str = "_amplitude_",
) -> None:
    """
    Generate all plots and metrics tables from
    previously saved numeric simulation results.
    """

    # Group files by baseline
    # Files are expected to be named in the format:
    # {control_name}_amplitude_{amplitude}.{extension}
    groups = {
        bl_file: glob(
            f"{results_dir}/*_{bl_file.split(separator)[-1]}")
        for bl_file in glob(f"{results_dir}/uncontrolled_*.{extension}")
    }
    for baseline, controls in groups.items():
        generate_metrics_tables(
            control_files=controls,
            baseline_file=baseline,
        )
        generate_plots(
            control_files=controls,
            baseline_file=baseline,
            separator=separator,
            run_key="nominal_run",
        )


if __name__ == "__main__":

    generate_all()
