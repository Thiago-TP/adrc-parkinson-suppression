"""
This script runs statistical tests on the results of the tremor suppression simulations.
Control strategies are compared against the baseline open-loop (uncontrolled) case
across metrics, and the results are saved to CSV files in the results/metrics folder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from metrics import metrics_table_for_file, write_csv
from scipy.stats import friedmanchisquare, wilcoxon

DATA_DIR = Path("results/metrics")
FILES = {
    "EADRC+EBMFLC": "eadrc_ebmflc_amplitude_{}_metrics-stats.csv",
    "EADRC+ZPLP": "eadrc_zplp_amplitude_{}_metrics-stats.csv",
    "DE-PID": "pid_de_amplitude_{}_metrics-stats.csv",
    "IMC-PID": "pid_imc_amplitude_{}_metrics-stats.csv",
}
METRIC_MAP = {
    "tpsr_percent": "TPSR [%]",
    "r2": "R2",
    "response_entropy": "Entropy [bits]",
    "iae": "IAE [rad x s]",
    "control_power": "Power [(Nm)^2]",
    "control_tvc": "TV [Nm x s]",
}
SCENARIOS = {
    0.0: "Resting (tau_v=0)",
    1.0: "Deliberate motion (tau_v given by eq.22)",
}
TARGET = "EADRC+ZPLP"
BASELINES = ["EADRC+EBMFLC", "DE-PID", "IMC-PID"]
ALPHA = 0.05


def rank_biserial(diff: np.ndarray) -> tuple[float, int]:
    """Calculate rank-biserial correlation for paired differences."""
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) == 0:
        return np.nan, 0

    ranks = pd.Series(np.abs(nonzero_diff)).rank().values
    positive_ranks = ranks[nonzero_diff > 0].sum()
    negative_ranks = ranks[nonzero_diff < 0].sum()
    effect = (positive_ranks - negative_ranks) / (positive_ranks + negative_ranks)
    return effect, len(nonzero_diff)


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in their original order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


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


def evaluate_controls() -> None:
    """
    Evaluate control strategies against the baseline open-loop (uncontrolled) case
    for both null and non-null voluntary tremor amplitudes.
    One CSV file is generated per control strategy per scenario, containing metrics for each run.
    Runs from same key use the same random seed, so they are later paired for statistical comparison.
    """
    # Case 1: null amplitude (0.0) voluntary tremor
    control_files = list(Path("results/runs").glob("*_amplitude_0.0.data"))
    baseline_file = Path("results/runs/uncontrolled_amplitude_0.0.data")
    table_metrics_from_blosc(control_files, baseline_file)

    # Case 2: non-null amplitude (1.0) voluntary tremor
    control_files = list(Path("results/runs").glob("*_amplitude_1.0.data"))
    baseline_file = Path("results/runs/uncontrolled_amplitude_1.0.data")
    table_metrics_from_blosc(control_files, baseline_file)


def wilcoxon_analysis() -> None:
    """
    Run paired Wilcoxon signed-rank tests to compare control strategies against the baseline
    open-loop (uncontrolled) case across metrics, for both null and non-null voluntary tremor amplitudes.
    Results are saved to CSV and LaTeX tables in the results/tables folder.
    """
    data = {}
    for amplitude in SCENARIOS:
        data[amplitude] = {}
        for controller, filename in FILES.items():
            path = DATA_DIR / filename.format(amplitude)
            data[amplitude][controller] = pd.read_csv(path).set_index("run_key")

        key_sets = [set(frame.index) for frame in data[amplitude].values()]
        if not all(keys == key_sets[0] for keys in key_sets):
            raise ValueError(
                f"run_key mismatch across controllers for scenario amp={amplitude}"
            )

    records = []
    for amplitude, scenario_label in SCENARIOS.items():
        frames = data[amplitude]
        common_keys = frames[TARGET].index

        for column, metric_label in METRIC_MAP.items():
            controller_arrays = [
                frames[controller].loc[common_keys, column].values
                for controller in ["EADRC+EBMFLC", "EADRC+ZPLP", "DE-PID", "IMC-PID"]
            ]
            _, friedman_p = friedmanchisquare(*controller_arrays)

            target_values = frames[TARGET].loc[common_keys, column].values
            for baseline in BASELINES:
                baseline_values = frames[baseline].loc[common_keys, column].values
                statistic, p_value = wilcoxon(
                    target_values,
                    baseline_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=True,
                    method="approx",
                )
                effect, effective_pairs = rank_biserial(target_values - baseline_values)
                records.append(
                    {
                        "scenario": scenario_label,
                        "metric": metric_label,
                        "comparison": f"{TARGET} vs {baseline}",
                        "median_target": np.median(target_values),
                        "median_baseline": np.median(baseline_values),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p": p_value,
                        "rank_biserial_r": effect,
                        "n_pairs": effective_pairs,
                        "friedman_p_omnibus": friedman_p,
                    }
                )

    results = pd.DataFrame(records)
    results["p_holm"] = np.nan
    for scenario_label in SCENARIOS.values():
        mask = results["scenario"] == scenario_label
        results.loc[mask, "p_holm"] = holm_bonferroni(
            results.loc[mask, "wilcoxon_p"].values
        )

    results["significant"] = results["p_holm"] < ALPHA
    results["stars"] = results["p_holm"].apply(significance_stars)
    results = results[
        [
            "scenario",
            "metric",
            "comparison",
            "median_target",
            "median_baseline",
            "wilcoxon_statistic",
            "wilcoxon_p",
            "p_holm",
            "stars",
            "rank_biserial_r",
            "n_pairs",
            "friedman_p_omnibus",
            "significant",
        ]
    ]

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)

    output_dir = Path("results/stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "wilcoxon_full_results.csv", index=False)

    metric_order = list(METRIC_MAP.values())
    metric_tex = {
        "TPSR [%]": r"TPSR [\%]",
        "R2": r"$R^2$",
        "Entropy [bits]": "Entropy [bits]",
        "IAE [rad x s]": r"IAE [rad$\times$s]",
        "Power [(Nm)^2]": r"Power [(N$\times$m)$^2$]",
        "TV [Nm x s]": r"TV [N$\times$m$\times$s]",
    }
    scenario_tex = {
        "Resting (tau_v=0)": r"$\boldsymbol{\tau_v = 0}$ (Table~4)",
        "Deliberate motion (tau_v given by eq.22)": (
            r"$\boldsymbol{\tau_v}$ eq.~(22) (Table~5)"
        ),
    }

    def format_p(p_value: float) -> str:
        return "$<$0.001" if p_value < 0.001 else f"{p_value:.3f}"

    def format_cell(row: pd.Series) -> str:
        return (
            f"{format_p(row['p_holm'])}{row['stars']} ({row['rank_biserial_r']:+.2f})"
        )

    lines = [
        r"\begin{table}",
        (
            r"\caption{Paired Wilcoxon signed-rank tests (EADRC+ZPLP vs.\ baseline "
            r"controllers, $n=100$ paired Monte Carlo runs). Holm--Bonferroni "
            r"corrected $p$-value and rank-biserial correlation $r$ (in "
            r"parentheses) are reported per scenario family. Significance: "
            r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$, ns = not significant.}"
        ),
        r"\label{tab:wilcoxon}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"Scenario & Metric & vs.\ EBMFLC & vs.\ DE-PID & vs.\ IMC-PID " + r"\\",
        r"\midrule",
    ]

    scenario_labels = list(SCENARIOS.values())
    for scenario_index, scenario_label in enumerate(scenario_labels):
        for metric_index, metric_label in enumerate(metric_order):
            subset = results[
                (results["scenario"] == scenario_label)
                & (results["metric"] == metric_label)
            ]
            row_cells = [
                format_cell(
                    subset[subset["comparison"] == f"{TARGET} vs {baseline}"].iloc[0]
                )
                for baseline in BASELINES
            ]
            prefix = (
                r"\multirow{6}{*}{\shortstack{" + scenario_tex[scenario_label] + "}}"
                if metric_index == 0
                else ""
            )
            lines.append(
                f"{prefix} & {metric_tex[metric_label]} & "
                f"{row_cells[0]} & {row_cells[1]} & {row_cells[2]} \\\\"
            )
        if scenario_index == 0:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    tex = "\n".join(lines)
    (output_dir / "table_wilcoxon.tex").write_text(tex)
    print("\n--- LaTeX table written to table_wilcoxon.tex ---\n")


def main() -> None:
    """
    Run statistical tests on the results of the tremor suppression simulations.
    Control strategies are compared against the baseline open-loop (uncontrolled) case
    across metrics, and the results are saved to CSV and TEX tables in the results/tables folder.
    """
    # Generate metrics tables from saved simulation outputs
    evaluate_controls()
    # Apply Wilcoxon signed-rank tests to compare control strategies against baseline
    # (also calculates Holm-Bonferroni corrected p-values and rank-biserial correlation coefficients)
    wilcoxon_analysis()


if __name__ == "__main__":
    main()
