# Source package overview

This directory contains the executable simulation and analysis code for the tremor suppression project. It is the main development area for the model, controllers, and post-processing pipeline.

## Entry points and core modules

- [main.py](main.py) — simulation entry point; loads [../configs.yaml](../configs.yaml), instantiates controllers, runs nominal and sampled uncertainty cases, and saves results
- [system.py](system.py) — model parameters and dynamics used by all control strategies
- [pid_tuning.py](pid_tuning.py) — PID tuning utilities and parameter search scripts
- [requirements.txt](requirements.txt) — project dependencies for the source tree

## Control strategy implementations

The [control_strategies](control_strategies) package contains the controllers used during simulation:

- [control_strategies/afe_notch.py](control_strategies/afe_notch.py) — adaptive notch filtering approach
- [control_strategies/eadrc_ebmflc.py](control_strategies/eadrc_ebmflc.py) — EADRC controller with EBMFLC logic
- [control_strategies/eadrc_zplp.py](control_strategies/eadrc_zplp.py) — EADRC controller with ZPLP design
- [control_strategies/pi_gallego.py](control_strategies/pi_gallego.py) — Gallego PI controller
- [control_strategies/pid.py](control_strategies/pid.py) — PID implementation used for tuning and comparison
- [control_strategies/uncontrolled.py](control_strategies/uncontrolled.py) — uncontrolled baseline reference

## Post-processing and analysis

The [postprocessing](postprocessing) package converts saved simulation results into plots and metrics:

- [postprocessing/postprocess.py](postprocessing/postprocess.py) — top-level pipeline to generate tables and plots
- [postprocessing/metrics.py](postprocessing/metrics.py) — metric calculations and CSV writing utilities
- [postprocessing/plots.py](postprocessing/plots.py) — plotting routines
- [postprocessing/statistics.py](postprocessing/statistics.py) — summary/statistical helpers

## Tremor estimation methods

The [tremor_estimation_strategies](tremor_estimation_strategies) folder contains the literature benchmark and algorithm comparison components:

- [tremor_estimation_strategies/methods](tremor_estimation_strategies/methods) — estimator implementations
- [tremor_estimation_strategies/results](tremor_estimation_strategies/results) — per-method result folders
- [tremor_estimation_strategies/utils](tremor_estimation_strategies/utils) — constants, logging, plotting, and signal helpers
- [tremor_estimation_strategies/input_examples](tremor_estimation_strategies/input_examples) — example tremor signals
- [tremor_estimation_strategies/literature_review](tremor_estimation_strategies/literature_review) — papers and review artifacts
- [tremor_estimation_strategies/run_methods.py](tremor_estimation_strategies/run_methods.py) — execute the estimation methods
- [tremor_estimation_strategies/table_results.py](tremor_estimation_strategies/table_results.py) — aggregate comparison tables

## Typical workflow

From the repo root:

```bash
cd src
python main.py
python postprocessing/postprocess.py
```

This mirrors the project workflow used in the repository:

1. load the configuration file
2. simulate the nominal and perturbed models
3. save the numerical outputs in the results directory
4. generate plots and summary metrics from those outputs

## Important notes

- The project is structured as a research simulation workflow, not as a packaged library.
- A number of imports are designed to work when the current working directory is the [src](.) folder.
- The simulation and post-processing steps depend on the configuration in [../configs.yaml](../configs.yaml) and the outputs in [../results](../results).