# EADRC Parkinson's Tremor Suppression

This repository studies tremor suppression in a 3-DOF biomechanical arm model by comparing several control strategies under parameter uncertainty. The codebase is organized around a simulation workflow: define the model, run multiple stochastic realizations, and then summarize the resulting trajectories with post-processing metrics and plots.

## Project goals

- Model a human arm as a 3-joint system with nominal parameters and sampled stiffness uncertainty
- Compare multiple suppression strategies against the same baseline conditions
- Save simulation outputs to a reproducible results directory
- Generate summary plots and metrics tables after the simulations finish

## Repository layout

```text
.
├── LICENSE
├── README.md
├── configs.yaml
├── pyproject.toml
├── results/
│   ├── metrics/
│   ├── plots/
│   └── runs/
├── docs/
│   └── literature_review/
└── src/
    ├── README.md
    ├── main.py
    ├── pid_tuning.py
    ├── requirements.txt
    ├── system.py
    ├── control_strategies/
    │   ├── afe_notch.py
    │   ├── eadrc_ebmflc.py
    │   ├── eadrc_zplp.py
    │   ├── pi_gallego.py
    │   ├── pid.py
    │   └── uncontrolled.py
    ├── postprocessing/
    │   ├── metrics.py
    │   ├── plots.py
    │   ├── postprocess.py
    │   └── statistics.py
    └── tremor_estimation_strategies/
        ├── input_examples/
        ├── literature_review/
        ├── methods/
        ├── results/
        ├── run_methods.py
        ├── table_results.py
        └── utils/
```

## Model and configuration

The nominal arm parameters and uncertainty ranges are defined in [configs.yaml](configs.yaml). This file contains:

- geometric and inertial parameters for the shoulder, elbow, and wrist dynamics
- stiffness values and stiffness intervals for Monte Carlo-style sampling
- initial conditions for the joint states

The main model logic is implemented in [src/system.py](src/system.py), and the top-level simulation driver is [src/main.py](src/main.py).

## Control strategies in the current codebase

The active simulation entry point currently instantiates and runs multiple controller variants, including:

- AFE notch filtering
- EADRC with EBMFLC
- EADRC with ZPLP
- Gallego PI
- PID controllers
- uncontrolled baseline

The exact controller implementations live under [src/control_strategies](src/control_strategies).

## Getting started

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

The project declares dependencies in [pyproject.toml](pyproject.toml) and includes a pip-compatible list in [src/requirements.txt](src/requirements.txt).

```bash
pip install -r src/requirements.txt
```

If you prefer the package metadata workflow:

```bash
pip install -e .
```

### 3. Run the simulations

The simulation entry point expects to run from inside the [src](src) directory because the scripts import modules directly from that folder.

```bash
cd src
python main.py
```

This executes the configured control strategies across the nominal model and additional sampled-stiffness cases. Results are written under [results/runs](results/runs).

## Output files

The workflow writes numerical outputs and then post-processes them.

### Simulation outputs

The generated run files live in [results/runs](results/runs). These are the primary numeric artifacts used to compare controllers and amplitudes.

### Post-processing

From the repository root, or from within [src](src), run:

```bash
cd src
python postprocessing/postprocess.py
```

This script reads the saved run files and generates summary plots and metrics under:

- [results/plots](results/plots)
- [results/metrics](results/metrics)

## Notes on the workflow

- The simulation driver in [src/main.py](src/main.py) accepts `num_simulations` and `amplitude_voluntary` parameters.
- The configuration file drives both the nominal model and the uncertainty sampling ranges for robustness analysis.
- This repository is organized as a research/simulation project rather than a packaged application, so the source directory is the primary execution context.

## Related documentation

- [src/README.md](src/README.md) describes the source tree in more detail.
- [docs/literature_review](docs/literature_review) contains background material and review artifacts.

## License

This project is licensed under the [LICENSE](LICENSE) file.