# ADRC Parkinson's Tremor Suppression

A simulation framework for comparing control strategies for mitigating Parkinson's disease tremor in the human arm. This project implements three control approaches—open-loop, PID, and Active Disturbance Rejection Control (ADRC)—evaluated on a 3-DOF biomechanical arm model with parameter uncertainty.

## Overview

Parkinson's disease is characterized by involuntary tremors that significantly impact motor control and quality of life. This project provides a computational framework to evaluate and compare different control strategies for tremor suppression using a realistic human arm biomechanical model.

### Key Features

- **Three control strategies**: Open-loop baseline, classical PID control, and modern ADRC
- **Biomechanical arm model**: 3-DOF system (shoulder, elbow, wrist) with realistic parameters
- **Parameter uncertainty**: Stiffness intervals model inter-individual variability
- **"Monte Carlo" simulations**: Evaluate controller robustness across parameter variations
- **Comprehensive visualization**: Time response, control input, and comparative plots

## System Model

The system models the human arm as a three-degree-of-freedom mechanism:

| Joint | Description |
|:-----:|:-----------:|
| θ₁ | Shoulder angle |
| θ₂ | Elbow angle |
| θ₃ | Wrist angle |

### Model Parameters

The arm dynamics are governed by:
- **Geometric parameters**: Link lengths and centroid locations
- **Inertial properties**: Masses and rotational inertias
- **Stiffness**: Joint rotational stiffness (shoulder, elbow, biceps, wrist)
- **Damping**: Rotational damper coefficients at each joint

All nominal parameters and uncertainty intervals are defined in [configs.yaml](configs.yaml).

## Control Strategies

### 1. Open-Loop Control
A baseline strategy with no active disturbance rejection. Provides only voluntary torque input without feedback control.

### 2. PID Control
Classical proportional-integral-derivative control for trajectory tracking. Gain tuning is automated based on arm inertial properties.

### 3. ADRC (Active Disturbance Rejection Control)
A modern control approach featuring:
- Extended State Observer (ESO) for disturbance estimation
- Real-time compensation for model uncertainties and external disturbances
- Configuration parameter: `omega_c` (control bandwidth, default: 10 rad/s)

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adrc-parkinson-suppression.git
cd adrc-parkinson-suppression
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
```

### Running Simulations

Run simulations from the `src/` directory:

```bash
cd src
python main.py
```

**Main function parameters:**
- `num_simulations` (int): Total number of simulation runs (1 nominal + n-1 randomized)
- `plot_nominal` (bool): Whether to generate plots for nominal model results
- `amplitude_voluntary` (float): Amplitude of voluntary torque profile (default: 1.0)

**Example:**
```python
main(num_simulations=50, plot_nominal=True, amplitude_voluntary=1.5)
```

### Configuration

Edit [configs.yaml](configs.yaml) to modify:
- Nominal arm parameters (lengths, masses, inertias)
- Stiffness parameters and uncertainty intervals
- Initial conditions (joint angles and velocities)

## Project Structure

```
.
├── configs.yaml                          # Model parameters and initial conditions
├── src/
│   ├── main.py                           # Main simulation entry point
│   ├── system.py                         # Base system class and dynamics
│   ├── plots.py                          # Visualization utilities
│   ├── requirements.txt                  # Python dependencies
│   ├── control_strategies/               # Control implementations
│   │   ├── adrc.py                       # ADRC controller
│   │   ├── pid.py                        # PID controller
│   │   └── open_loop.py                  # Open-loop baseline
│   └── tremor_estimation_strategies/     # Literature review methods
├── results/                              # Simulation output (.npz files)
├── docs/                                 # Documentation
│   └── literature_review/                # Background research
└── LICENSE
```

## Output Files

Simulation results are saved as NumPy `.npz` files in the `results/` directory:
- `adrc_amplitude_{X}.npz`
- `pid_amplitude_{X}.npz`
- `open_loop_amplitude_{X}.npz`

Each file contains state trajectories and control inputs across all simulation runs.

## Dependencies

- **Python 3.7+**
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing (solving ODEs, linear algebra)
- **Matplotlib**: Data visualization
- **PyYAML**: Configuration file parsing

See [src/requirements.txt](src/requirements.txt) for exact versions.

## Usage Examples

### Basic Simulation
```python
from src.main import main

# Run 10 simulations with nominal model plots
main(num_simulations=10, plot_nominal=True)
```

### Custom Amplitude
```python
# Run 50 simulations with 1.5x amplitude
main(num_simulations=50, amplitude_voluntary=1.5)
```

## Results Interpretation

The simulations generate:
1. **Time response plots**: State trajectories (joint angles and velocities)
2. **Control inputs**: Torque profiles for each control strategy
3. **Comparison plots**: Performance metrics across all three controllers
4. **Monte Carlo results**: Robustness statistics across parameter variations

## References

This project builds upon control theory research in:
- Active Disturbance Rejection Control (ADRC)
- Biomechanical modeling of the human arm
- Tremor estimation and suppression techniques

See [docs/literature_review/](docs/literature_review/) for relevant papers and background research.

## License

This project is licensed under the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- Parameters are properly documented
- Simulations include validation against nominal models
- Results are saved with descriptive filenames

## Contact

For questions or suggestions, please open an issue on the repository.