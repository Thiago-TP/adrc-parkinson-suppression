import warnings

import numpy as np
import yaml
from scipy.optimize import differential_evolution

from control_strategies import pid
from postprocessing.metrics import _compute_metrics
from system import InitialConditions, ModelParameters


def objective_function(
    gains: list[float],
    parameters: ModelParameters,
    ic: InitialConditions,
    amplitude_voluntary: float,
) -> float:
    """
    Cost function for differential evolution optimization.
    """
    kp, ki, kd = gains

    # 1. Instantiate the PID system with the current gains
    pid_system = pid.PIDControl(
        name="pid_de_eval",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        manual=True,
        kp=kp,
        ki=ki,
        kd=kd,
        perfect_tracking=True,
    )

    return evaluate_pid(pid_system)


def evaluate_pid(pid_system) -> float:
    # 2. Execute the simulation, ignoring overflow warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pid_system.simulate_system()

        # 3. Retrieve the results of the current simulation
        run_payload = pid_system.results.get("nominal_run")

        if run_payload is None:
            # Return a very high cost if the simulation failed
            print("Simulation failed: No results found for nominal run.")
            return 1e12

        # 4. Check whether the gains led to NaN or Inf values in the results,
        # which indicates instability
        theta = run_payload["theta"][:, 2]  # Focus on the third joint angle
        if np.isnan(theta).any() or np.isinf(theta).any():
            # Return a very high cost to penalize unstable solutions
            print("Simulation failed: NaN or Inf values detected in theta.")
            return 1e12

        # 5. Calculate the cost based on the performance metrics
        # (combination of one of more: ISE, IAE, ITAE, ITSE, TV, RMS, ISC, IAC)
        try:
            metrics = _compute_metrics(run_payload=run_payload, baseline_payload=None)
            cost = -metrics["r2"]

            # Ensure that the cost is a finite number; if not, penalize it
            if np.isnan(cost) or np.isinf(cost):
                print("Simulation failed: Invalid cost value.")
                return 1e12

            return cost

        except Exception as e:
            # If any error occurs during metric computation,
            # return a high cost to penalize this solution
            print(f"Simulation failed: Error during metric computation: {e}")
            return 1e12


def tune_perfect_tracker(amplitude_voluntary: float = 1.0) -> None:
    """
    Main function to run the PID tuning using Differential Evolution.
    """

    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    parameters = ModelParameters(**cfgs["parameters"])
    ic = tuple(cfgs["initial_conditions"].values())

    print(
        "Initializing PID tuning optimization "
        f"with amplitude_voluntary={amplitude_voluntary}..."
    )

    # Bounds for Kp, Ki, Kd (you can adjust these based on expected ranges)
    # For this problem, Kp and Kd are expected to be small, and Ki big.
    bounds = [(0.0, 5.0), (0.0, 100.0), (0.0, 5.0)]

    # Optimize the PID gains using Differential Evolution
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(parameters, ic, amplitude_voluntary),
        strategy="best1bin",
        maxiter=30,  # Max. number of generations (iterations)
        popsize=5,  # Population scale factor
        disp=True,  # Print progress messages
        tol=1e-3,  # Convergence tolerance
        polish=False,  # Disable final polishing step (optional)
        seed=42,  # Set a random seed for reproducibility
    )

    # Extract the best gains found
    best_kp, best_ki, best_kd = result.x

    print("\n" + "=" * 40)
    print("PID Tuning Optimization Completed!")
    print("=" * 40)
    print(f"Best Kp: {best_kp:.7f}")
    print(f"Best Ki: {best_ki:.7f}")
    print(f"Best Kd: {best_kd:.7f}")
    print(f"Minimum Cost (-R²): {result.fun:.7f}")
    print("=" * 40)


def tune_flawed_tracker(amplitude_voluntary: float = 1.0) -> None:
    """
    Function to run a manual grid search for PID tuning.
    Tuning is based on IMC, and controller does not enjoy perfect tracking.
    """

    grid_slow_factor = np.arange(0.1, 5.1, 0.1)
    results = np.zeros((len(grid_slow_factor), 2))  # columns: slow_factor, cost
    results[:, 0] = grid_slow_factor

    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    parameters = ModelParameters(**cfgs["parameters"])
    ic = tuple(cfgs["initial_conditions"].values())

    for i, sf in enumerate(grid_slow_factor):
        print(f"\nTuning PID IMC with slow_factor={sf:.3f}...")
        pid_imc_control = pid.PIDControl(
            name="pid_imc",
            params=parameters,
            ic=ic,
            amplitude_voluntary=amplitude_voluntary,
            manual=False,  # since we automate via IMC tuning rules
            slow_factor=sf,  # IMC slow factor
            perfect_tracking=False,
        )
        results[i, 1] = evaluate_pid(pid_imc_control)
        print(f"Cost for slow_factor={sf:.3f}: {results[i, 1]:.7f}")

    best_cost = np.min(results[:, 1])
    best_slow_factor = results[np.argmin(results[:, 1]), 0]
    best_pid_imc = pid.PIDControl(
        name="pid_imc_best",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        manual=False,
        slow_factor=best_slow_factor,
        perfect_tracking=False,
    )

    print("\n" + "=" * 40)
    print("PID Tuning Optimization Completed!")
    print("=" * 40)
    print(f"Best slow factor: {best_slow_factor:.7f}")
    print(f"Best Kp: {best_pid_imc.kp:.7f}")
    print(f"Best Ki: {best_pid_imc.ki:.7f}")
    print(f"Best Kd: {best_pid_imc.kd:.7f}")
    print(f"Minimum Cost (-R²): {best_cost:.7f}")
    print("=" * 40)


if __name__ == "__main__":
    import time

    __start = time.time()
    tune_perfect_tracker(amplitude_voluntary=0.0)
    tune_perfect_tracker(amplitude_voluntary=1.0)
    tune_flawed_tracker(amplitude_voluntary=0.0)
    tune_flawed_tracker(amplitude_voluntary=1.0)
    __stop = time.time()

    delta_s = __stop - __start
    delta_m = delta_s / 60
    print(f"\nAll finished in {delta_s:.2f}s ({delta_m:.2f} minutes)")
