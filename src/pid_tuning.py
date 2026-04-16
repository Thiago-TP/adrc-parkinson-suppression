import warnings
import yaml
import numpy as np
from scipy.optimize import differential_evolution

from control_strategies import pid
from system import ModelParameters
from postprocessing.metrics import _compute_metrics


def objective_function(gains: list[float], parameters: ModelParameters,
                       ic: tuple, amplitude_voluntary: float) -> float:
    """
    Cost function for differential evolution optimization.
    """
    kp, ki, kd = gains

    # 1. Instaciate the PID system with the current gains
    pid_system = pid.PIDControl(
        name="pid_de_eval",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        manual=True,
        kp=kp,
        ki=ki,
        kd=kd,
        pid_tuning=True,
    )

    # 2. Execute the simulation, ignoring overflow warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pid_system.simulate_system()

        # 3. Retrieve the results of the current simulation
        run_payload = pid_system.results.get("nominal_run")

        if run_payload is None:
            # Return a very high cost if the simulation failed
            return 1e12

        # 4. Check wheter the gains led to NaN or Inf values in the results, which indicates instability
        theta = run_payload["theta"]
        if np.isnan(theta).any() or np.isinf(theta).any():
            # Return a very high cost to penalize unstable solutions
            return 1e12

        # 5. Calculate the cost based on the performance metrics (ISE, IAE, ITAE, ITSE)
        try:
            metrics = _compute_metrics(
                run_payload=run_payload, baseline_payload=None)
            cost = metrics["ise"] + metrics["iae"] + \
                metrics["itae"] + metrics["itse"]

            # Ensure that the cost is a finite number; if not, penalize it
            if np.isnan(cost) or np.isinf(cost):
                return 1e12

            return cost

        except Exception:
            # If any error occurs during metric computation,
            # return a high cost to penalize this solution
            return 1e12


def main(amplitude_voluntary: float = 1.0) -> None:
    """
    Main function to run the PID tuning using Differential Evolution.
    """

    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    parameters = ModelParameters(**cfgs["parameters"])
    ic = tuple(cfgs["initial_conditions"].values())

    print(
        f"Initializing PID tuning optimization with amplitude_voluntary={amplitude_voluntary}...")

    # Bounds for Kp, Ki, Kd (you can adjust these based on expected ranges)
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

    # Optimize the PID gains using Differential Evolution
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(parameters, ic, amplitude_voluntary),
        strategy='best1bin',
        maxiter=10,    # Max. number of generations (iterations)
        popsize=1,     # Population scale factor
        disp=True,     # Print progress messages
        tol=1e-3,      # Convergence tolerance
        polish=False,  # Disable final polishing step (optional)
        seed=42        # Set a random seed for reproducibility
    )

    # Extract the best gains found
    best_kp, best_ki, best_kd = result.x

    print("\n" + "="*40)
    print("PID Tuning Optimization Completed!")
    print("="*40)
    print(f"Best Kp: {best_kp:.4f}")
    print(f"Best Ki: {best_ki:.4f}")
    print(f"Best Kd: {best_kd:.4f}")
    print(f"Minimum Cost (Sum of Errors): {result.fun:.4f}")
    print("="*40)

    # Optional: Run a final simulation with the best parameters and save the results
    print("\nRunning final simulation with the optimized parameters to save the results...")
    best_pid = pid.PIDControl(
        name="pid_optimized",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        manual=True,
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
    )

    # Save the results of the final simulation for post-processing
    best_pid.simulate_system()
    best_pid.save_results()


if __name__ == "__main__":
    main(amplitude_voluntary=1.0)

    # from postprocessing.postprocess import generate_plots, generate_metrics_tables
    # generate_plots(run_key="nominal_run")
