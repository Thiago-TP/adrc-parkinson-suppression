import yaml

from control_strategies import adrc, open_loop, pid
from system import ModelParameters


def main(
    num_simulations: int,
    amplitude_voluntary: float = 1.0
) -> None:
    """
    Main function to run and persist simulations.

    Parameters
    ----------
    num_simulations : int
        The number of simulations to run.
        The first run is always the nominal model, and remaining runs are
        for models with parameters sampled from the specified intervals.
        A properly formatted configs.yaml file is required to specify
        nominal parameters and stiffness uncertainty intervals.
    amplitude_voluntary : float, optional
        Amplitude of the voluntary torque profile. Defaults to 1.0.
    """

    # Load configurations
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    # Load nominal model parameters
    parameters = ModelParameters(**cfgs["parameters"])

    # Load initial conditions
    ic = tuple(cfgs["initial_conditions"].values())

    # Shared arguments across control strategies
    _shared = {
        "params": parameters,
        "ic": ic,
        "amplitude_voluntary": amplitude_voluntary
    }

    # Run nominal model with different control strategies
    no_control = open_loop.OpenLoopControl(
        name="open_loop",
        **_shared
    )
    pid_control = pid.PIDControl(
        name="pid",
        **_shared
    )
    adr_control = adrc.ADRControl(
        name="adrc",
        **_shared
    )

    print("\nRunning nominal model simulations...")

    controls = [
        adr_control,
        pid_control,
        no_control
    ]

    for control in controls:
        control.simulate_system()

    # Remaining runs with parameters sampled from uniform intervals
    print(
        f"\nRunning {num_simulations - 1} "
        "non-nominal model simulations with parameter sampling..."
    )
    for _ in range(num_simulations - 1):
        for control in controls:
            # Constant random seed -> per-control resample is valid
            control.resample_stiffness()
            control.simulate_system()

    # Save results across runs to npz files in results folder
    adr_control.save_results()
    pid_control.save_results()
    no_control.save_results()


if __name__ == "__main__":
    main(
        num_simulations=1,
        amplitude_voluntary=0.0
    )
    main(
        num_simulations=1,
        amplitude_voluntary=1.0
    )
