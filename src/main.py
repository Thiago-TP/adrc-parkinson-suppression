import yaml
from control_strategies import (
    open_loop,
    # pid,
    adrc
)
from plots import Plots
from system import ModelParameters


def main(
    num_simulations: int,
    plot_nominal: bool = False,
    amplitude_voluntary: float = 1.0
) -> None:
    """
    Main function to run the simulations and
    generate plots for different control strategies.

    Parameters
    ----------
    num_simulations : int
        The number of simulations to run.
        The first run is always the nominal model, and remaining runs are
        for models with parameters sampled from the specified intervals.
        A properly formatted configs.yaml file is required to specify
        nominal parameters and stiffness uncertainty intervals.
    plot_nominal : bool, optional
        Whether to plot the nominal model results, i.e. first run's results.
        Defaults to False.
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

    # Run nominal model with different control strategies
    no_control = open_loop.OpenLoopControl(
        "open_loop",
        parameters,
        ic,
        amplitude_voluntary=amplitude_voluntary
    )
    # pid_control = pid.PIDControl(
    #     "pid",
    #     parameters,
    #     ic,
    #     amplitude_voluntary=amplitude_voluntary
    # )
    adr_control = adrc.ADRControl(
        "adrc",
        parameters,
        ic,
        amplitude_voluntary=amplitude_voluntary
    )

    print("Running nominal model simulations...")

    for control in [
        adr_control,
        # pid_control,
        no_control,
    ]:
        control.simulate_system()

        if plot_nominal:
            plots = Plots(control)
            plots.plot_time_response()
            # plots.plot_voluntary_time_response()
            plots.plot__control()

    # Plot torques only once since they are the same for all control strategies
    if plot_nominal:
        plots.plot_torque_profiles()

    # Remaining runs with parameters sampled from uniform intervals
    print(
        f"\nRunning {num_simulations - 1} "
        "non-nominal model simulations with parameter sampling..."
    )
    for _ in range(num_simulations - 1):
        for control in [
            adr_control,
            # pid_control,
        ]:
            # Constant random seed -> per-control resample is valid
            control.resample_stiffness()
            control.simulate_system()

    # Save results across runs to npz files in results folder
    adr_control.save_results()
    # pid_control.save_results()


if __name__ == "__main__":
    main(
        num_simulations=100,
        plot_nominal=True,
        amplitude_voluntary=0.0
    )
    main(
        num_simulations=100,
        plot_nominal=False,
        amplitude_voluntary=1.0
    )
