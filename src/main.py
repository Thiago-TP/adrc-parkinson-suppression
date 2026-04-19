import yaml

from control_strategies import (
    afe_notch,
    eadrc_ebmflc,
    eadrc_zplp,
    pi_gallego,
    pid,
    uncontrolled,
)
from system import ModelParameters


def main(
    num_simulations: int,
    amplitude_voluntary: float = 1.0,
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
        Amplitude of the voluntary torque profile.
        Defaults to 1.0 (the more interesting case).
    """

    # Load configurations
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    # Load nominal model parameters
    parameters = ModelParameters(**cfgs["parameters"])

    # Load initial conditions
    ic = tuple(cfgs["initial_conditions"].values())

    # Instantiate control strategies
    # with same model parameters and initial conditions
    afe_notch_control = afe_notch.AFE_NotchControl(
        name="afe_notch",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary
    )
    eadr_ebmflc_control = eadrc_ebmflc.EADRC_EBMFLC(
        name="eadrc_ebmflc",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary
    )
    eadr_zplp_control = eadrc_zplp.EADRC_ZPLP(
        name="eadrc_zplp",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary
    )
    pi_gallego_control = pi_gallego.GallegoPIControl(
        name="pi_gallego",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary
    )
    pid_imc_control = pid.PIDControl(
        name="pid_imc",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        # Values below were found with slow_factor=5.0 on
        # the nominal model with amplitude_voluntary=1.0
        manual=True,
        kp=0.0998772,
        ki=98.7732779,
        kd=0.040496,
    )
    pid_de_control = pid.PIDControl(
        name="pid_de",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary,
        # Values below were found from src/pid_tuning.py
        manual=True,
        kp=1.2998816,
        ki=20.2188130,
        kd=3.2374438,
        perfect_tracking=True,
    )
    no_control = uncontrolled.Uncontrolled(
        name="uncontrolled",
        params=parameters,
        ic=ic,
        amplitude_voluntary=amplitude_voluntary
    )

    # Run nominal model simulations for selected control strategies
    print("\nRunning nominal model simulations...")
    controls = [
        afe_notch_control,
        eadr_ebmflc_control,
        eadr_zplp_control,
        pi_gallego_control,
        pid_de_control,
        pid_imc_control,
        no_control,
    ]
    for control in controls:
        control.simulate_system()

    # Run non-nominal model simulations with stiffness sampling
    # for selected control strategies
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
    for control in controls:
        control.save_results()


if __name__ == "__main__":

    import time

    __start = time.time()

    main(
        num_simulations=1,
        amplitude_voluntary=0.0
    )
    main(
        num_simulations=1,
        amplitude_voluntary=1.0
    )
    __stop = time.time()

    delta_s = __stop - __start
    delta_m = delta_s / 60
    print(f"\nAll finished in {delta_s:.2f}s ({delta_m:.2f} minutes)")
