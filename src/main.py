import yaml
from control_strategies import open_loop, pid, adrc
from plots import Plots
from system import ModelParameters


def main() -> None:
    # Load general configurations
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    # Load nominal model parameters
    nominal_model = ModelParameters(**cfgs["nominal_parameters"])

    # Load initial conditions
    ic = tuple(cfgs["initial_conditions"].values())

    # Run model with different control strategies
    ol = open_loop.OpenLoopControl("open_loop", nominal_model, ic)
    pidc = pid.PIDControl("pid", nominal_model, ic)
    adr = adrc.ADRControl("adrc", nominal_model, ic)

    for control in [
        ol, 
        pidc, 
        adr,
    ]:
        control.load_torque_profiles()
        control.simulate_system()

        plots = Plots(control)
        plots.plot_time_response()
        plots.plot_torque_profiles()
        plots.plot_voluntary_time_response()
        plots.plot_control()


if __name__ == "__main__":
    main()
