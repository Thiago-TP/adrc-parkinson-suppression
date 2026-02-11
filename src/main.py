import yaml
from model import ModelParameters
from open_loop import OpenLoopModel
from plots import Plots


def main() -> None:
    # Load general configurations
    with open("configs.yaml") as f:
        cfgs = yaml.safe_load(f)

    # Load nominal model parameters
    nominal_model = ModelParameters(**cfgs["nominal_parameters"])

    # Load initial conditions
    ic = tuple(cfgs["initial_conditions"].values())

    # Run model in open loop, i.e. no controls
    ol_model = OpenLoopModel("open_loop", nominal_model, ic)
    ol_model.load_torque_profiles()
    ol_model.time_response()
    ol_model.estimate_voluntary()
    ol_model.simulate_voluntary()

    ol_plots = Plots(ol_model)
    ol_plots.plot_time_response()
    ol_plots.plot_torque_profiles()
    ol_plots.plot_voluntary_time_response()


if __name__ == "__main__":
    main()
