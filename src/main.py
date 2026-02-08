from model import ModelParameters
from open_loop import OpenLoopModel
import yaml


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
    ol_model.plot_torque_profiles()
    ol_model.time_response()
    ol_model.plot_time_response()
    ol_model.estimate_voluntary_time_response()
    ol_model.simulate_voluntary_time_response()
    ol_model.plot_voluntary_time_response()

    ol_model.frequency_response()
    ol_model.plot_frequency_response()


if __name__ == "__main__":
    main()
