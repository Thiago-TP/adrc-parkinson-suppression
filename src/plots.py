from matplotlib import pyplot as plt
import numpy as np
from model import Model


class Plots:

    def __init__(self, model: Model, xlim: tuple[float, float] = (0, 5)):
        self.m = model
        self.xlim = xlim

    def plot_torque_profiles(self, save_results: bool = True):
        tau_v = np.array([self.m.tau_v(t) for t in self.m.t])
        tau_i = np.array([self.m.tau_i(t) for t in self.m.t])

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=1, sharex=True, sharey=True, figsize=(10, 8))

        fig.suptitle("Torque profiles")

        axs[0].set_title("Voluntary")
        axs[0].plot(self.m.t, tau_v, label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("Torque [Nm]")
        axs[0].grid()
        axs[0].legend()

        axs[1].set_title("Involuntary (PD tremor)")
        axs[1].plot(self.m.t, tau_i[:, 0], color="tab:blue")
        axs[1].set_ylabel("Torque [Nm]")
        axs[1].grid()

        axs[2].plot(self.m.t, tau_i[:, 1], color="tab:orange")
        axs[2].set_ylabel("Torque [Nm]")
        axs[2].grid()

        axs[3].plot(self.m.t, tau_i[:, 2], color="tab:green")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Torque [Nm]")
        axs[3].grid()

        if save_results:
            plt.savefig(
                f"results/torque_profiles_{self.m.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    def plot_time_response(self, save_results: bool = True):
        theta = self.m.theta * 180 / np.pi  # Convert from radians to degrees
        plt.figure()
        fig, axs = plt.subplots(
            nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 8))

        axs[0].plot(self.m.t, theta[:, 0], color="tab:blue", label="Upper arm")
        axs[0].set_title("Upper limb time response")
        axs[0].set_ylabel("$\\theta$ [˚]")
        axs[0].set_xlabel("")
        axs[0].set_xlim(*self.xlim)
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(self.m.t, theta[:, 1], color="tab:orange", label="Forearm")
        axs[1].set_ylabel("$\\theta$ [˚]")
        axs[1].set_xlabel("")
        axs[1].set_xlim(*self.xlim)
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(self.m.t, theta[:, 2], color="tab:green", label="Palm")
        axs[2].set_ylabel("$\\theta$ [˚]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_xlim(*self.xlim)
        axs[2].legend()
        axs[2].grid()

        if save_results:
            plt.savefig(
                f"results/time_response_{self.m.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    def plot_voluntary_time_response(self, save_results: bool = True):
        # Plot actual and estimated voluntary time response

        fig, axs = plt.subplots(
            nrows=5, ncols=1, sharex=True, sharey=True, figsize=(8, 15))
        fig.suptitle("Upper Limb Time response")

        # Actual voluntary time response
        theta_v = self.m.theta_v * 180 / np.pi  # convert radians to degrees
        axs[0].plot(self.m.t, theta_v, "--",
                    label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_title("Actual Voluntary")
        axs[0].set_ylabel("$\\theta_v$ [˚]")
        axs[0].set_xlabel("")
        axs[0].set_xlim(*self.xlim)
        axs[0].legend()
        axs[0].grid()

        # Estimation from low-pass filtering
        theta_v_hat = self.m.theta_v_hat * 180 / np.pi  # radians to degrees
        axs[1].plot(self.m.t, theta_v_hat, "--")
        axs[1].set_title("Estimated Voluntary (Low-pass filtered)")
        axs[1].set_ylabel("$\\hat{\\theta}_v$ [˚]")
        axs[1].set_xlabel("")
        axs[1].set_xlim(*self.xlim)
        axs[1].grid()

        # Errors
        error = theta_v - theta_v_hat
        axs[2].plot(self.m.t, error[:, 0], color="tab:blue")
        axs[2].set_title("Errors")
        axs[2].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[2].set_xlim(*self.xlim)
        axs[2].grid()

        axs[3].plot(self.m.t, error[:, 1], color="tab:orange")
        axs[3].set_title("")
        axs[3].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[3].set_xlim(*self.xlim)
        axs[3].grid()

        axs[4].plot(self.m.t, error[:, 2], color="tab:green")
        axs[4].set_title("")
        axs[4].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[4].set_xlim(*self.xlim)
        axs[4].grid()

        axs[4].set_xlabel("Time [s]")

        if save_results:
            plt.savefig(
                f"results/voluntary_time_response_{self.m.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()
