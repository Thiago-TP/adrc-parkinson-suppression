from matplotlib import pyplot as plt
import numpy as np
from system import System


class Plots:

    def __init__(self, system: System, 
                 xlim: tuple[float, float] = (0, 5), 
                 ylim: tuple[float, float] = (-100, 100)):
        self.s = system
        self.xlim = xlim
        self.ylim = ylim

    def plot_torque_profiles(self, save_results: bool = True):
        tau_v = np.array([self.s.tau_v(t) for t in self.s.t])
        tau_i = np.array([self.s.tau_i(t) for t in self.s.t])

        plt.figure()
        fig, axs = plt.subplots(
            nrows=4, ncols=1, sharex=True, sharey=True, figsize=(10, 8))

        fig.suptitle("Torque profiles")

        axs[0].set_title("Voluntary")
        axs[0].plot(self.s.t, tau_v, label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("Torque [Nm]")
        axs[0].grid()
        axs[0].legend()

        axs[1].set_title("Involuntary (PD tremor)")
        axs[1].plot(self.s.t, tau_i[:, 0], color="tab:blue")
        axs[1].set_ylabel("Torque [Nm]")
        axs[1].grid()

        axs[2].plot(self.s.t, tau_i[:, 1], color="tab:orange")
        axs[2].set_ylabel("Torque [Nm]")
        axs[2].grid()

        axs[3].plot(self.s.t, tau_i[:, 2], color="tab:green")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Torque [Nm]")
        axs[3].grid()

        if save_results:
            plt.savefig(
                f"results/torque_profiles_{self.s.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    def plot_time_response(self, save_results: bool = True):
        # Convert from radians to degrees
        theta_t = np.array(self.s.theta_true) * 180 / np.pi  
        theta = np.array(self.s.theta) * 180 / np.pi
        theta_f = np.array(self.s.theta_filtered) * 180 / np.pi
        theta_v = np.array(self.s.theta_v) * 180 / np.pi

        plt.figure()
        fig, axs = plt.subplots(
            nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 8))

        axs[0].plot(self.s.t, theta[:, 0], color="tab:blue", label="Upper arm")
        axs[0].plot(self.s.t, theta_f[:, 0], alpha=0.25, linestyle="--", color="tab:gray")
        axs[0].plot(self.s.t, theta_t[:, 0], alpha=0.25, linestyle=":", color="black")
        axs[0].set_title("Upper limb time response")
        axs[0].set_ylabel("$\\theta$ [˚]")
        axs[0].set_xlabel("")
        axs[0].set_xlim(*self.xlim)
        axs[0].set_ylim(*self.ylim)
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(self.s.t, theta[:, 1], color="tab:orange", label="Forearm")
        axs[1].plot(self.s.t, theta_f[:, 1], alpha=0.25, linestyle="--", color="tab:gray")
        axs[1].plot(self.s.t, theta_t[:, 1], alpha=0.25, linestyle=":", color="black")
        axs[1].set_ylabel("$\\theta$ [˚]")
        axs[1].set_xlabel("")
        axs[1].set_xlim(*self.xlim)
        axs[1].set_ylim(*self.ylim)
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(self.s.t, theta[:, 2], color="tab:green", label="$\\theta$")
        axs[2].plot(self.s.t, theta_f[:, 2], alpha=0.25, linestyle="--", color="tab:gray")
        axs[2].plot(self.s.t, theta_t[:, 2], alpha=0.25, linestyle=":", color="black")
        axs[2].plot(self.s.t, theta_v[:, 2], linestyle="--", color="tab:olive", label="$\\theta_v$")
        axs[2].set_ylabel("Palm angle [˚]")
        axs[2].set_xlabel("Time [s]")
        axs[2].set_xlim(*self.xlim)
        axs[2].set_ylim(*self.ylim)
        axs[2].legend()
        axs[2].grid()

        if save_results:
            plt.savefig(
                f"results/time_response_{self.s.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    def plot_voluntary_time_response(self, save_results: bool = True):
        # Plot actual and estimated voluntary time response

        plt.figure()
        fig, axs = plt.subplots(
            nrows=5, ncols=1, sharex=True, sharey=True, figsize=(8, 15))
        fig.suptitle("Upper Limb Time response")

        # Actual voluntary time response
        theta_v = np.array(self.s.theta_v) * 180 / np.pi  # convert radians to degrees
        axs[0].plot(self.s.t, theta_v, "--",
                    label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_title("Actual voluntary")
        axs[0].set_ylabel("$\\theta_v$ [˚]")
        axs[0].set_xlabel("")
        axs[0].set_xlim(*self.xlim)
        axs[0].set_ylim(*self.ylim)
        axs[0].legend()
        axs[0].grid()

        # Estimation from low-pass filtering
        theta_v_hat = np.array(self.s.theta_v_hat) * 180 / np.pi  # radians to degrees
        axs[1].plot(self.s.t, theta_v_hat, "--")
        axs[1].set_title("Estimated voluntary (low-pass filtered)")
        axs[1].set_ylabel("$\\hat{\\theta}_v$ [˚]")
        axs[1].set_xlabel("")
        axs[1].set_xlim(*self.xlim)
        axs[1].set_ylim(*self.ylim)
        axs[1].grid()

        # Errors
        error = theta_v - theta_v_hat
        axs[2].plot(self.s.t, error[:, 0], color="tab:blue")
        axs[2].set_title("Errors")
        axs[2].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[2].set_xlim(*self.xlim)
        axs[2].set_ylim(*self.ylim)
        axs[2].grid()

        axs[3].plot(self.s.t, error[:, 1], color="tab:orange")
        axs[3].set_title("")
        axs[3].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[3].set_xlim(*self.xlim)
        axs[3].set_ylim(*self.ylim)
        axs[3].grid()

        axs[4].plot(self.s.t, error[:, 2], color="tab:green")
        axs[4].set_title("")
        axs[4].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[4].set_xlim(*self.xlim)
        axs[4].set_ylim(*self.ylim)
        axs[4].grid()

        axs[4].set_xlabel("Time [s]")

        if save_results:
            plt.savefig(
                f"results/voluntary_time_response_{self.s.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    def plot_control(self, save_results: bool = True) -> None:
        # Plot control signal applied to wrist joint

        plt.figure()
        plt.plot(self.s.t, np.array(self.s.u)[:, 2], color="tab:green")
        plt.title("Control torque applied at wrist joint")
        plt.ylabel("$u_3$ [Nm]")
        plt.xlim(*self.xlim)
        plt.grid()

        plt.xlabel("Time [s]")

        if save_results:
            plt.savefig(
                f"results/control_{self.s.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

