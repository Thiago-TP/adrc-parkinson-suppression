import os
import pickle

import blosc
import numpy as np
import scipy
from matplotlib import pyplot as plt

# Change if LaTeX is not available in the system
# (or if you want to use a different font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm, amsmath}",
    "font.size": 20,
})

COLORS = {
    "afe_notch": "#0077BB",
    "eadrc_ebmflc": "#FF7F00",
    "eadrc_zplp": "#1CA1DA",
    "open_loop": "#D8C303",
    "pi_gallego": "#2E8B57",
    "pid_optimized": "#FF7F50",
    "pid": "#FF3867",
}

SAVEFIG_ARGS = {  # slightly better than tight_layout
    "pad_inches": 0.1,
    "bbox_inches": "tight",
}


class Plots:

    def __init__(self,
                 control_name: str,
                 time: np.ndarray,
                 theta: np.ndarray,
                 theta_baseline: np.ndarray,
                 theta_v: np.ndarray,
                 theta_v_hat: np.ndarray,
                 theta_i: np.ndarray,
                 theta_i_hat: np.ndarray,
                 u: np.ndarray,
                 tau_v: np.ndarray,
                 tau_i: np.ndarray,
                 amplitude_voluntary: float,
                 xlim: tuple[float, float] = (0.0, 6.0),
                 ylim: tuple[float, float] = (-80.0, 80.0),
                 flim: tuple[float, float] = (0.0, 20.0),
                 savedir: str = "results/plots",
                 **kwargs,  # add more data without changing the signature
                 ) -> None:
        self.control_name = control_name
        self.t = time
        self.theta = theta
        self.theta_baseline = theta_baseline
        self.theta_v = theta_v
        self.theta_v_hat = theta_v_hat
        self.theta_i = theta_i
        self.theta_i_hat = theta_i_hat
        self.u = u
        self.tau_v = tau_v
        self.tau_i = tau_i
        self.amplitude_voluntary = amplitude_voluntary
        self.xlim = xlim
        self.ylim = ylim
        self.flim = flim
        self.suffix = "_".join([
            self.control_name, "amplitude", str(self.amplitude_voluntary)
        ])
        self.savedir = f"{savedir}/{self.suffix}"
        self.fs = 1.0 / (self.t[1] - self.t[0])

        # Warning on frequency resolution if time vector is small
        f_res = 1 / (self.t[-1] - self.t[0])
        if f_res > 0.5:
            print(
                f"Warning: Frequency resolution is {f_res:.2f} Hz, "
                "which may be too coarse for meaningful frequency analysis."
                "Consider increasing duration to at least 10s."
            )

    def plot_torque_profiles(self):
        _, axs = plt.subplots(
            nrows=4, ncols=1, sharex=True, sharey=True, figsize=(10, 8)
        )

        axs[0].plot(self.t, self.tau_v[:, 0], color="black", label=r"$\tau_{v_1}$")  # noqa: E501
        axs[0].plot(self.t, self.tau_v[:, 1], "-.", color="black", label=r"$\tau_{v_2}$")  # noqa: E501
        axs[0].plot(self.t, self.tau_v[:, 2], "--", color="black", label=r"$\tau_{v_3}$")  # noqa: E501
        axs[0].set_xlabel("")
        axs[0].set_ylabel(r"$\bm{\tau}_v$ [Nm]")
        axs[0].grid()
        axs[0].legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1))

        axs[1].plot(self.t, self.tau_i[:, 0], color="black")
        axs[1].set_ylabel(r"$\tau_{i_1}$ [Nm]")
        axs[1].grid()

        axs[2].plot(self.t, self.tau_i[:, 1], color="black")
        axs[2].set_ylabel(r"$\tau_{i_2}$ [Nm]")
        axs[2].grid()

        axs[3].plot(self.t, self.tau_i[:, 2], color="black")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel(r"$\tau_{i_3}$ [Nm]")
        axs[3].grid()

        plt.xlim(*self.xlim)

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/torque_profiles_{self.suffix}.pdf",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("torque_profiles")
        plt.close()

    def plot_time_response(self):
        # Convert from radians to degrees
        theta = np.array(self.theta) * 180 / np.pi
        theta_v = np.array(self.theta_v) * 180 / np.pi
        theta_v3_hat = np.array(self.theta_v_hat) * 180 / np.pi

        plt.figure(figsize=(10, 3))

        plt.plot(self.t, theta[:, 2], color=COLORS.get(self.control_name, "black"), label=r"$\theta_3$")  # noqa: E501
        if self.control_name not in ["afe_notch", "open_loop"]:
            plt.plot(self.t, theta_v3_hat[:, 2], color="#BD1AEA", label=r"$\widehat{\theta}_{v_3}$")  # noqa: E501
        plt.plot(self.t, theta_v[:, 2], linestyle="--", color="black", label=r"$\theta_{v_3}$")  # noqa: E501
        plt.ylabel(r"Palm angle [\textdegree]")
        plt.xlabel("Time [s]")
        plt.xlim(*self.xlim)
        plt.ylim(*self.ylim)
        plt.legend(loc="upper right", ncols=3)
        plt.grid()

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/time_response_{self.suffix}.pdf",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("time_response")
        plt.close()

    def plot_control(self) -> None:
        # Plot control signal applied to wrist joint

        plt.figure(figsize=(10, 3))
        plt.plot(self.t, self.u[:, 2], color=COLORS.get(self.control_name, "black"))  # noqa: E501
        plt.ylabel(r"$u_3$ [Nm]")
        plt.xlim(*self.xlim)
        plt.grid()

        plt.xlabel("Time [s]")

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/control_{self.suffix}.pdf",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("control")
        plt.close()

    def plot_dft_response(self) -> None:
        # DFT magnitude of measured and voluntary components (theta_3)
        theta3 = self.theta[:, 2]
        theta3_base = self.theta_baseline[:, 2]
        theta_v3 = self.theta_v[:, 2]

        freqs = np.fft.rfftfreq(theta3.size, d=1.0 / self.fs)
        dft_theta3 = np.abs(np.fft.rfft(theta3))
        dft_theta3_base = np.abs(np.fft.rfft(theta3_base))
        dft_theta_v3 = np.abs(np.fft.rfft(theta_v3))

        plt.figure(figsize=(10, 3))
        if self.control_name != "open_loop":
            plt.semilogy(
                freqs,
                dft_theta3,
                color=COLORS.get(self.control_name, "black"),
                label=r"$\theta_3^\text{con.}$"
            )
        plt.semilogy(
            freqs,
            dft_theta3_base,
            color="#BD1AEA",
            label=r"$\theta_3^\text{unc.}$"
        )
        plt.semilogy(
            freqs,
            dft_theta_v3,
            linestyle="--",
            color="black",
            label=r"$\theta_{v_3}$"
        )
        plt.ylabel("DFT Magnitude")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(*self.flim)
        plt.legend(loc="lower right", ncols=3)
        plt.grid()

        # Inlay zoom on voluntary frequencies
        # [left, bottom, width, height]
        ax_inlay = plt.axes((0.125, -0.5, 0.25, 0.4))
        if self.control_name != "open_loop":
            ax_inlay.semilogy(
                freqs,
                dft_theta3,
                color=COLORS.get(self.control_name, "black"),
                label=r"$\theta_3^\text{con.}$"
            )
        ax_inlay.semilogy(
            freqs,
            dft_theta3_base,
            color="#BD1AEA",
            label=r"$\theta_3^\text{unc.}$"
        )
        ax_inlay.semilogy(
            freqs,
            dft_theta_v3,
            linestyle="--",
            color="black",
            label=r"$\theta_{v_3}$"
        )
        ax_inlay.set_xlim(0, 0.4)
        ax_inlay.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax_inlay.grid()

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/dft_response_{self.suffix}.pdf",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("dft_response")
        plt.close()

    def plot_psd_response(self) -> None:
        # PSD of measured and voluntary components (theta_3)
        theta3 = self.theta[:, 2]
        theta3_base = self.theta_baseline[:, 2]
        theta_v3 = self.theta_v[:, 2]

        nperseg = len(theta3)
        nfft = nperseg
        noverlap = nperseg // 2
        welch_args = {
            "fs": self.fs,
            "nfft": nfft,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "window": "hann",
        }
        freqs, psd_theta3 = scipy.signal.welch(theta3, **welch_args)
        _, psd_theta3_base = scipy.signal.welch(theta3_base, **welch_args)
        _, psd_theta_v3 = scipy.signal.welch(theta_v3, **welch_args)

        plt.figure(figsize=(10, 3))
        if self.control_name != "open_loop":
            plt.semilogy(
                freqs,
                psd_theta3,
                color=COLORS.get(self.control_name, "black"),
                label=r"$\theta_3^\text{con.}$"
            )
        plt.semilogy(
            freqs,
            psd_theta3_base,
            color="#BD1AEA",
            label=r"$\theta_3^\text{unc.}$"
        )
        plt.semilogy(
            freqs,
            psd_theta_v3,
            linestyle="--",
            color="black",
            label=r"$\theta_{v_3}$"
        )
        plt.ylabel("PSD [rad$^2$/Hz]")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(*self.flim)
        plt.legend(loc="lower right", ncols=3)
        plt.grid()

        # Inlay zoom on voluntary frequencies
        # [left, bottom, width, height]
        ax_inlay = plt.axes((0.125, -0.5, 0.25, 0.4))
        if self.control_name != "open_loop":
            ax_inlay.semilogy(
                freqs,
                psd_theta3,
                color=COLORS.get(self.control_name, "black"),
                label=r"$\theta_3^\text{con.}$"
            )
        ax_inlay.semilogy(
            freqs,
            psd_theta3_base,
            color="#BD1AEA",
            label=r"$\theta_3^\text{unc.}$"
        )
        ax_inlay.semilogy(
            freqs,
            psd_theta_v3,
            linestyle="--",
            color="black",
            label=r"$\theta_{v_3}$"
        )
        ax_inlay.set_xlim(0, 0.4)
        ax_inlay.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax_inlay.grid()

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/psd_response_{self.suffix}.pdf",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("psd_response")
        plt.close()

    def plot_spectrogram_response(self) -> None:
        # Time-frequency representation of measured response (theta_3)
        theta3 = self.theta[:, 2]

        spec_args = {
            "fs": self.fs,
            "nperseg": 1024,
            "noverlap": 512,
            "window": "hann",
            "nfft": 2048,
        }
        freq, time, spec = scipy.signal.spectrogram(theta3, **spec_args)

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(time, freq, spec, shading="gouraud", cmap="coolwarm")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.ylim(*self.flim)
        plt.xlim(0.5, self.xlim[1])  # skip initial transient (white strip)
        plt.colorbar(label="Power")

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/spectrogram_response_{self.suffix}.png",
            **SAVEFIG_ARGS
        )
        self._saved_plot_msg("spectrogram_response")
        plt.close()

    def _saved_plot_msg(self, plot_type: str) -> None:
        print(
            f"* Saved {plot_type} plot for "
            f"amplitude {self.amplitude_voluntary} "
            f"to {self.savedir} folder."
        )


def plot_from_data(
    data_path: str,
    baseline_path: str,
    control_name: str,
    run_key: str = "nominal_run",
) -> None:
    with open(data_path, "rb") as f:
        compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        data = pickle.loads(depressed_pickle)

    with open(baseline_path, "rb") as f:
        compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        data_baseline = pickle.loads(depressed_pickle)

    print("\nPlotting results in file:", data_path)

    plots = Plots(
        control_name=control_name,
        theta_baseline=data_baseline[run_key]["theta"],
        **data[run_key],
    )
    plots.plot_time_response()
    plots.plot_control()
    plots.plot_torque_profiles()
    plots.plot_dft_response()
    plots.plot_psd_response()
    plots.plot_spectrogram_response()
