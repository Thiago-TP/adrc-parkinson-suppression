import os
import numpy as np
import scipy
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm}",
    "font.size": 20,
})

COLORS = {
    "open_loop": "#D8C303",
    "pid": "#FF3867",
    "adrc": "#1CA1DA",
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
                 theta_v: np.ndarray,
                 theta_v_hat: np.ndarray,
                 u: np.ndarray,
                 tau_v: np.ndarray,
                 tau_i: np.ndarray,
                 amplitude_voluntary: float,
                 xlim: tuple[float, float] = (0, 6),
                 ylim: tuple[float, float] = (-80, 80),
                 savedir="results/figures"):
        self.control_name = control_name
        self.t = time
        self.theta = theta
        self.theta_v = theta_v
        self.theta_v_hat = theta_v_hat
        self.u = u
        self.tau_v = tau_v
        self.tau_i = tau_i
        self.amplitude_voluntary = amplitude_voluntary
        self.xlim = xlim
        self.ylim = ylim
        self.suffix = "_".join([
            self.control_name, "amplitude", str(self.amplitude_voluntary)
        ])
        self.savedir = f"{savedir}/{self.suffix}"
        self.fs = 1.0 / (self.t[1] - self.t[0])

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

        plt.plot(self.t, theta[:, 2], color=COLORS[self.control_name], label=r"$\theta_3$")  # noqa: E501
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
        plt.plot(self.t, self.u[:, 2], color=COLORS[self.control_name])
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
        theta_v3_hat = self.theta_v_hat[:, 2]
        theta_v3 = self.theta_v[:, 2]

        freqs = np.fft.rfftfreq(theta3.size, d=1.0 / self.fs)
        dft_theta3 = np.abs(np.fft.rfft(theta3))
        dft_theta_v3_hat = np.abs(np.fft.rfft(theta_v3_hat))
        dft_theta_v3 = np.abs(np.fft.rfft(theta_v3))

        plt.figure(figsize=(10, 3))
        plt.plot(freqs, dft_theta3, color=COLORS[self.control_name], label=r"$\theta_3$")  # noqa: E501
        plt.plot(freqs, dft_theta_v3_hat, color="#BD1AEA", label=r"$\widehat{\theta}_{v_3}$")  # noqa: E501
        plt.plot(freqs, dft_theta_v3, linestyle="--", color="black", label=r"$\theta_{v_3}$")  # noqa: E501
        plt.ylabel("DFT Magnitude")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, self.fs / 2)
        plt.legend(loc="upper right", ncols=3)
        plt.grid()

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
        theta_v3_hat = self.theta_v_hat[:, 2]
        theta_v3 = self.theta_v[:, 2]

        freqs, psd_theta3 = scipy.signal.welch(theta3, fs=self.fs)
        _, psd_theta_v3_hat = scipy.signal.welch(theta_v3_hat, fs=self.fs)
        _, psd_theta_v3 = scipy.signal.welch(theta_v3, fs=self.fs)

        plt.figure(figsize=(10, 3))
        plt.semilogy(freqs, psd_theta3, color=COLORS[self.control_name], label=r"$\theta_3$")  # noqa: E501
        plt.semilogy(freqs, psd_theta_v3_hat, color="#BD1AEA", label=r"$\widehat{\theta}_{v_3}$")  # noqa: E501
        plt.semilogy(freqs, psd_theta_v3, linestyle="--", color="black", label=r"$\theta_{v_3}$")  # noqa: E501
        plt.ylabel("PSD [rad$^2$/Hz]")
        plt.xlabel("Frequency [Hz]")
        plt.xlim(0, self.fs / 2)
        plt.legend(loc="upper right", ncols=3)
        plt.grid()

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

        freq, time, spec = scipy.signal.spectrogram(theta3, fs=self.fs)

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(time, freq, spec, shading="gouraud")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [s]")
        plt.ylim(0, self.fs / 2)
        plt.colorbar(label="Power")

        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(
            f"{self.savedir}/spectrogram_response_{self.suffix}.pdf",
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


def _run_data(
    npz_data: np.lib.npyio.NpzFile,
    run_key: str = "nominal_run",
) -> dict[str, np.ndarray | float]:
    if run_key not in npz_data.files:
        raise KeyError(
            f"Run key '{run_key}' was not found in the result file. "
            f"Available keys: {npz_data.files}"
        )

    run_payload = npz_data[run_key].item()
    return {
        "time": run_payload["time"],
        "theta": run_payload["theta"],
        "theta_v": run_payload["theta_v"],
        "theta_v_hat": run_payload["theta_v_hat"],
        "u": run_payload["u"],
        "tau_v": run_payload["tau_v"],
        "tau_i": run_payload["tau_i"],
        "amplitude_voluntary": float(run_payload["amplitude_voluntary"]),
    }


def plot_from_npz(
    npz_path: str,
    control_name: str,
    run_key: str = "nominal_run",
) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        run_data = _run_data(data, run_key)

    print("\nPlotting results in file:", npz_path)

    plots = Plots(
        control_name=control_name,
        time=run_data["time"],
        theta=run_data["theta"],
        theta_v=run_data["theta_v"],
        theta_v_hat=run_data["theta_v_hat"],
        u=run_data["u"],
        tau_v=run_data["tau_v"],
        tau_i=run_data["tau_i"],
        amplitude_voluntary=run_data["amplitude_voluntary"],
    )
    plots.plot_time_response()
    plots.plot_control()
    plots.plot_torque_profiles()
    plots.plot_dft_response()
    plots.plot_psd_response()
    plots.plot_spectrogram_response()
