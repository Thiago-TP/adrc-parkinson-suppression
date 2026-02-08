from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import final, Callable

import numpy as np
from matplotlib import pyplot as plt
import scipy


@dataclass(frozen=True)
class ModelParameters:
    # Lengths
    l1: float  # upper arm
    l2: float  # forearm
    l3: float  # palm

    # Centroids coefficients
    a1: float  # upper arm
    a2: float  # forearm
    a3: float  # palm

    # Mass
    m1: float  # upper arm
    m2: float  # forearm
    m3: float  # palm

    # Rotational inertia
    j1: float  # upper arm
    j2: float  # forearm
    j3: float  # palm

    # Rotational stiffness
    k1: float  # shoulder
    k2: float  # elbow
    k3: float  # biceps
    k4: float  # wrist

    # Rotational damper coefficients
    c1: float  # shoulder
    c2: float  # elbow
    c3: float  # biceps
    c4: float  # wrist


class Model(ABC):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
        t0: float = 0.0,
        t1: float = 5.0,
        dt: float = 1e-4,
    ) -> None:

        # Model name
        self.name = name

        # Define matrices variables
        self.j: np.ndarray = None  # inertia
        self.k: np.ndarray = None  # stiffness
        self.c: np.ndarray = None  # damping

        # Define vector and scalar variables
        self.tau_v: Callable[[float], np.ndarray] = None  # voluntary torque
        self.tau_i: Callable[[float], np.ndarray] = None  # involuntary torque (tremor)
        self.u: list[np.ndarray] = []  # control signal history
        self.theta: np.ndarray = None  # time response vector history
        self.theta_filtered: np.ndarray = None  # filtered time response vector history
        self.theta_voluntary: np.ndarray = (
            None  # voluntary time response vector history
        )
        self.theta_fft: np.ndarray = None  # frequency response vector history

        # Time and frequency parameters
        self.t: np.ndarray = np.arange(t0, t1 + dt, dt)
        n = len(self.t)
        self.f: np.ndarray = np.array([i / (dt * n) for i in range(n // 2 + 1)])
        self.initial_conditions: tuple[float] = ic  # initial conditions

        # Load model parameters to fill matrices
        self.update_model_parameters(params)

        # Load torque profiles
        self.load_torque_profiles()

        return

    def time_response(self, save_results: bool = True, voluntary_only: bool = False):

        # Define matrices for the state-space representation of the system
        null = np.zeros((3, 3))  # Zero matrix
        iden = np.identity(3)  # Identity matrix
        inv_j = scipy.linalg.inv(self.j)  # Inverse of the mass matrix

        k_hat = -inv_j @ self.k
        c_hat = -inv_j @ self.c

        a_num = np.concatenate((null, iden), axis=1)
        a_den = np.concatenate((k_hat, c_hat), axis=1)

        a = np.concatenate((a_num, a_den), axis=0)  # state matrix
        b = np.concatenate((null, inv_j), axis=0)  # input matrix

        # State derivative
        def x_dot(x: np.ndarray, t: float) -> np.ndarray:
            tau = self.tau_v(t) + (not voluntary_only) * (
                self.tau_i(t) + self.control(t)
            )
            return (a @ x) + (b @ tau)

        # Time response via ODE integration
        x = scipy.integrate.odeint(x_dot, self.initial_conditions, self.t)

        # Extract theta values (first 3 elements of each row)
        if not voluntary_only:
            self.theta = x[:, :3]
            if save_results:
                np.save(f"results/theta_{self.name}.npy", self.theta)
                np.save(f"results/u_{self.name}.npy", self.u)
        else:
            self.theta_voluntary = x[:, :3]
            if save_results:
                np.save(
                    f"results/theta_voluntary_{self.name}.npy", self.theta_voluntary
                )

    def low_pass_filter(self, f_cutoff: float) -> np.ndarray:
        # Design a Butterworth low-pass filter
        fs = 1 / (self.t[1] - self.t[0])  # sampling frequency in Hz
        b, a = scipy.signal.butter(
            N=4, Wn=2 * np.pi * f_cutoff, fs=fs, btype="low", analog=False
        )

        # Apply the filter to each column of theta
        return scipy.signal.filtfilt(b, a, self.theta, axis=0)

    def estimate_voluntary_time_response(self, save_results: bool = True):
        # Estimate voluntary time response by subtracting involuntary from total response
        if self.theta is None:
            raise ValueError(
                "Time response must be computed before estimating voluntary response."
            )

        self.theta_filtered = self.low_pass_filter(f_cutoff=1.0)  # cutoff in Hz

        if save_results:
            np.save(f"results/theta_filtered_{self.name}.npy", self.theta_filtered)

    def simulate_voluntary_time_response(self, save_results: bool = True):
        # Simulate voluntary time response by integrating the system with only voluntary torque
        self.time_response(save_results=save_results, voluntary_only=True)

    def frequency_response(self, save_results: bool = True):

        # Discrete Fourier Transform of time response via FFT
        self.theta_fft = np.fft.fft(self.theta, axis=0) / len(self.f)

        if save_results:
            np.save(f"results/theta_fft_{self.name}.npy", self.theta_fft)

    @final
    def plot_torque_profiles(self, save_results: bool = True):
        tau_v = np.array([self.tau_v(t) for t in self.t])
        tau_i = np.array([self.tau_i(t) for t in self.t])

        plt.figure()
        fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(10, 8))

        fig.suptitle("Torque profiles")

        axs[0].set_title("Voluntary")
        axs[0].plot(self.t, tau_v, label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_xlabel("")
        axs[0].set_ylabel("Torque [Nm]")
        axs[0].grid()
        axs[0].legend()

        axs[1].set_title("Involuntary (PD tremor)")
        axs[1].plot(self.t, tau_i[:, 0], color="tab:blue")
        axs[1].set_ylabel("Torque [Nm]")
        axs[1].grid()

        axs[2].plot(self.t, tau_i[:, 1], color="tab:orange")
        axs[2].set_ylabel("Torque [Nm]")
        axs[2].grid()

        axs[3].plot(self.t, tau_i[:, 2], color="tab:green")
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("Torque [Nm]")
        axs[3].grid()

        if save_results:
            plt.savefig(
                f"results/torque_profiles_{self.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    @final
    def plot_control_signal(self, save_results: bool = True):
        pass

    @final
    def plot_voluntary_time_response(self, save_results: bool = True):
        # Plot actual voluntary time response and low-pass filtered time response

        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(8, 15))
        fig.suptitle("Upper Limb Time response")

        # Actual voluntary time response
        theta_v = self.theta_voluntary * 180 / np.pi  # convert from radians to degrees
        axs[0].plot(self.t, theta_v, "--", label=["Upper arm", "Forearm", "Palm"])
        axs[0].set_title("Actual Voluntary")
        axs[0].set_ylabel("$\\theta_v$ [˚]")
        axs[0].set_xlabel("")
        axs[0].set_xlim(0, 5)
        axs[0].legend()
        axs[0].grid()

        # Estimation from low-pass filtering
        theta_f = self.theta_filtered * 180 / np.pi  # convert from radians to degrees
        axs[1].plot(self.t, theta_f, "--")
        axs[1].set_title("Estimated Voluntary (Low-pass filtered)")
        axs[1].set_ylabel("$\\hat{\\theta}_v$ [˚]")
        axs[1].set_xlabel("")
        axs[1].set_xlim(0, 5)
        axs[1].grid()

        # Errors
        error = theta_v - theta_f
        axs[2].plot(self.t, error[:, 0], color="tab:blue")
        axs[2].set_title("Errors")
        axs[2].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[2].set_xlim(0, 5)
        axs[2].grid()

        axs[3].plot(self.t, error[:, 1], color="tab:orange")
        axs[3].set_title("")
        axs[3].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[3].set_xlim(0, 5)
        axs[3].grid()

        axs[4].plot(self.t, error[:, 2], color="tab:green")
        axs[4].set_title("")
        axs[4].set_ylabel("$\\theta_v-\\hat{\\theta}_v$ [˚]")
        axs[4].set_xlim(0, 5)
        axs[4].grid()

        axs[4].set_xlabel("Time [s]")

        if save_results:
            plt.savefig(
                f"results/voluntary_time_response_{self.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    @final
    def plot_time_response(self, save_results: bool = True):
        theta = self.theta * 180 / np.pi  # Convert from radians to degrees
        plt.figure()
        plt.plot(self.t, theta, "--", label=["Upper arm", "Forearm", "Palm"])
        plt.title("Upper limb time response")
        plt.ylabel("$\\theta$ [˚]")
        plt.xlabel("Time [s]")
        plt.xlim(0, 5)
        plt.legend()
        plt.grid()

        if save_results:
            plt.savefig(
                f"results/time_response_{self.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    @final
    def plot_frequency_response(self, save_results: bool = True):
        # Convert from radians to degrees
        theta_fft = abs(self.theta_fft[: len(self.f), :]) * 180 / np.pi
        plt.figure()
        plt.semilogy(self.f, theta_fft, label=["Upper arm", "Forearm", "Palm"])
        plt.title("Upper limb frequency response")
        plt.ylabel("Amplitude [°]")
        plt.xlabel("Frequency [Hz]")
        plt.legend()
        plt.grid()

        if save_results:
            plt.savefig(
                f"results/frequency_response_{self.name}.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
            )
        plt.close()

    @final
    def update_model_parameters(self, p: ModelParameters) -> None:
        a1 = p.a1 * p.l1
        a2 = p.a2 * p.l2
        a3 = p.a3 * p.l3

        c1 = p.c1 * p.k1
        c2 = p.c2 * p.k2
        c3 = p.c3 * p.k3
        c4 = p.c4 * p.k4

        j11 = (
            (p.j1 + p.m1 * a1**2)
            + (p.j2 + p.m2 * a2**2)
            + p.m2 * p.l1**2
            + p.j3
            + p.m3 * (p.l1**2 + p.l2**2 + a3**2 + 2 * p.l2 * a3)
        )
        j12 = (p.j2 + p.m2 * a2**2) + p.j3 + p.m3 * (p.l2**2 + a3**2 + 2 * p.l2 * a3)
        j13 = p.j3 + p.m3 * (a3**2 + p.l2 * a3)

        j21 = j12
        j22 = j12
        j23 = j13

        j31 = j13
        j32 = j13
        j33 = p.j3 + p.m3 * a3**2

        self.j = np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])
        self.k = np.array(
            [
                [p.k1 + p.k3, p.k3, 0],
                [p.k3, p.k2 + p.k3, 0],
                [0, 0, p.k4],
            ]
        )
        self.c = np.array(
            [
                [c1 + c3, c3, 0],
                [c3, c2 + c3, 0],
                [0, 0, c4],
            ]
        )

    def load_torque_profiles(self) -> None:
        # Placeholders for now
        self.tau_v = lambda t: np.array(
            [
                np.cos(2 * np.pi * 0.1 * t),
                np.cos(2 * np.pi * 0.2 * t),
                np.cos(2 * np.pi * 0.3 * t),
            ]
        )
        self.tau_i = lambda t: np.array(
            [
                np.cos(2 * np.pi * 3.58803 * t),
                np.cos(2 * np.pi * 5.30097 * t),
                np.cos(2 * np.pi * 14.34746 * t),
            ]
        )
        return

    @abstractmethod
    def control(self, t: float) -> np.ndarray:
        pass

    @final
    def add_noise(self):
        pass
