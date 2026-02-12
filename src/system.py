from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import final, Callable

import numpy as np
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


class System(ABC):
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

        # Model dynamics parameters
        self.j: np.ndarray | None = None  # inertia
        self.k: np.ndarray | None = None  # stiffness
        self.c: np.ndarray | None = None  # damping

        # State space matrices
        self.a: np.ndarray | None = None  # state matrix
        self.b: np.ndarray | None = None  # input matrix
        self.c_ss: np.ndarray | None = None  # output matrix

        # Torque profiles (voluntary and involuntary)
        self.tau_v: Callable[[float], np.ndarray] | None = None
        self.tau_i: Callable[[float], np.ndarray] | None = None

        # Control signal history
        self.u: list[np.ndarray] = [np.array([0.0, 0.0, 0.0])]

        # Time response
        self.theta: np.ndarray | None = None

        # Voluntary portion of the time response (actual and estimated)
        self.theta_v: np.ndarray | None = None
        self.theta_v_hat: np.ndarray | None = None

        # Time vector and initial conditions
        self.dt: float = dt
        self.t: np.ndarray = np.arange(t0, t1 + self.dt, self.dt)
        self.initial_conditions: tuple[float] = ic

        # Load model parameters to fill matrices
        self.update_model_parameters(params)

        return

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

    def simulate_system(self) -> None:

        # print(self.a)
        # print(x)
        # print(self.initial_conditions)

        # State dynamics
        def f_vol(t, x): return self.a @ x + self.b @ self.tau_v(t)
        def f_all(t, x, u): return f_vol(t, x) + self.b @ (self.tau_i(t) + u)

        # Initialization
        x = np.array(self.initial_conditions)
        x_v = np.array(self.initial_conditions)
        self.theta = [self.c_ss @ x]
        self.theta_v = [self.c_ss @ x_v]

        # 4th order Runge-Kutta with fixed time step
        for t in self.t[:-1]:
            u = self.control() # implicitly updates self.u

            # Update k1 through k4 (Measured response)
            k1 = f_all(t, x, u)
            k2 = f_all(t + (self.dt / 2), x + (self.dt * k1 / 2), u)
            k3 = f_all(t + (self.dt / 2), x + (self.dt * k2 / 2), u)
            k4 = f_all(t + (self.dt), x + (self.dt * k3), u)

            # Update state, response
            x += (self.dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
            self.theta.append(self.c_ss @ x)

            # Update k1 through k4 (Voluntary trajectory)
            k1 = f_vol(t, x_v)
            k2 = f_vol(t + (self.dt / 2), x_v + (self.dt * k1 / 2))
            k3 = f_vol(t + (self.dt / 2), x_v + (self.dt * k2 / 2))
            k4 = f_vol(t + (self.dt), x_v + (self.dt * k3))

            # Update state, response
            x_v += (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.theta_v.append(self.c_ss @ x_v)

        # Cast to numpy array
        self.theta = np.array(self.theta)
        self.theta_v = np.array(self.theta_v)
        self.u = np.array(self.u)
        return

    def estimate_voluntary(
            self, f_cutoff: float = 1.0):
        # Estimate voluntary time response using low-pass filtering
        if self.theta is None:
            raise ValueError(
                "No time response found, please run a simulation."
            )

        # Design a Butterworth low-pass filter
        fs = 1 / self.dt  # sampling frequency in Hz
        b, a = scipy.signal.butter(
            N=4, Wn=2 * np.pi * f_cutoff, fs=fs, btype="low", analog=False
        )

        # Apply the filter to each column of theta
        self.theta_v_hat = scipy.signal.filtfilt(b, a, self.theta, axis=0)

    @final
    def update_model_parameters(self, p: ModelParameters) -> None:
        self._set_dynamics(p)
        self._set_state_space()

    @final
    def _set_dynamics(self, p: ModelParameters) -> None:
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
        j12 = (p.j2 + p.m2 * a2**2) + p.j3 + p.m3 * \
            (p.l2**2 + a3**2 + 2 * p.l2 * a3)
        j13 = p.j3 + p.m3 * (a3**2 + p.l2 * a3)

        j21 = j12
        j22 = j12
        j23 = j13

        j31 = j13
        j32 = j13
        j33 = p.j3 + p.m3 * a3**2

        self.j = np.array([
            [j11, j12, j13], 
            [j21, j22, j23], 
            [j31, j32, j33]
        ])
        self.k = np.array([
            [p.k1 + p.k3, p.k3, 0],
            [p.k3, p.k2 + p.k3, 0],
            [0, 0, p.k4],
        ])
        self.c = np.array([
            [c1 + c3, c3, 0],
            [c3, c2 + c3, 0],
            [0, 0, c4],
        ])

    @final
    def _set_state_space(self) -> None:
        # Define matrices for the state-space representation of the system
        null = np.zeros((3, 3))  # Zero matrix
        iden = np.identity(3)  # Identity matrix
        inv_j = np.linalg.inv(self.j)  # Inverse of the mass matrix

        k_hat = -inv_j @ self.k
        c_hat = -inv_j @ self.c

        a_num = np.concatenate((null, iden), axis=1)
        a_den = np.concatenate((k_hat, c_hat), axis=1)

        self.a = np.concatenate((a_num, a_den), axis=0)  # state matrix
        self.b = np.concatenate((null, inv_j), axis=0)  # input matrix
        self.c_ss = np.concatenate((iden, null), axis=1)  # output matrix

    @final
    def add_noise(self):
        pass
    
    @abstractmethod
    def control(self, t: float) -> np.ndarray:
        pass

