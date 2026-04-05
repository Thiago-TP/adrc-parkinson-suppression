from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import final, Callable

import scipy
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(42)))


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


InitialConditions = tuple[float, float, float, float, float, float]


class System(ABC):
    """
    Implementation of system dynamics, including state-space representation,
    noise injection and filtering, and placeholders for control strategies.

    Parameters
    ----------
    name: str
        Name of the system, used for saving results.
    params: ModelParameters
        Model parameters to be used in the system dynamics.
    ic: InitialConditions
        Initial conditions for the state variables
        (theta and theta_dot for each joint).
    t0: float, optional
        Initial time for the simulation in seconds, by default 0.0.
    t1: float, optional
        Final time for the simulation in seconds, by default 6.0.
    dt: float, optional
        Time step for the simulation in seconds, by default 1e-3.
    noise_var: float, optional
        Variance of the measurement noise, by default 4 * np.pi / 180 (4°).
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        t0: float = 0.0,
        t1: float = 6.0,
        dt: float = 1e-3,
        noise_var: float = 4 * np.pi / 180,
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
        self.theta_true: list[np.ndarray] | None = None
        self.theta: list[np.ndarray] | None = None

        # Voluntary portion of the time response (actual and estimated)
        self.theta_v: list[np.ndarray] | None = None
        self.theta_v_hat: list[np.ndarray] | None = None

        # Time vector and initial conditions
        self.dt: float = dt
        self.fs = 1 / self.dt
        self.t: np.ndarray = np.arange(t0, t1 + self.dt, self.dt)
        self.initial_conditions: InitialConditions = ic

        # Noise injection/filtering parameters
        self.noise_var: float = noise_var
        self.alpha: list[float] = [1.0]

        # Load model parameters to fill matrices
        self.update_model_parameters(params)

        # Set voluntary motion estimator
        self.butter_sos = scipy.signal.butter(
            N=1,
            Wn=5.0,
            fs=self.fs,
            btype="low",
            output="sos"
        )

        return

    def load_torque_profiles(self) -> None:
        # Placeholders for now
        self.tau_v = lambda t: 1.0 * np.array(
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

        # State dynamics
        def f_vol(t, x): return self.a @ x + self.b @ self.tau_v(t)
        def f_all(t, x, u): return f_vol(t, x) + self.b @ (self.tau_i(t) + u)

        # Initializations
        x = np.array(self.initial_conditions)
        x_v = np.array(self.initial_conditions)
        self.x_hat = [x]
        self.theta_true = [self.c_ss @ x]
        self.theta = [self.add_noise(self.theta_true[-1])]
        self.theta_filtered = [self.theta_true[-1]]
        self.theta_v = [self.c_ss @ x_v]
        self.theta_v_hat = [self.theta_filtered[-1]]

        # 4th order Runge-Kutta with fixed time step
        for t in self.t[1:]:

            u = self.control()

            # Update k1 through k4 (Measured response)
            k1 = f_all(t, x, u)
            k2 = f_all(t + (self.dt / 2), x + (self.dt * k1 / 2), u)
            k3 = f_all(t + (self.dt / 2), x + (self.dt * k2 / 2), u)
            k4 = f_all(t + (self.dt), x + (self.dt * k3), u)

            # Update state
            x += (self.dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

            # Update true response
            self.theta_true.append(self.c_ss @ x)

            # Update measured response
            self.theta.append(self.add_noise(self.theta_true[-1]))

            # Mitigate measurement noise
            self.theta_filtered.append(self.adaptive_filter(self.theta[-1]))

            # Update estimation of voluntary response
            self.theta_v_hat = self._estimate_voluntary()

            # Update estimation of state
            theta_dot = np.diff(self.theta, axis=0)
            self.x_hat.append(
                np.concat([self.theta[-1], theta_dot[-1]], axis=None)
            )

            # Repeat Runge-Kutta process to obtain true voluntary response
            k1 = f_vol(t, x_v)
            k2 = f_vol(t + (self.dt / 2), x_v + (self.dt * k1 / 2))
            k3 = f_vol(t + (self.dt / 2), x_v + (self.dt * k2 / 2))
            k4 = f_vol(t + (self.dt), x_v + (self.dt * k3))

            # Update voluntary state
            x_v += (self.dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)

            # Update true voluntary response
            self.theta_v.append(self.c_ss @ x_v)

        return

    def _estimate_voluntary(self) -> list[np.ndarray]:
        # Apply the filter to each column of theta
        try:
            return scipy.signal.sosfiltfilt(
                self.butter_sos, self.theta, axis=0,
            )
        except ValueError:
            return self.theta_filtered

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
    def add_noise(self, theta: np.ndarray) -> np.ndarray:
        noise = rs.normal(0.0, self.noise_var, size=theta.shape)
        return theta + noise

    @final
    def adaptive_filter(self, measured_theta: np.ndarray) -> np.ndarray:
        # Calculate innovation, i.e. error
        innovation = measured_theta - self.theta_filtered[-1]

        # Calculate alpha only relative to wrist angle
        alpha = scipy.special.erf(
            abs(innovation[2]) / (2 * np.sqrt(2 * self.noise_var)))
        self.alpha.append(alpha)

        # Return filtered measurement
        return self.theta_filtered[-1] + self.alpha[-1] * innovation

    @abstractmethod
    def control(self) -> np.ndarray:
        pass
