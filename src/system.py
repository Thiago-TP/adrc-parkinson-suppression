import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import blosc
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(42)))


@dataclass(frozen=False)
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

    # Stiffness uncertainty intervals
    stiffness_intervals: dict[str, tuple[float, float]]


InitialConditions = tuple[float, float, float, float, float, float]
RunResult = dict[str, float | np.ndarray | ModelParameters]


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
    amplitude_voluntary: float, optional
        Amplitude of the voluntary torque profile, by default 1.0.
    savedir: str, optional
        Directory where results will be saved, by default "results/runs".
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        t0: float = 0.0,
        t1: float = 6.0,
        dt: float = 1e-3,
        amplitude_voluntary: float = 1.0,
        savedir: str = "results/runs"
    ) -> None:

        # Model name
        self.name: str = name
        self.params: ModelParameters = params
        self.ic: InitialConditions = ic
        self.t0: float = t0
        self.t1: float = t1
        self.dt: float = dt
        self.amplitude_voluntary: float = amplitude_voluntary
        self.savedir: str = savedir

        # Time vector and sampling frequency
        self.fs: float = 1 / self.dt  # useful for many tremor estimators
        self.t: np.ndarray = np.arange(
            self.t0,
            self.t1 + self.dt,
            self.dt
        )

        # Control signal history
        self.u: np.ndarray = np.zeros((len(self.t), 3))

        # States from state space representation (for use in Runge-Kutta)
        self.x: np.ndarray = np.zeros((len(self.t), 6))
        self.x_v: np.ndarray = np.zeros((len(self.t), 6))

        # Time response:
        # observation, true voluntary, estimated voluntary,
        # true tremor, estimated tremor
        self.theta: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_v: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_v_hat: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_i: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_i_hat: np.ndarray = np.zeros((len(self.t), 3))

        # Model dynamics parameters
        self.j: np.ndarray = np.zeros((3, 3))  # inertia
        self.k: np.ndarray = np.zeros((3, 3))  # stiffness
        self.c: np.ndarray = np.zeros((3, 3))  # damping

        # State space matrices
        self.a: np.ndarray = np.zeros((6, 6))  # state matrix
        self.b: np.ndarray = np.zeros((6, 3))  # input matrix
        self.c_ss: np.ndarray = np.zeros((3, 6))  # output matrix

        # Load model parameters to fill matrices
        self._set_model()

        # Initializations of simulation-relevant attributes:
        self.u[0] = np.array([0.0, 0.0, 0.0])
        self.x[0] = np.array(self.ic)
        self.x_v[0] = np.array(self.ic)
        self.theta[0] = self.c_ss @ self.x[0]
        self.theta_v[0] = self.c_ss @ self.x_v[0]
        self.theta_v_hat[0] = self.theta[0]
        self.theta_i[0] = np.zeros(3)
        self.theta_i_hat[0] = np.zeros(3)

        # Results storage across runs
        self.suffix: str = f"{self.name}_amplitude_{self.amplitude_voluntary}"
        self.results: dict[str, RunResult] = {}

        return

    # Torque profiles (voluntary and involuntary)
    @final
    def _tau_v(self, t: float) -> np.ndarray:
        return self.amplitude_voluntary * np.array([
            np.cos(2 * np.pi * 0.1 * t),
            np.cos(2 * np.pi * 0.2 * t),
            np.cos(2 * np.pi * 0.3 * t),
        ])

    @final
    @staticmethod
    def _tau_i(t: float) -> np.ndarray:
        return np.array([
            np.cos(2 * np.pi * 3.58803 * t),
            np.cos(2 * np.pi * 5.30097 * t),
            np.cos(2 * np.pi * 14.34746 * t),
        ])

    @final
    def simulate_system(self) -> None:

        print(f"Simulating system {self.name}...")
        __start = time.time()

        # State dynamics
        def f_vol(t, x): return self.a @ x + self.b @ self._tau_v(t)
        def f_all(t, x, u): return f_vol(t, x) + self.b @ (self._tau_i(t) + u)

        # 4th order Runge-Kutta with fixed time step
        for k, t in enumerate(self.t[1:], start=1):

            # Update k1 through k4 (Measured response)
            k1 = f_all(t, self.x[k-1], self.u[k-1])
            k2 = f_all(t + (self.dt / 2), self.x[k-1] + (self.dt * k1 / 2), self.u[k-1])  # noqa: E501
            k3 = f_all(t + (self.dt / 2), self.x[k-1] + (self.dt * k2 / 2), self.u[k-1])  # noqa: E501
            k4 = f_all(t + (self.dt), self.x[k-1] + (self.dt * k3), self.u[k-1])  # noqa: E501

            # Update state
            update = (self.dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
            self.x[k] = self.x[k - 1] + update

            # Update response
            self.theta[k] = self.c_ss @ self.x[k]

            # Update estimation of voluntary/tremor response
            self._update_estimates(k)

            # Update control signal
            self._update_control(k)

            # Repeat Runge-Kutta process to obtain true voluntary response
            k1 = f_vol(t, self.x_v[k-1])
            k2 = f_vol(t + (self.dt / 2), self.x_v[k-1] + (self.dt * k1 / 2))
            k3 = f_vol(t + (self.dt / 2), self.x_v[k-1] + (self.dt * k2 / 2))
            k4 = f_vol(t + (self.dt), self.x_v[k-1] + (self.dt * k3))

            # Update voluntary state
            update_v = (self.dt / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
            self.x_v[k] = self.x_v[k - 1] + update_v

            # Update true voluntary response
            self.theta_v[k] = self.c_ss @ self.x_v[k]

        __end = time.time()
        print(f"Run took {__end - __start:.2f} s.")

        # Store run results
        if self.results == {}:
            key = "nominal_run"
        else:
            key = f"non_nominal_run_{len(self.results)}"

        self.results[key]: RunResult = {
            "time": self.t,
            "theta": self.theta,
            "theta_v": self.theta_v,
            "theta_v_hat": self.theta_v_hat,
            "theta_i": self.theta_i,
            "theta_i_hat": self.theta_i_hat,
            "u": self.u,
            "tau_v": np.array([self._tau_v(t) for t in self.t]),
            "tau_i": np.array([self._tau_i(t) for t in self.t]),
            "amplitude_voluntary": self.amplitude_voluntary,
            "state_matrix": self.a,
            "input_matrix": self.b,
        }

        return

    @final
    def save_results(self) -> None:
        """
        Dumps simulation results across runs to a data file in savedir.
        Overwrites file if data already existed.
        """
        os.makedirs(self.savedir, exist_ok=True)
        path = f"{self.savedir}/{self.suffix}.data"
        with open(path, "wb") as f:
            pickled_data = pickle.dumps(self.results)
            compressed_pickle = blosc.compress(pickled_data, typesize=8)
            f.write(compressed_pickle)
        print(f"Results saved to {path}.")
        return

    @final
    def _set_model(self) -> None:
        self._set_dynamics()
        self._set_state_space()

    @final
    def _set_dynamics(self) -> None:
        p = self.params  # shorthand for readability

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
    def resample_stiffness(self) -> None:
        """
        Resamples stiffness parameters from their uncertainty intervals
        and updates model (matrices K and C).
        """
        print("Resampling stiffness parameters...")
        self.params.k1 = rs.uniform(*self.params.stiffness_intervals["k1"])
        self.params.k2 = rs.uniform(*self.params.stiffness_intervals["k2"])
        self.params.k3 = rs.uniform(*self.params.stiffness_intervals["k3"])
        self.params.k4 = rs.uniform(*self.params.stiffness_intervals["k4"])
        self._set_model()

        # Restarts simulation-relevant attributes
        self.u: np.ndarray = np.zeros((len(self.t), 3))
        self.x: np.ndarray = np.zeros((len(self.t), 6))
        self.x_v: np.ndarray = np.zeros((len(self.t), 6))
        self.theta: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_v: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_v_hat: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_i: np.ndarray = np.zeros((len(self.t), 3))
        self.theta_i_hat: np.ndarray = np.zeros((len(self.t), 3))

        return

    @abstractmethod
    def _update_control(self, k: int) -> None:
        """
        Update control signal self.u[k] based on
        the control strategy implemented in each subclass.

        Parameters
        ----------
        k: int
            Index of the current time step in the simulation.
        """
        pass

    @abstractmethod
    def _update_estimates(self, k: int) -> None:
        """
        Estimate voluntary and/or tremor portion of the response.
        Estimators change between control strategies and
        must be implemented in each subclass.

        Parameters
        ----------
        k: int
            Index of the current time step in the simulation.
        """
        pass
