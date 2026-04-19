import numpy as np

from system import InitialConditions, ModelParameters, System


class EADRC_EBMFLC(System):
    """
    EADRC-based PD control for Parkinson's tremor suppression,
    using the EBMFLC algorithm for tremor/voluntary motion estimation.

    Based on the work of Qi et al. (2024):
    P. Qi, J. Yang, Y. Dai, H. Zhang and J. Tong,
    "A Novel Approach to Parkinson's Tremor Suppression:
    E-BMFLC and LADRC Integration,"
    in IEEE Access, vol. 12, pp. 145871-145880, 2024,
    doi: 10.1109/ACCESS.2024.3424395.

    Control strategy compensates for tremor using a PD controller based on
    the estimation of voluntary motion,
    which is done using the EBMFLC algorithm.
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float,
        omega_c: float = 20,  # 50 in the paper
        n: int = 100,  # not given in the paper
        mu: float = 0.005,  # not given in the paper
        total_bandwidth: tuple[float, float] = (0, 12),
        voluntary_bandwidth: tuple[float, float] = (0, 4),
        tremor_bandwidth: tuple[float, float] = (4, 12),
        window_time: float = 0.59,  # a.k.a. Tp
        minimum_impact: float = 0.02  # a.k.a. alpha
    ) -> None:
        super().__init__(name, params, ic,
                         amplitude_voluntary=amplitude_voluntary)

        # Set voluntary motion estimator
        self.n = n
        self.mu = mu

        self.w_m = np.zeros(2 * self.n)

        f_min, f_max = total_bandwidth
        freqs = f_min + np.arange(n) * (f_max - f_min) / (n - 1)

        f_min_vol, f_max_vol = voluntary_bandwidth
        self.voluntary_indices = np.where(
            (f_min_vol <= freqs) &
            (freqs <= f_max_vol)
        )[0].tolist()

        f_min_trem, f_max_trem = tremor_bandwidth
        self.tremor_indices = np.where(
            (freqs >= f_min_trem) &
            (freqs <= f_max_trem)
        )[0].tolist()

        self.ws = 2 * np.pi * freqs

        self.d = self.fs * window_time  # fs = 160 Hz in the paper
        self.p = max(0, min(1, minimum_impact)) ** (1 / self.d)

        # Proportional, derivative gains
        self.kp = omega_c ** 2
        self.kd = 2 * omega_c

        # Error based Extended State Observer gains and states
        omega_o = 4 * omega_c
        self.lambda1 = 3 * omega_o
        self.lambda2 = 3 * (omega_o ** 2)
        self.lambda3 = omega_o ** 3
        self.xe1_hat = 0.0
        self.xe2_hat = 0.0
        self.z = 0.0
        self.b0 = np.linalg.inv(self.j)[2][2]

        return

    def _update_control(self, k: int) -> None:

        # Last control output
        u3_old = self.u[k - 1, 2]

        # Tracking error
        xe1 = self.theta_v_hat[k, 2] - self.theta[k, 2]  # error on theta3
        delta_xe1 = xe1 - self.xe1_hat

        # Extended State Observer of error
        dxe1_hat = self.xe2_hat + self.lambda1 * delta_xe1
        dxe2_hat = - self.b0 * u3_old \
            + self.z \
            + self.lambda2 * delta_xe1
        dz = self.lambda3 * delta_xe1

        self.xe1_hat += (dxe1_hat * self.dt)
        self.xe2_hat += (dxe2_hat * self.dt)
        self.z += (dz * self.dt)

        # Control (PD)
        u3_adrc = (self.kp * self.xe1_hat) + (self.kd * self.xe2_hat) + self.z
        u3 = u3_adrc / self.b0

        # Update of control history
        self.u[k] = np.array([0.0, 0.0, u3])

    def _update_estimates(self, k: int) -> None:
        # Harmonics of total motion, voluntary and tremor components
        x_m = np.concatenate([
            np.sin(self.ws * k * self.dt),
            np.cos(self.ws * k * self.dt)
        ])
        x_v = x_m[self.voluntary_indices * 2]
        x_t = x_m[self.tremor_indices * 2]

        # Estimate of motion
        m_hat = self.w_m.T @ x_m

        # Estimates of voluntary and tremor components
        # Indices are doubled (concatenated) to include both harmonics
        w_v = self.w_m[self.voluntary_indices * 2]
        w_t = self.w_m[self.tremor_indices * 2]
        v_hat = w_v.T @ x_v
        t_hat = w_t.T @ x_t

        # Update of estimates histories
        self.theta_v_hat[k, 2] = v_hat
        self.theta_i_hat[k, 2] = t_hat

        # Motion weights update
        error_m = self.theta[k, 2] - m_hat
        self.w_m = self.p * self.w_m + 2 * self.mu * error_m * x_m

    def _reset_control_variables(self) -> None:
        # Reset EADRC states, EBMFLC weights
        self.w_m = np.zeros(2 * self.n)
        self.xe1_hat = 0.0
        self.xe2_hat = 0.0
        self.z = 0.0
