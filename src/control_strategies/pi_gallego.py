import numpy as np
import scipy

from system import InitialConditions, ModelParameters, System


class GallegoPIControl(System):
    """
    PI control for Parkinson's tremor suppression,
    based on the work of Gallego et al. (2013):

    Gallego, J.Á., Rocon, E., Belda-Lois, J.M. et al.
    A neuroprosthesis for tremor management
    through the control of muscle co-contraction.
    J NeuroEngineering Rehabil 10, 36 (2013).
    https://doi.org/10.1186/1743-0003-10-36

    Control strategy is based on the estimation of tremor.
    The algorithm used by the authors is a slight adaptation
    of the CDF-WFLC-KF described in their 2010 paper (eqs. 14-17, 22-23):

    Gallego, J. A. and Rocon, E. and Roa, J. O. and
    Moreno, J. C. and Pons, J. L. (2010).
    "Real-Time Estimation of Pathological Tremor Parameters
    from Gyroscope Data".
    Sensors, 10(3), 2129-2149.
    https://doi.org/10.3390/s100302129
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float,
        # Controller parameters
        kp: float = 0.0987727,
        ki: float = 98.7727,
        tr_th: float = 0.0,
        tvr_th: float = 0.0,
        th_int_reset: float = 0.0,
        th_int_gain: float = 0.0,
        # CDF parameters
        theta_cdf: float = 0.99,
        # WFLC parameters
        f0: float = 12.0,
        n: int = 1,
        mu: float = 0.01,
        mu_0: float = 0.001,
        mu_bias: float = 0.01,
        # KF parameters
        cov_process: float = 1.0,
        cov_measurement: float = 0.01,
    ) -> None:
        super().__init__(name,
                         params,
                         ic,
                         amplitude_voluntary=amplitude_voluntary)

        # CDF setup
        self.theta_cdf = theta_cdf
        self.alpha = 1 - (self.theta_cdf ** 2)
        self.beta = (1 - self.theta_cdf) ** 2
        self._x = 0.0  # voluntary position estimate
        self._v = 0.0  # voluntary velocity estimate

        # WFLC setup
        self.f0 = f0
        self.w0 = 2 * np.pi * self.f0
        self.n = n
        self.mu = mu
        self.mu_0 = mu_0
        self.w = np.zeros(2 * self.n)
        self.mu_bias = mu_bias
        self.w0_sum = self.w0

        # KF setup (see kalman_filter.py)
        self.F = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [np.cos(self.w0_sum * self.dt), np.sin(self.w0_sum * self.dt), 0]
        ])
        self.H = np.array([[0, 0, 1]])
        self.Q = cov_process * np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ])  # only update position and velocity
        self.R = np.array([[cov_measurement]])
        self.x_kf = np.zeros((3, 1))
        self.P = 1e-3 * np.eye(3)

        # Set controller parameters
        self.kp = kp or 0.0
        self.ki = ki or 0.0
        self.tremor_sum = 0.0  # for integral term
        self.tr_th = tr_th
        self.tvr_th = tvr_th
        self.th_int_reset = th_int_reset
        self.th_int_gain = th_int_gain

        return

    def _update_control(self, k: int) -> None:

        # Tremor onset first check: tremor amplitude
        if abs(self.theta_i_hat[k, 2]) < self.tr_th:
            self.u[k] = np.array([0.0, 0.0, 0.0])
            return

        # Tremor-Voluntary Ratio:
        # integral of the amplitude spectrum in 3-12Hz (tremor)
        # divided by integral in 0-3Hz (voluntary motion)
        amplitude_spectrum = np.abs(np.fft.rfft(self.theta[:k + 1, 2]))
        freqs = np.fft.rfftfreq(k + 1, d=self.dt)
        tr = scipy.integrate.trapezoid(
            amplitude_spectrum[(freqs >= 3) & (freqs <= 12)]
        )
        vol = scipy.integrate.trapezoid(
            amplitude_spectrum[(freqs >= 0) & (freqs <= 3)]
        )
        tvr = tr / (vol + 1e-6)

        # If TVR is below threshold, do not apply control
        if tvr < self.tvr_th:
            self.u[k] = np.array([0.0, 0.0, 0.0])
            return

        # If inferred tremor amplitude is very low,
        # apply only proportional control
        if abs(self.theta_i_hat[k, 2]) < self.th_int_gain:
            gain_i = 0.0
        else:
            gain_i = self.ki

        # If tremor amplitude is too small, reset the integral term
        if abs(self.theta_i_hat[k, 2]) < self.th_int_reset:
            self.tremor_sum = 0.0
        else:
            self.tremor_sum += self.theta_i_hat[k, 2]

        u3 = np.dot(
            [self.kp, gain_i],
            [self.theta_i_hat[k, 2],
             self.tremor_sum * self.dt],
        )

        self.u[k] = np.array([0.0, 0.0, u3])

    def _update_estimates(self, k: int) -> None:
        s = self.theta[k, 2]

        # CDF step: remove voluntary motion from signal
        _r = s - self._x
        self._x += self.alpha * _r
        self._v += (self.beta / self.dt) * _r
        self._x += (self._v * self.dt)

        # WFLC step: estimate tremor frequency
        tr = s - self._x  # CDF-based tremor estimate
        r = np.arange(self.n) + 1
        x = np.concatenate([
            np.sin(self.w0_sum * r * self.dt),
            np.cos(self.w0_sum * r * self.dt)
        ])
        tr_hat = self.w.T @ x
        error = tr - tr_hat - self.mu_bias
        self.w += 2 * self.mu * error * x
        self.w0 += 2 * self.mu_0 * error * sum(
            r * (self.w[r - 1] * x[r + self.n - 1] -
                 self.w[r + self.n - 1] * x[r - 1])
        )
        self.w0_sum += self.w0

        # KF step: tremor amplitude refinement
        self.F[2, 0] = np.cos(self.w0_sum * self.dt)
        self.F[2, 1] = np.sin(self.w0_sum * self.dt)
        x_pred = self.F @ self.x_kf
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = tr - (self.H @ x_pred).item()
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x_kf = x_pred + K * y
        self.P = (np.eye(3) - K @ self.H) @ P_pred
        z = self.H @ self.x_kf

        # Update estimates
        self.theta_i_hat[k, 2] = z.item()
        self.theta_v_hat[k, 2] = self.theta[k, 2] - self.theta_i_hat[k, 2]

    def _reset_control_variables(self) -> None:
        # Reset CDF estimates
        self._x = 0.0  # voluntary position estimate
        self._v = 0.0  # voluntary velocity estimate

        # Reset WFLC weights, frequencies
        self.w0 = 2 * np.pi * self.f0
        self.w = np.zeros(2 * self.n)
        self.w0_sum = self.w0

        # Reset KF matrices, states
        self.F = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [np.cos(self.w0_sum * self.dt), np.sin(self.w0_sum * self.dt), 0]
        ])
        self.x_kf = np.zeros((3, 1))
        self.P = 1e-3 * np.eye(3)

        # Reset PI error integral
        self.tremor_sum = 0.0
