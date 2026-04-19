import numpy as np
import scipy

from system import InitialConditions, ModelParameters, System


class EADRC_ZPLP(System):
    """
    Implements a standard Error-based Active Disturbance Rejection Control
    (EADRC) strategy for Parkinson's tremor suppression.
    Compensation is based on the estimation of voluntary motion,
    which is done using a zero-phase low-pass (ZPLP) Butterworth filter.
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float,
        omega_c: float = 10,
    ) -> None:
        super().__init__(name, params, ic,
                         amplitude_voluntary=amplitude_voluntary)

        # Set voluntary motion estimator
        self.butter_sos = scipy.signal.butter(
            N=1,
            Wn=5.0,
            fs=self.fs,
            btype="low",
            output="sos"
        )

        # Proportional, derivative gains
        self.kp = omega_c ** 2
        self.kd = 2 * omega_c

        # Error based Extended State Observer gains and states
        omega_o = 10 * omega_c
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
        # Zero-phase low-pass Butterworth filter to estimate voluntary response
        try:
            self.theta_v_hat = scipy.signal.sosfiltfilt(
                self.butter_sos, self.theta, axis=0,
            )
        except ValueError:
            self.theta_v_hat = self.theta.copy()

    def _reset_control_variables(self) -> None:
        # Reset EADRC states
        self.xe1_hat = 0.0
        self.xe2_hat = 0.0
        self.z = 0.0
