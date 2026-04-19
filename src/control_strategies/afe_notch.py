import scipy

from system import InitialConditions, ModelParameters, System


class AFE_NotchControl(System):
    """
    AFE (Adaptive Frequency Estimator) with notch filtering
    for tremor suppression.
    Based on the work of Zamanian and Richer (2019):

    Amir Hosein Zamanian, Edmond Richer,
    Adaptive notch filter for pathological tremor suppression using permanent magnet linear motor,
    Mechatronics,
    Volume 63,
    2019,
    102273,
    ISSN 0957-4158,
    https://doi.org/10.1016/j.mechatronics.2019.102273.

    The proposed method filters the velocity signal first
    with a 6th order Butterworth high-pass filter then with a
    adaptive bandpass filter to estimate the tremor frequency.
    The control law is a notch filter that suppresses the tremor
    at the estimated frequency.
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float = 1.0,
        # AFE parameters
        zeta_f: float = 0.01,
        zeta_e: float = 4,
        gamma: float = 8,
        wt0: float = 8.0,
        # Control parameters
        zeta_1: float = 0.01,
        zeta_2: float = 0.01,
        b1: float = 2,
        b2: float = 2,
    ) -> None:
        super().__init__(name, params, ic,
                         amplitude_voluntary=amplitude_voluntary)

        # 6th order Butterworth high-pass filter
        # with a cutoff frequency of 2 Hz
        self.butter_sos = scipy.signal.butter(
            N=6,
            Wn=2,
            btype='hp',
            fs=self.fs,
            output='sos'
        )

        # AFE attributes: parameters and states
        self.wt0 = wt0
        self.zeta_f = zeta_f
        self.zeta_e = zeta_e
        self.gamma = gamma
        self.yf1 = 0.0
        self.yf2 = 0.0
        self.ye1 = 0.0
        self.ye2 = 0.0
        self.ye3 = self.wt0
        self.wt = self.ye3

        # Control attributes: parameters and states
        self.zeta_1 = zeta_1
        self.zeta_2 = zeta_2
        self.b1 = b1
        self.b2 = b2
        self.y1 = 0.0
        self.y2 = 0.0
        self.y3 = 0.0
        self.y4 = 0.0

        return

    def _update_control(self, k: int) -> None:

        # Tremor frequencies
        w1 = self.wt
        w2 = self.wt / 2

        # Velocity
        x_dot = (self.theta[k, 2] - self.theta[k - 1, 2]) / self.dt

        # State equations of the control law
        y1_dot = self.y2
        y2_dot = - (w1 ** 2) * self.y1 - (2 * self.zeta_1 * w1 * self.y2) + x_dot  # noqa: E501
        y3_dot = self.y4
        y4_dot = - (w2 ** 2) * self.y3 - (2 * self.zeta_2 * w2 * self.y4) + x_dot  # noqa: E501

        # State updates using Euler integration
        self.y1 += y1_dot * self.dt
        self.y2 += y2_dot * self.dt
        self.y3 += y3_dot * self.dt
        self.y4 += y4_dot * self.dt

        # Control update
        fo = (self.b1 * self.y1) + (self.b2 * self.y4)
        self.u[k, 2] = fo

    def _update_estimates(self, k: int) -> None:
        """AFE estimation of tremor."""

        # The voluntary motion components in the velocity signal are removed
        # by a 6th order Butterworth high-pass filter
        # with a cutoff frequency of 2 Hz.
        theta_dot = (self.theta[k, 2] - self.theta[k - 1, 2]) / self.dt
        x_dot_hp = scipy.signal.sosfilt(self.butter_sos, [theta_dot])[0]

        # State space equations of the bandpass filter
        yf1_dot = self.yf2
        yf2_dot = - (self.ye3 ** 2) * self.yf1 + (2 * self.zeta_f * self.ye3 * (x_dot_hp - self.yf2))  # noqa: E501

        # State updates using Euler integration
        self.yf1 += yf1_dot * self.dt
        self.yf2 += yf2_dot * self.dt

        # State space equations of the adaptive frequency estimator
        ye1_dot = self.ye2
        ye2_dot = - (self.ye3 ** 2) * self.ye1 + (2 * self.zeta_e * self.ye3 * (self.yf2 - self.ye2))  # noqa: E501
        ye3_dot = - self.gamma * self.ye1 * self.ye3 * (self.yf2 - self.ye2)

        # State updates using Euler integration
        self.ye1 += ye1_dot * self.dt
        self.ye2 += ye2_dot * self.dt
        self.ye3 += ye3_dot * self.dt

        # Estimated tremor frequency update
        self.wt = self.ye3

        return

    def _reset_control_variables(self) -> None:
        # Reset states, estimated frequency
        self.yf1 = 0.0
        self.yf2 = 0.0
        self.ye1 = 0.0
        self.ye2 = 0.0
        self.ye3 = self.wt0
        self.wt = self.ye3
        self.y1 = 0.0
        self.y2 = 0.0
        self.y3 = 0.0
        self.y4 = 0.0
