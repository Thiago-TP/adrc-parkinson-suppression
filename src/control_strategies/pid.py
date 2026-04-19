import numpy as np
import scipy

from system import InitialConditions, ModelParameters, System


class PIDControl(System):
    """
    Proportional-Integral-Derivative (PID) control strategy.
    The PID implemented may be adjusted manually
    by setting the kp, ki, and kd parameters,
    or automatically using Internal Model Control (IMC)
    tuning rules based on the system's physical parameters.
    For a Differential Evolution (DE)-based offline
    optimization of the PID gains, see src/pid_tuning.py.
    PID DE is assumed to be a best-case controller with perfect tracking
    of voluntary motion. For PID IMC, estimation of voluntary motion
    is done using a zero-phase low-pass Butterworth filter.

    Parameters
    ----------
    name: str
        Name of the control strategy.
    params: ModelParameters
        Model parameters for the system.
    ic: InitialConditions
        Initial conditions for the system.
    amplitude_voluntary: float, optional
        Amplitude of the voluntary motion.
    kp: float, optional
        Proportional gain for manual tuning, by default None.
    ki: float, optional
        Integral gain for manual tuning, by default None.
    kd: float, optional
        Derivative gain for manual tuning, by default None.
    manual: bool, optional
        If True, uses manual tuning with provided kp, ki, kd.
        If False, uses IMC tuning based on system parameters, by default False.
    slow_factor: float, optional
        Slow factor for IMC tuning, by default None.
        Higher values lead to slower response.
    perfect_tracking: bool, optional
        If True, assumes perfect tracking of voluntary motion (DE tuning only),
        by default False.
    """

    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        manual: bool = False,
        slow_factor: float | None = None,
        perfect_tracking: bool = False,
    ) -> None:
        super().__init__(name,
                         params,
                         ic,
                         amplitude_voluntary=amplitude_voluntary)

        # Set voluntary motion estimator
        self.butter_sos = scipy.signal.butter(
            N=1,
            Wn=5.0,
            fs=self.fs,
            btype="low",
            output="sos"
        )

        if manual:
            if kp is None or ki is None or kd is None:
                raise ValueError(
                    "For manual PID tuning, kp, ki, and kd must be provided."
                )
            # Proportional, integral, derivative gains
            self.kp = kp
            self.ki = ki
            self.kd = kd
        else:
            if slow_factor is None:
                raise ValueError(
                    "For IMC PID tuning, slow_factor must be provided."
                )
            # Compute the PID gains using IMC
            self._calculate_imc_pid_gains(slow_factor)

        self.perfect_tracking = perfect_tracking

        # Errors for calculating control
        self.error_control = 0.0
        self.error_sum = 0.0
        self.error_delta = 0.0
        self.error_previous = 0.0

        return

    def _calculate_imc_pid_gains(self, slow_factor: float = 5.0) -> None:
        """
        PID IMC Tuning:
        Detects if poles are real or complex and
        applies the appropriate synthesis formula for theta3.
        """
        # 1. Physical parameters from the pre-calculated matrices
        j33 = self.j[2, 2]
        k4 = self.k[2, 2]
        c4 = self.c[2, 2]

        # 2. System identification (Standard 2nd Order Form)
        # G(s) = K / (tau^2*s^2 + 2*zeta*tau*s + 1)
        k_sys = 1.0 / k4
        tau = np.sqrt(j33 / k4)
        zeta = c4 / (2 * np.sqrt(j33 * k4))

        # 3. Desired closed-loop speed (Lambda)
        lmbda = slow_factor * tau

        # 4. IMC logic selection based on damping ratio (zeta)
        if zeta >= 1.0:
            # --- REAL POLES CASE ---
            # Factorize into (tau1*s + 1)(tau2*s + 1)
            # Using the quadratic formula for the time constants
            discriminant = np.sqrt(zeta**2 - 1)
            tau1 = tau * (zeta + discriminant)
            tau2 = tau * (zeta - discriminant)

            # Use the slower pole for tuning
            lmbda = max(tau1, tau2) * slow_factor

            # PID tuning for real poles
            self.kp = (tau1 + tau2) / (lmbda * k_sys)
            ti = tau1 + tau2
            td = (tau1 * tau2) / (tau1 + tau2)

        else:
            # --- COMPLEX CONJUGATE POLES CASE ---
            # The controller must account for the oscillatory behavior
            # Kp formula for underdamped systems:
            self.kp = (2 * zeta * tau) / (lmbda * k_sys)
            ti = 2 * zeta * tau
            td = tau / (2 * zeta)

        # 5. Final scaling of integral and derivative gains
        self.ki = self.kp / ti
        self.kd = self.kp * td

    def _update_control(self, k: int) -> None:
        # PID control with fixed gains
        # For more details, check out
        # https://alphaville.github.io/qub/pid-101/#/

        self.error_control = self.theta_v_hat[k, 2] - self.theta[k, 2]
        self.error_delta = self.error_control - self.error_previous

        u3 = np.dot(
            [self.kp, self.ki, self.kd],
            [self.error_control,
             self.error_sum * self.dt,
             self.error_delta / self.dt],
        )

        self.u[k] = np.array([0.0, 0.0, u3])

        self.error_sum += self.error_control
        self.error_previous = self.error_control

    def _update_estimates(self, k: int) -> None:
        # During DE tuning, control has perfect tracking
        if self.perfect_tracking:
            self.theta_v_hat[k] = self.theta_v[k]
            return
        # Zero-phase low-pass Butterworth filter to estimate voluntary response
        try:
            self.theta_v_hat = scipy.signal.sosfiltfilt(
                self.butter_sos, self.theta, axis=0,
            )
        except ValueError:
            self.theta_v_hat = self.theta.copy()

    def _reset_control_variables(self) -> None:
        # Reset errors
        self.error_control = 0.0
        self.error_sum = 0.0
        self.error_delta = 0.0
        self.error_previous = 0.0
        return
