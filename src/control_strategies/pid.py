import numpy as np

from system import InitialConditions, ModelParameters, System


class PIDControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float = 1.0,
        kp: float = 2.0,
        ki: float = 4.0,
        kd: float = 0.42,
        manual: bool = False,
        slow_factor: float = 5.0,
    ) -> None:
        super().__init__(name,
                         params,
                         ic,
                         amplitude_voluntary=amplitude_voluntary)
        self.kp, self.ki, self.kd = 0.0, 0.0, 0.0
        if manual:
            # Proportional, integral, derivative gains
            self.kp = kp
            self.ki = ki
            self.kd = kd
        else:
            if np.isclose(self.j, np.zeros((3, 3))).all():
                self._set_dynamics()
            # Compute the PID gains using IMC
            self._calculate_imc_pid_gains(slow_factor)

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

        # 5. Final discretization for the PID class
        # Convert continuous gains to the discrete variables used in control()
        self.ki = self.kp / ti
        self.kd = self.kp * td

    def _control(self) -> np.ndarray:
        # PID control with fixed gains
        # For more details, check out
        # https://alphaville.github.io/qub/pid-101/#/

        if self.theta_v_hat is None or self.theta_filtered is None:
            # If the system hasn't been simulated yet, return zero control
            raise ValueError(
                "System state is not initialized. "
                "Run simulate_system() before calling _control()."
            )

        self.error_control = self.theta_v_hat[-1][2] - \
            self.theta_filtered[-1][2]
        self.error_delta = self.error_control - self.error_previous

        u3 = np.dot(
            [self.kp, self.ki, self.kd],
            [self.error_control,
             self.error_sum * self.dt,
             self.error_delta / self.dt],
        )

        self.error_sum += self.error_control
        self.error_previous = self.error_control

        self.u.append(np.array([0.0, 0.0, u3]))

        return self.u[-1]
